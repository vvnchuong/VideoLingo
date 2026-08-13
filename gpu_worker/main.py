"""
FastAPI - "model server" thật, đứng SAU worker.py (PyWorker). Nhận request đã được
Vast Serverless + PyWorker forward tới, chạy run_job.py, trả kết quả.

THIẾT KẾ:
  - Mỗi request (/jobs/sub, /jobs/dub, /jobs/sub-only) ĐỒNG BỘ: nhận đủ input, chạy
    hết pipeline, trả kết quả trong CHÍNH response đó (không submit-rồi-poll, xem lý
    do trong phần bàn về routing không sticky của Vast Serverless).
  - Video trả về bằng FileResponse (streaming), KHÔNG base64: base64 tốn thêm ~33%
    băng thông và phải load hết file vào RAM trước khi gửi - với video vài GB,
    streaming đọc/gửi từng chunk nhỏ, RAM dùng gần như không đổi bất kể file to nhỏ.
  - Vì FileResponse chỉ trả được 1 file nhị phân làm response body, log/trans.srt đi
    kèm được gửi qua HTTP header riêng (X-Job-Logs, X-Exit-Code) thay vì nhét chung
    vào 1 JSON như bản base64 cũ.
  - /jobs/dub và /jobs/sub-only KHÔNG dựa vào workdir cũ của lần /jobs/sub trước -
    worker xử lý request này có thể là máy vật lý khác. Java phải gửi lại video gốc
    + trans.srt (đã qua tay user sửa) trong CHÍNH request đó.

Chạy: uvicorn main:app --host 127.0.0.1 --port 8000
(worker.py forward vào đây - port này PHẢI khớp FASTAPI_PORT trong worker.py)
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile

import yaml
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

app = FastAPI(title="AIDubbing GPU Model Server")

VIDEOLINGO_DIR = os.environ.get("VIDEOLINGO_DIR", "/root/VideoLingo")
JOB_WORKDIR_ROOT = os.environ.get("JOB_WORKDIR_ROOT", "/root/aidubbing-workdir")
PYTHON_EXECUTABLE = os.environ.get("PYTHON_EXECUTABLE", sys.executable)

# Ngưỡng VRAM (MB) - đọc từ config.yaml (D:/project/pyhon/VideoLingo/config.yaml
# trên máy GPU, key "vram_threshold_mb") thay vì biến môi trường, để cậu sửa 1 chỗ
# quen thuộc, không lo quên set lại mỗi lần mở terminal mới (biến môi trường không
# tồn tại lâu dài giữa các phiên). Đọc LẠI MỖI LẦN gọi (không cache 1 lần lúc
# startup) - sửa config.yaml có hiệu lực ngay, không cần restart uvicorn.
def _load_vram_threshold_mb() -> int:
    config_path = Path(VIDEOLINGO_DIR) / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return int(data.get("vram_threshold_mb", 20000))


# Lock CHỈ để bảo vệ đoạn "đo VRAM + quyết định nhận job" khỏi race condition khi
# nhiều request tới gần như cùng lúc - KHÔNG giữ trong lúc chạy run_job.py (khác hẳn
# việc giữ slot suốt cả job như thiết kế đếm-job cũ).
_vram_check_lock = asyncio.Lock()


def _get_gpu_memory_used_mb() -> int:
    """Gọi nvidia-smi để lấy VRAM đang dùng THẬT (GPU index 0) - cùng cơ chế với
    vram_monitor.py đã dùng để đo thủ công trước đó, giờ tự động hoá luôn trong
    FastAPI để tự quyết định nhận/từ chối job mới."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


async def _reserve_vram_slot_or_reject():
    """Gọi ở ĐẦU mỗi endpoint /jobs/* - kiểm tra VRAM hiện tại có dưới ngưỡng
    không, TRƯỚC khi bắt đầu chạy run_job.py. Nếu đầy, raise HTTPException(503) -
    Java coi đây là 'GPU bận, thử lại sau', KHÔNG PHẢI job lỗi thật (khác hẳn 500 -
    không set FAILED/refund credit, chỉ đưa job về lại PENDING để thử lại)."""
    async with _vram_check_lock:
        try:
            used_mb = _get_gpu_memory_used_mb()
            threshold_mb = _load_vram_threshold_mb()
        except Exception as e:
            # Không đo được VRAM (vd máy không có GPU NVIDIA, hoặc nvidia-smi lỗi)
            # hoặc không đọc được config.yaml - failopen: vẫn cho chạy, không chặn
            # cứng vì lỗi đo, chỉ log cảnh báo.
            print(f"[FASTAPI] Khong kiem tra duoc VRAM: {e} - cho chay tiep (failopen)", flush=True)
            return

        print(f"[FASTAPI] VRAM hien tai: {used_mb}MB / nguong {threshold_mb}MB", flush=True)
        if used_mb >= threshold_mb:
            raise HTTPException(
                503,
                detail=f"GPU dang day VRAM ({used_mb}MB >= nguong {threshold_mb}MB) - thu lai sau",
            )


def _job_workdir(job_id: str) -> Path:
    return Path(JOB_WORKDIR_ROOT) / f"job_{job_id}"


def _prepare_job_workdir(job_id: str) -> Path:
    """Copy các file TEMPLATE dùng chung (không đổi theo job) từ VIDEOLINGO_DIR vào
    workdir riêng - core/ đọc chúng bằng relative path (resolve theo cwd = workdir).
    QUAN TRỌNG: chỉ có 2 file loại này trong toàn bộ pipeline (đã grep xác nhận) -
    config.yaml VÀ custom_terms.xlsx (core/_4_1_summarize.py đọc để lấy thuật ngữ
    tùy chỉnh khi tóm tắt/dịch) - lỗi thật đã gặp khi thiếu file thứ 2 này."""
    workdir = _job_workdir(job_id)
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(VIDEOLINGO_DIR) / "config.yaml", workdir / "config.yaml")
    shutil.copy2(Path(VIDEOLINGO_DIR) / "custom_terms.xlsx", workdir / "custom_terms.xlsx")
    return workdir


async def _save_upload(upload: UploadFile, dest_path: Path):
    """Ghi theo chunk (không load hết vào RAM) - áp dụng cho cả video gốc VÀ
    trans.srt upload lên, đối xứng với cách trả kết quả về bằng streaming."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        while chunk := await upload.read(1024 * 1024):
            f.write(chunk)


IDLE_TIMEOUT_SECONDS = 300  # 5 phút không có dòng log mới nào -> coi là treo, tự kill


async def _run_job_and_wait(job_id: str, args: list[str]) -> tuple[int, list[str]]:
    """QUAN TRỌNG - bảo vệ khỏi treo vô hạn: đã gặp thật (log cũ jobId=69, CapCut
    TTS lỗi rồi fallback treo mãi, Java phải tắt hẳn app mới chạy lại được). Với
    GPU thuê tính tiền theo giờ, process treo mãi sẽ tốn tiền vô hạn nếu không có
    gì tự kill nó. Dùng IDLE TIMEOUT (không thấy log mới trong N phút) thay vì
    timeout tổng cố định - vì video dài THẬT SỰ có thể cần chạy lâu (30 phút gói
    cao), timeout tổng cố định sẽ giết nhầm job đang chạy đúng. Idle timeout chỉ
    giết khi PROCESS KHÔNG CÒN TIẾN TRIỂN GÌ (không in log mới), đúng bản chất
    "treo" cần phát hiện."""
    workdir = _job_workdir(job_id)
    cmd = [PYTHON_EXECUTABLE, str(Path(VIDEOLINGO_DIR) / "run_job.py"), *args]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    process = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(workdir),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env,
    )

    log_lines: list[str] = []
    assert process.stdout is not None
    timed_out = False

    while True:
        try:
            # wait_for bọc quanh ĐÚNG 1 lần đọc dòng - nếu không có dòng nào mới
            # trong IDLE_TIMEOUT_SECONDS, ném TimeoutError thay vì block vô hạn
            # như "async for" thông thường.
            raw_line = await asyncio.wait_for(
                process.stdout.readline(), timeout=IDLE_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            timed_out = True
            break

        if not raw_line:  # EOF - process đã đóng stdout, kết thúc bình thường
            break

        line = raw_line.decode("utf-8", errors="replace").rstrip()
        print(f"[job {job_id}] {line}", flush=True)
        log_lines.append(line)

    if timed_out:
        idle_msg = (f"[FASTAPI] Khong co log moi trong {IDLE_TIMEOUT_SECONDS}s "
                     f"- coi la treo, tu kill process.")
        print(f"[job {job_id}] {idle_msg}", flush=True)
        log_lines.append(idle_msg)
        process.kill()
        await process.wait()  # đợi kill thật sự hoàn tất, tránh zombie process
        return -1, log_lines  # exit code -1 = bị kill do idle timeout, không phải lỗi Python thật

    exit_code = await process.wait()
    return exit_code, log_lines


def _cleanup_workdir(job_id: str):
    """Chạy sau khi FileResponse đã gửi xong toàn bộ file (BackgroundTask chỉ chạy
    SAU KHI response hoàn tất, không xoá workdir giữa chừng lúc còn đang stream)."""
    shutil.rmtree(_job_workdir(job_id), ignore_errors=True)


def _logs_header(log_lines: list[str]) -> str:
    """Nhét log vào 1 HTTP header - giới hạn độ dài header thường ~8KB tuỳ server,
    nên chỉ lấy N dòng cuối (đủ để thấy traceback lỗi, xem case SameFileError trước
    đó) thay vì toàn bộ log dài. Java in log đầy đủ hơn thì đọc trực tiếp trên máy
    GPU khi cần debug sâu (đã thống nhất: log chi tiết KHÔNG bắt buộc phải về tới
    Java, miễn có cách debug được khi cần)."""
    tail = log_lines[-50:]
    # JSON-encode để giữ được ký tự đặc biệt/tiếng Việt an toàn trong header, và
    # để Java parse ngược lại dễ dàng bằng thư viện JSON có sẵn.
    return json.dumps(tail, ensure_ascii=True)


@app.post("/jobs/sub")
async def run_sub_job(
    input_video: UploadFile = File(...),
    subtitle_source: str = Form(""),
    source_lang: str = Form(""),
    ocr_top: str = Form(""),
    ocr_bottom: str = Form(""),
    ocr_left: str = Form(""),
    ocr_right: str = Form(""),
    duration_seconds: str = Form("300"),  # đọc bởi workload_calculator trong worker.py
):
    await _reserve_vram_slot_or_reject()

    job_id = uuid.uuid4().hex
    workdir = _prepare_job_workdir(job_id)

    input_ext = Path(input_video.filename or "input.mp4").suffix or ".mp4"
    input_path = workdir / f"input{input_ext}"
    await _save_upload(input_video, input_path)

    args = [
        "--input", str(input_path),
        "--job-id", job_id,
        "--stage", "sub",
        "--subtitle-source", subtitle_source,
        "--source-lang", source_lang,
        "--ocr-top", ocr_top,
        "--ocr-bottom", ocr_bottom,
        "--ocr-left", ocr_left,
        "--ocr-right", ocr_right,
    ]

    exit_code, log_lines = await _run_job_and_wait(job_id, args)

    if exit_code != 0:
        headers = {"X-Job-Logs": _logs_header(log_lines), "X-Exit-Code": str(exit_code)}
        _cleanup_workdir(job_id)
        raise HTTPException(500, detail="SUB stage failed", headers=headers)

    # Video KHÔNG zip chung (zip không nén thêm được gì cho file đã nén sẵn như
    # mp4, chỉ tốn thời gian vô ích) - trả qua route RIÊNG (/jobs/{job_id}/video),
    # Java tải 2 lần: 1 lần lấy zip (text/json), 1 lần lấy video. workdir CHƯA bị
    # dọn ở đây - chỉ dọn sau khi CẢ 2 lần tải đều xong (xem
    # /jobs/{job_id}/video bên dưới).
    # QUAN TRỌNG: KHÔNG loại trừ "audio" - Java cần audio/ (đặc biệt raw.mp3 đã
    # demucs) để gửi lại khi bấm Dub sau đó (xem GpuWorkerClient.zipOutputDir() -
    # đã gặp lỗi thật: thiếu audio/ khiến demucs_audio() ở stage dub báo
    # LoadAudioError vì raw.mp3 không tồn tại, dù đã tách sẵn từ lần sub này).
    zip_dir = workdir / "sub_result_files"
    if zip_dir.exists():
        shutil.rmtree(zip_dir)
    shutil.copytree(
        workdir / "output", zip_dir,
        ignore=shutil.ignore_patterns("*.mp4"),
    )
    zip_base = workdir / "sub_result"
    zip_path = shutil.make_archive(str(zip_base), "zip", str(zip_dir))
    # Dọn ngay bản copy tạm SAU KHI nén xong - trước đây bỏ sót bước này, để lại
    # rác trùng lặp với output/ trên đĩa GPU (cậu đã phát hiện: output_sub.mp4 tồn
    # tại 2 lần trên đĩa GPU do bản chạy CODE CŨ chưa loại trừ *.mp4 lúc copytree -
    # bản code hiện tại đã loại trừ đúng ở dòng ignore_patterns phía trên, dòng
    # rmtree này thêm 1 lớp an toàn dọn dẹp, không phụ thuộc vào đúng bản code nào).
    shutil.rmtree(zip_dir)

    headers = {"X-Job-Logs": _logs_header(log_lines), "X-Exit-Code": "0", "X-Job-Id": job_id}
    return FileResponse(
        zip_path, filename="sub_result.zip", media_type="application/zip",
        headers=headers,
        # KHÔNG cleanup ở đây nữa - job_id vẫn cần sống tới khi Java gọi tiếp
        # GET /jobs/{job_id}/video để lấy video, mới thật sự dọn.
    )


@app.get("/jobs/{job_id}/video")
async def get_sub_video(job_id: str):
    """Java gọi NGAY sau khi tải xong zip từ /jobs/sub - lấy video, sau đó mới dọn
    workdir (BackgroundTask ở đây). Tách route để tránh zip video chung (không có
    lợi ích nén, chỉ tốn thời gian - xem comment trong /jobs/sub)."""
    video_path = _job_workdir(job_id) / "output" / "output_sub.mp4"
    if not video_path.exists():
        raise HTTPException(404, f"Không tìm thấy video tại {video_path} - job_id sai hoặc đã bị dọn")
    return FileResponse(
        video_path, filename="output_sub.mp4", media_type="video/mp4",
        background=BackgroundTask(_cleanup_workdir, job_id),
    )


@app.post("/jobs/dub")
async def run_dub_job(
    input_video: UploadFile = File(...),
    output_bundle: UploadFile = File(...),
    voice_id: str = Form(""),
    subtitle_source: str = Form(""),
    ocr_top: str = Form(""),
    ocr_bottom: str = Form(""),
    ocr_left: str = Form(""),
    ocr_right: str = Form(""),
    background_music_volume: str = Form(""),
    duration_seconds: str = Form("300"),
):
    """Nhận LẠI video gốc + output_bundle.zip (chứa TOÀN BỘ output/ từ lần /jobs/sub
    trước, đã qua tay user sửa trên FE) - KHÔNG giả định workdir cũ còn tồn tại (xem
    docstring đầu file). QUAN TRỌNG: KHÔNG chỉ gửi trans.srt - lỗi thật đã gặp:
    zh_gen_audio_tasks() (pipeline ZH) đọc output/log/zh_sync.json, không phải
    trans.srt, và có thể còn file khác cần tùy pipeline. Gửi cả bundle (giống hệt
    cách /jobs/sub trả về) để không phải đoán/liệt kê từng file.

    QUAN TRỌNG - subtitle_source PHẢI truyền lại: config.yaml của lần dub này là
    bản MỚI copy từ template (KHÔNG kế thừa 'subtitle_source: ocr' đã set lúc sub
    trước, vì mỗi request /jobs/* độc lập hoàn toàn - bản chất serverless). Thiếu
    tham số này, _12_dub_to_vid.py đọc subtitle_source rỗng -> use_ocr_crop_style
    luôn False -> MẤT crop/blur cho video OCR (lỗi thật đã gặp: dub video OCR
    không blur, sub rơi về vị trí mặc định như Whisper)."""
    await _reserve_vram_slot_or_reject()

    job_id = uuid.uuid4().hex
    workdir = _prepare_job_workdir(job_id)

    bundle_zip_path = workdir / "output_bundle.zip"
    await _save_upload(output_bundle, bundle_zip_path)
    output_dir = workdir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip_path) as zf:
        zf.extractall(output_dir)
    bundle_zip_path.unlink()

    # QUAN TRỌNG: video gốc phải nằm TRỰC TIẾP TRONG output/, không phải ngang
    # hàng với nó - lỗi thật đã gặp: run_job.py --stage dub KHÔNG nhận --input để
    # tự copy vào output/ (khác --stage sub, nó tự gọi _prepare_output_dir() làm
    # việc này) - --stage dub giả định video gốc ĐÃ CÓ SẴN trong output/ từ lần sub
    # trước (đúng thiết kế on-demand gốc, cùng máy, cùng thư mục). Với serverless,
    # ta phải tự đặt nó vào đúng chỗ TRƯỚC khi chạy, giống hệt _prepare_output_dir()
    # làm - core/_1_ytdlp.py.find_video_files() quét output/*, lọc theo
    # allowed_video_formats, loại trừ file bắt đầu bằng "output/output".
    input_ext = Path(input_video.filename or "input.mp4").suffix or ".mp4"
    input_path = output_dir / f"input{input_ext}"
    await _save_upload(input_video, input_path)

    output_path = workdir / "output" / "output_dub.mp4"

    args = [
        "--output", str(output_path),
        "--job-id", job_id,
        "--stage", "dub",
        "--voice-id", voice_id,
        "--subtitle-source", subtitle_source,
        "--ocr-top", ocr_top,
        "--ocr-bottom", ocr_bottom,
        "--ocr-left", ocr_left,
        "--ocr-right", ocr_right,
        "--background-music-volume", background_music_volume,
    ]

    exit_code, log_lines = await _run_job_and_wait(job_id, args)

    if exit_code != 0:
        headers = {"X-Job-Logs": _logs_header(log_lines), "X-Exit-Code": str(exit_code)}
        _cleanup_workdir(job_id)
        raise HTTPException(500, detail="DUB stage failed", headers=headers)

    # Chỉ 1 file (video dub) - trả trực tiếp bằng FileResponse, không cần zip như /jobs/sub.
    headers = {"X-Job-Logs": _logs_header(log_lines), "X-Exit-Code": "0"}
    return FileResponse(
        output_path, filename="output_dub.mp4", media_type="video/mp4",
        headers=headers,
        background=BackgroundTask(_cleanup_workdir, job_id),
    )


@app.post("/jobs/sub-only")
async def run_sub_only_job(
    input_video: UploadFile = File(...),
    output_bundle: UploadFile = File(...),
    subtitle_source: str = Form(""),
    ocr_top: str = Form(""),
    ocr_bottom: str = Form(""),
    ocr_left: str = Form(""),
    ocr_right: str = Form(""),
    duration_seconds: str = Form("300"),
):
    """Y hệt /jobs/dub nhưng chỉ burn sub, không TTS - cũng nhận output_bundle.zip
    thay vì chỉ trans.srt, và cũng cần subtitle_source (xem docstring /jobs/dub -
    _7_sub_into_vid.py cùng dùng use_ocr_crop_style, cùng lỗi mất crop/blur nếu thiếu)."""
    await _reserve_vram_slot_or_reject()

    job_id = uuid.uuid4().hex
    workdir = _prepare_job_workdir(job_id)

    bundle_zip_path = workdir / "output_bundle.zip"
    await _save_upload(output_bundle, bundle_zip_path)
    output_dir = workdir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip_path) as zf:
        zf.extractall(output_dir)
    bundle_zip_path.unlink()

    # Video gốc phải nằm TRỰC TIẾP TRONG output/ - _7_sub_into_vid.py cũng gọi
    # find_video_files() (xem docstring /jobs/dub, cùng lý do y hệt).
    input_ext = Path(input_video.filename or "input.mp4").suffix or ".mp4"
    input_path = output_dir / f"input{input_ext}"
    await _save_upload(input_video, input_path)

    output_path = workdir / "output" / "output_sub.mp4"

    args = [
        "--output", str(output_path),
        "--job-id", job_id,
        "--stage", "sub-only",
        "--subtitle-source", subtitle_source,
        "--ocr-top", ocr_top,
        "--ocr-bottom", ocr_bottom,
        "--ocr-left", ocr_left,
        "--ocr-right", ocr_right,
    ]

    exit_code, log_lines = await _run_job_and_wait(job_id, args)

    if exit_code != 0:
        headers = {"X-Job-Logs": _logs_header(log_lines), "X-Exit-Code": str(exit_code)}
        _cleanup_workdir(job_id)
        raise HTTPException(500, detail="SUB-ONLY stage failed", headers=headers)

    headers = {"X-Job-Logs": _logs_header(log_lines), "X-Exit-Code": "0"}
    return FileResponse(
        output_path, filename="output_sub.mp4", media_type="video/mp4",
        headers=headers,
        background=BackgroundTask(_cleanup_workdir, job_id),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}