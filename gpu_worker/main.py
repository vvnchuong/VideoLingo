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


# --- SUB stage: đổi sang async (submit -> poll status -> lấy result riêng) ---
# LÝ DO ĐỔI: mỗi request /jobs/sub trước đây giữ HTTP connection mở suốt cả pipeline
# (transcribe+dịch có thể mất vài phút). Khi traffic đi qua Cloudflare Tunnel (đang
# dùng cho test local qua gpu.aidub.tech), Cloudflare có giới hạn CỨNG 100 giây cho
# mỗi request/response - job nào chạy quá 100s bị Cloudflare tự cắt và trả lỗi 524,
# dù bản thân job vẫn tiếp tục chạy xong bình thường ở phía dưới (không liên quan gì
# tới lỗi thật của job). Việc tách submit/poll áp dụng đúng pattern đã dùng cho
# /download-video trước đó.
#
# LƯU Ý (đã bàn trước đây, giữ lại cho lần đọc sau): pattern submit/poll này CHỈ an
# toàn khi job luôn nằm cố định trên CÙNG một máy GPU vật lý (đúng trường hợp hiện
# tại: 1 máy Windows local duy nhất). Nếu sau này chuyển sang Vast.ai Serverless
# (routing không sticky - mỗi request có thể rơi vào instance khác nhau), pattern
# này sẽ cần thiết kế lại (ví dụ dùng shared storage cho job state, hoặc quay lại
# đồng bộ). Chưa cần lo việc đó bây giờ - đang test local.

_sub_jobs: dict[str, dict] = {}  # job_id -> {"status": "PENDING"|"DONE"|"FAILED", "logs": [...], "exit_code": int|None, "error": str|None}


async def _run_sub_job_background(
    job_id: str,
    workdir: Path,
    subtitle_source: str,
    source_lang: str,
    ocr_top: str,
    ocr_bottom: str,
    ocr_left: str,
    ocr_right: str,
    subtitle_position_top: str,
    subtitle_position_bottom: str,
):
    args = [
        "--input", str(next(workdir.glob("input.*"))),
        "--job-id", job_id,
        "--stage", "sub",
        "--subtitle-source", subtitle_source,
        "--source-lang", source_lang,
        "--ocr-top", ocr_top,
        "--ocr-bottom", ocr_bottom,
        "--ocr-left", ocr_left,
        "--ocr-right", ocr_right,
        "--subtitle-position-top", subtitle_position_top,
        "--subtitle-position-bottom", subtitle_position_bottom,
    ]

    try:
        exit_code, log_lines = await _run_job_and_wait(job_id, args)

        if exit_code != 0:
            _sub_jobs[job_id] = {
                "status": "FAILED", "logs": log_lines[-50:], "exit_code": exit_code,
                "error": "SUB stage failed",
            }
            _cleanup_workdir(job_id)
            return

        # Video KHÔNG zip chung (không có lợi ích nén thêm cho mp4 đã nén sẵn) -
        # tách route riêng /jobs/{job_id}/video, giống thiết kế cũ.
        # QUAN TRỌNG: KHÔNG loại trừ "audio" - stage dub sau này cần audio/raw.mp3
        # đã demucs từ bước sub này (xem comment gốc, đã gặp lỗi LoadAudioError
        # thật khi thiếu file này).
        zip_dir = workdir / "sub_result_files"
        if zip_dir.exists():
            shutil.rmtree(zip_dir)
        shutil.copytree(
            workdir / "output", zip_dir,
            ignore=shutil.ignore_patterns("*.mp4"),
        )
        zip_base = workdir / "sub_result"
        shutil.make_archive(str(zip_base), "zip", str(zip_dir))
        shutil.rmtree(zip_dir)

        _sub_jobs[job_id] = {
            "status": "DONE", "logs": log_lines[-50:], "exit_code": 0, "error": None,
        }
    except Exception as e:
        print(f"[job {job_id}] Loi khong luong truoc duoc trong background task: {e}", flush=True)
        _sub_jobs[job_id] = {
            "status": "FAILED", "logs": [], "exit_code": None, "error": str(e),
        }
        _cleanup_workdir(job_id)


@app.post("/jobs/sub")
async def run_sub_job(
    input_video: UploadFile = File(...),
    subtitle_source: str = Form(""),
    source_lang: str = Form(""),
    ocr_top: str = Form(""),
    ocr_bottom: str = Form(""),
    ocr_left: str = Form(""),
    ocr_right: str = Form(""),
    subtitle_position_top: str = Form(""),
    subtitle_position_bottom: str = Form(""),
    duration_seconds: str = Form("300"),  # đọc bởi workload_calculator trong worker.py
):
    await _reserve_vram_slot_or_reject()

    job_id = uuid.uuid4().hex
    workdir = _prepare_job_workdir(job_id)

    input_ext = Path(input_video.filename or "input.mp4").suffix or ".mp4"
    input_path = workdir / f"input{input_ext}"
    await _save_upload(input_video, input_path)

    _sub_jobs[job_id] = {"status": "PENDING", "logs": [], "exit_code": None, "error": None}

    # Chạy nền - KHÔNG await ở đây, để request trả về ngay lập tức (tránh giữ
    # connection mở qua Cloudflare quá 100s). Java sẽ poll GET /jobs/sub/{job_id}/status.
    asyncio.create_task(_run_sub_job_background(
        job_id, workdir, subtitle_source, source_lang,
        ocr_top, ocr_bottom, ocr_left, ocr_right,
        subtitle_position_top, subtitle_position_bottom,
    ))

    return {"job_id": job_id, "status": "PENDING"}


@app.get("/jobs/sub/{job_id}/status")
async def get_sub_job_status(job_id: str):
    """Java poll endpoint này (vài giây/lần) tới khi status khác PENDING."""
    state = _sub_jobs.get(job_id)
    if state is None:
        raise HTTPException(404, f"Không tìm thấy sub job_id={job_id}")
    return {
        "job_id": job_id,
        "status": state["status"],
        "logs": state["logs"],
        "exit_code": state["exit_code"],
        "error": state["error"],
    }


@app.get("/jobs/sub/{job_id}/result")
async def get_sub_job_result(job_id: str):
    """Java gọi khi status=DONE để lấy zip (trans.srt + audio/...). Giữ đúng tên
    file/media-type như FileResponse cũ để không phải đổi cách Java giải nén."""
    state = _sub_jobs.get(job_id)
    if state is None or state["status"] != "DONE":
        raise HTTPException(404, f"sub job_id={job_id} chưa xong hoặc không tồn tại")

    zip_path = _job_workdir(job_id) / "sub_result.zip"
    if not zip_path.exists():
        raise HTTPException(404, f"Không tìm thấy sub_result.zip cho job_id={job_id}")

    return FileResponse(
        zip_path, filename="sub_result.zip", media_type="application/zip",
        # KHÔNG cleanup ở đây nữa - job_id vẫn cần sống tới khi Java gọi tiếp
        # GET /jobs/{job_id}/video để lấy video, mới thật sự dọn (giống thiết kế cũ).
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


_dub_jobs: dict[str, dict] = {}  # cùng cấu trúc _sub_jobs


async def _run_dub_job_background(job_id: str, workdir: Path, output_path: Path,
                                   voice_id: str, subtitle_source: str,
                                   ocr_top: str, ocr_bottom: str, ocr_left: str, ocr_right: str,
                                   subtitle_position_top: str, subtitle_position_bottom: str,
                                   background_music_volume: str, subtitle_style: str):
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
        "--subtitle-position-top", subtitle_position_top,
        "--subtitle-position-bottom", subtitle_position_bottom,
        "--background-music-volume", background_music_volume,
        "--subtitle-style", subtitle_style,
    ]

    try:
        exit_code, log_lines = await _run_job_and_wait(job_id, args)

        if exit_code != 0:
            _dub_jobs[job_id] = {
                "status": "FAILED", "logs": log_lines[-50:], "exit_code": exit_code,
                "error": "DUB stage failed",
            }
            _cleanup_workdir(job_id)
            return

        _dub_jobs[job_id] = {
            "status": "DONE", "logs": log_lines[-50:], "exit_code": 0, "error": None,
        }
    except Exception as e:
        print(f"[job {job_id}] Loi khong luong truoc duoc trong background task: {e}", flush=True)
        _dub_jobs[job_id] = {
            "status": "FAILED", "logs": [], "exit_code": None, "error": str(e),
        }
        _cleanup_workdir(job_id)


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
    subtitle_position_top: str = Form(""),
    subtitle_position_bottom: str = Form(""),
    background_music_volume: str = Form(""),
    subtitle_style: str = Form(""),
    duration_seconds: str = Form("300"),
):
    """Nhận LẠI video gốc + output_bundle.zip (chứa TOÀN BỘ output/ từ lần /jobs/sub
    trước, đã qua tay user sửa trên FE) - KHÔNG giả định workdir cũ còn tồn tại (xem
    docstring đầu file). QUAN TRỌNG: KHÔNG chỉ gửi trans.srt - lỗi thật đã gặp:
    src_gen_audio_tasks() (nhánh OCR, core/ocr_lines.py) đọc output/log/src_sync.json, không phải
    trans.srt, và có thể còn file khác cần tùy pipeline. Gửi cả bundle (giống hệt
    cách /jobs/sub trả về) để không phải đoán/liệt kê từng file.

    QUAN TRỌNG - subtitle_source PHẢI truyền lại: config.yaml của lần dub này là
    bản MỚI copy từ template (KHÔNG kế thừa 'subtitle_source: ocr' đã set lúc sub
    trước, vì mỗi request /jobs/* độc lập hoàn toàn - bản chất serverless). Thiếu
    tham số này, _12_dub_to_vid.py đọc subtitle_source rỗng -> use_ocr_crop_style
    luôn False -> MẤT crop/blur cho video OCR (lỗi thật đã gặp: dub video OCR
    không blur, sub rơi về vị trí mặc định như Whisper).

    ĐỔI SANG ASYNC (submit/poll) - cùng lý do với /jobs/sub: tránh giữ HTTP
    connection mở quá 100s qua Cloudflare Tunnel khi dub chạy lâu."""
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

    _dub_jobs[job_id] = {"status": "PENDING", "logs": [], "exit_code": None, "error": None}

    asyncio.create_task(_run_dub_job_background(
        job_id, workdir, output_path, voice_id, subtitle_source,
        ocr_top, ocr_bottom, ocr_left, ocr_right,
        subtitle_position_top, subtitle_position_bottom,
        background_music_volume, subtitle_style,
    ))

    return {"job_id": job_id, "status": "PENDING"}


@app.get("/jobs/dub/{job_id}/status")
async def get_dub_job_status(job_id: str):
    state = _dub_jobs.get(job_id)
    if state is None:
        raise HTTPException(404, f"Không tìm thấy dub job_id={job_id}")
    return {
        "job_id": job_id,
        "status": state["status"],
        "logs": state["logs"],
        "exit_code": state["exit_code"],
        "error": state["error"],
    }


@app.get("/jobs/dub/{job_id}/result")
async def get_dub_job_result(job_id: str):
    """Java gọi khi status=DONE để lấy video dub - chỉ 1 file, không cần zip."""
    state = _dub_jobs.get(job_id)
    if state is None or state["status"] != "DONE":
        raise HTTPException(404, f"dub job_id={job_id} chưa xong hoặc không tồn tại")

    output_path = _job_workdir(job_id) / "output" / "output_dub.mp4"
    if not output_path.exists():
        raise HTTPException(404, f"Không tìm thấy output_dub.mp4 cho job_id={job_id}")

    return FileResponse(
        output_path, filename="output_dub.mp4", media_type="video/mp4",
        background=BackgroundTask(_cleanup_workdir, job_id),
    )


_sub_only_jobs: dict[str, dict] = {}  # cùng cấu trúc _sub_jobs / _dub_jobs


async def _run_sub_only_job_background(job_id: str, workdir: Path, output_path: Path,
                                        subtitle_source: str,
                                        ocr_top: str, ocr_bottom: str, ocr_left: str, ocr_right: str,
                                        subtitle_position_top: str, subtitle_position_bottom: str,
                                        subtitle_style: str):
    args = [
        "--output", str(output_path),
        "--job-id", job_id,
        "--stage", "sub-only",
        "--subtitle-source", subtitle_source,
        "--ocr-top", ocr_top,
        "--ocr-bottom", ocr_bottom,
        "--ocr-left", ocr_left,
        "--ocr-right", ocr_right,
        "--subtitle-position-top", subtitle_position_top,
        "--subtitle-position-bottom", subtitle_position_bottom,
        "--subtitle-style", subtitle_style,
    ]

    try:
        exit_code, log_lines = await _run_job_and_wait(job_id, args)

        if exit_code != 0:
            _sub_only_jobs[job_id] = {
                "status": "FAILED", "logs": log_lines[-50:], "exit_code": exit_code,
                "error": "SUB-ONLY stage failed",
            }
            _cleanup_workdir(job_id)
            return

        _sub_only_jobs[job_id] = {
            "status": "DONE", "logs": log_lines[-50:], "exit_code": 0, "error": None,
        }
    except Exception as e:
        print(f"[job {job_id}] Loi khong luong truoc duoc trong background task: {e}", flush=True)
        _sub_only_jobs[job_id] = {
            "status": "FAILED", "logs": [], "exit_code": None, "error": str(e),
        }
        _cleanup_workdir(job_id)


@app.post("/jobs/sub-only")
async def run_sub_only_job(
    input_video: UploadFile = File(...),
    output_bundle: UploadFile = File(...),
    subtitle_source: str = Form(""),
    ocr_top: str = Form(""),
    ocr_bottom: str = Form(""),
    ocr_left: str = Form(""),
    ocr_right: str = Form(""),
    subtitle_position_top: str = Form(""),
    subtitle_position_bottom: str = Form(""),
    subtitle_style: str = Form(""),
    duration_seconds: str = Form("300"),
):
    """Y hệt /jobs/dub nhưng chỉ burn sub, không TTS - cũng nhận output_bundle.zip
    thay vì chỉ trans.srt, và cũng cần subtitle_source (xem docstring /jobs/dub -
    _7_sub_into_vid.py cùng dùng use_ocr_crop_style, cùng lỗi mất crop/blur nếu thiếu).

    ĐỔI SANG ASYNC (submit/poll) - cùng lý do với /jobs/sub và /jobs/dub."""
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

    _sub_only_jobs[job_id] = {"status": "PENDING", "logs": [], "exit_code": None, "error": None}

    asyncio.create_task(_run_sub_only_job_background(
        job_id, workdir, output_path, subtitle_source,
        ocr_top, ocr_bottom, ocr_left, ocr_right,
        subtitle_position_top, subtitle_position_bottom,
        subtitle_style,
    ))

    return {"job_id": job_id, "status": "PENDING"}


@app.get("/jobs/sub-only/{job_id}/status")
async def get_sub_only_job_status(job_id: str):
    state = _sub_only_jobs.get(job_id)
    if state is None:
        raise HTTPException(404, f"Không tìm thấy sub-only job_id={job_id}")
    return {
        "job_id": job_id,
        "status": state["status"],
        "logs": state["logs"],
        "exit_code": state["exit_code"],
        "error": state["error"],
    }


@app.get("/jobs/sub-only/{job_id}/result")
async def get_sub_only_job_result(job_id: str):
    """Java gọi khi status=DONE để lấy video sub-only - chỉ 1 file, không cần zip."""
    state = _sub_only_jobs.get(job_id)
    if state is None or state["status"] != "DONE":
        raise HTTPException(404, f"sub-only job_id={job_id} chưa xong hoặc không tồn tại")

    output_path = _job_workdir(job_id) / "output" / "output_sub.mp4"
    if not output_path.exists():
        raise HTTPException(404, f"Không tìm thấy output_sub.mp4 cho job_id={job_id}")

    return FileResponse(
        output_path, filename="output_sub.mp4", media_type="video/mp4",
        background=BackgroundTask(_cleanup_workdir, job_id),
    )


@app.post("/download-video")
async def download_video(url: str = Form(...)):
    """Endpoint mới - thay thế việc Java tự gọi ProcessBuilder chạy download_video.py
    trực tiếp (không còn khả thi khi Java chạy trong Docker container trên VPS,
    khác OS/filesystem với máy GPU chạy Windows). Logic tải giữ NGUYÊN, chỉ đổi
    cách gọi: HTTP thay vì subprocess trực tiếp từ Java.

    Dùng lại đúng script download_video.py hiện có (wrapper quanh
    core/_1_ytdlp.py, xử lý cả nhánh Douyin qua Playwright) - không viết lại
    logic tải, tránh trùng công và dễ sai (đặc biệt Douyin cần cookie/Playwright).
    """
    job_id = uuid.uuid4().hex
    workdir = _job_workdir(job_id)
    workdir.mkdir(parents=True, exist_ok=True)
    output_path = workdir / "downloaded_video.mp4"

    cmd = [
        PYTHON_EXECUTABLE, str(Path(VIDEOLINGO_DIR) / "download_video.py"),
        "--url", url,
        "--output", str(output_path),
    ]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    process = await asyncio.create_subprocess_exec(
        *cmd, cwd=VIDEOLINGO_DIR,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env,
    )

    log_lines: list[str] = []
    assert process.stdout is not None
    async for raw_line in process.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        print(f"[download {job_id}] {line}", flush=True)
        log_lines.append(line)

    exit_code = await process.wait()

    if exit_code != 0 or not output_path.exists():
        headers = {"X-Job-Logs": _logs_header(log_lines), "X-Exit-Code": str(exit_code)}
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(500, detail="Download video failed", headers=headers)

    headers = {"X-Job-Logs": _logs_header(log_lines), "X-Exit-Code": "0"}
    return FileResponse(
        output_path, filename="downloaded_video.mp4", media_type="video/mp4",
        headers=headers,
        background=BackgroundTask(_cleanup_workdir, job_id),
    )


async def _run_tts_script(cli_script: str, args: list[str]) -> tuple[int, list[str]]:
    """Helper chung cho 2 endpoint TTS bên dưới - cả 2 đều chạy 1 trong 2 file
    CLI có sẵn (tts_cli.py / vieneu_cli.py) trong VIDEOLINGO_DIR, y hệt cách
    TtsService.java cũ dùng ProcessBuilder, chỉ khác nơi gọi (HTTP thay vì
    subprocess trực tiếp từ Java)."""
    cmd = [PYTHON_EXECUTABLE, str(Path(VIDEOLINGO_DIR) / cli_script)] + args

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    process = await asyncio.create_subprocess_exec(
        *cmd, cwd=VIDEOLINGO_DIR,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env,
    )

    log_lines: list[str] = []
    assert process.stdout is not None
    async for raw_line in process.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        print(f"[tts] {line}", flush=True)
        log_lines.append(line)

    exit_code = await process.wait()
    return exit_code, log_lines


@app.post("/tts/generate")
async def tts_generate(
    text: str = Form(...),
    voice_id: str = Form(...),
    cli_script: str = Form(...),  # "tts_cli.py" hoặc "vieneu_cli.py" - Java quyết định
    rate: str | None = Form(None),
):
    job_id = uuid.uuid4().hex
    workdir = _job_workdir(job_id)
    workdir.mkdir(parents=True, exist_ok=True)

    text_file = workdir / "tts_text.txt"
    text_file.write_text(text, encoding="utf-8")
    output_path = workdir / "output.mp3"

    args = [
        "--text-file", str(text_file),
        "--voice", voice_id,
        "--output", str(output_path),
    ]
    if rate:
        args += ["--rate", rate]

    exit_code, log_lines = await _run_tts_script(cli_script, args)

    if exit_code != 0 or not output_path.exists():
        headers = {"X-Job-Logs": _logs_header(log_lines), "X-Exit-Code": str(exit_code)}
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(500, detail="TTS generate failed", headers=headers)

    headers = {"X-Job-Logs": _logs_header(log_lines), "X-Exit-Code": "0"}
    return FileResponse(
        output_path, filename="output.mp3", media_type="audio/mpeg",
        headers=headers,
        background=BackgroundTask(_cleanup_workdir, job_id),
    )


@app.post("/tts/voice-sample")
async def tts_voice_sample(
    text: str = Form(...),
    voice_id: str = Form(...),
    cli_script: str = Form(...),
):
    job_id = uuid.uuid4().hex
    workdir = _job_workdir(job_id)
    workdir.mkdir(parents=True, exist_ok=True)
    output_path = workdir / "sample.mp3"

    args = ["--text", text, "--voice", voice_id, "--output", str(output_path)]
    exit_code, log_lines = await _run_tts_script(cli_script, args)

    if exit_code != 0 or not output_path.exists():
        headers = {"X-Job-Logs": _logs_header(log_lines), "X-Exit-Code": str(exit_code)}
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(500, detail="TTS voice sample failed", headers=headers)

    headers = {"X-Job-Logs": _logs_header(log_lines), "X-Exit-Code": "0"}
    return FileResponse(
        output_path, filename="sample.mp3", media_type="audio/mpeg",
        headers=headers,
        background=BackgroundTask(_cleanup_workdir, job_id),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}