"""
run_job.py - chạy pipeline VideoLingo KHÔNG qua Streamlit, để Java ProcessBuilder gọi.

Mô phỏng lại đúng thứ tự các bước _get_text_steps() / _get_audio_steps() trong st.py,
KHÔNG import st.py (tránh streamlit tự chạy code UI lúc import).

Cách chạy (đứng tại thư mục gốc VideoLingo, cùng chỗ chứa st.py):
    python run_job.py --input "D:/aidubbing-uploads/xxx.mp4" --output "D:/aidubbing-results/job_1.mp4"

Exit code:
    0  = thành công, in ra đúng 1 dòng cuối "RESULT_PATH=<đường dẫn file kết quả>"
    1  = lỗi, in traceback ra stderr

LƯU Ý QUAN TRỌNG (đọc trước khi dùng thật):
- Script này CHỈ ĐƯỢC chạy 1 job tại 1 thời điểm (bị giới hạn bởi thư mục output/ dùng chung
  và VRAM GPU chỉ đủ 1 job) - phía Java (JobPollerScheduler) phải đảm bảo không gọi song song.
- Mình copy lại đúng logic branch ZH/EN từ st.py hiện tại - NẾU BẠN SỬA st.py SAU NÀY
  (thêm bước, đổi thứ tự...), PHẢI SỬA LẠI FILE NÀY THEO, không tự đồng bộ.
- Chưa test thật với dữ liệu ZH pipeline - bạn cần tự chạy thử, đối chiếu kỹ với luồng
  Streamlit đã chạy ổn, trước khi cho chạy tự động qua Java.
"""

import argparse
import json
import os
import shutil
import sys
import traceback


def _configure_utf8_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_console()

# Đứng đúng tại thư mục gốc VideoLingo khi chạy script này (giống cách st.py giả định)
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + current_dir
sys.path.append(current_dir)

from core.utils.config_utils import load_key, update_key  # noqa: E402
from core import (  # noqa: E402
    _2_asr,
    _3_1_split_nlp,
    _3_2_split_meaning,
    _4_1_summarize,
    _4_2_translate,
    _5_split_sub,
    _6_gen_sub,
    _7_sub_into_vid,
    _8_1_audio_task,
    _8_2_dub_chunks,
    _9_refer_audio,
    _10_gen_audio,
    _11_merge_audio,
    _12_dub_to_vid,
)

OUTPUT_DIR = "output"
SUB_VIDEO = "output/output_sub.mp4"
DUB_VIDEO = "output/output_dub.mp4"


def _get_source_language():
    try:
        return load_key("whisper.language")
    except Exception:
        return "en"


def _build_text_steps():
    """Copy nguyên logic từ st.py::_get_text_steps() - KHÔNG import st.py."""
    lang = _get_source_language()

    if lang == "zh":
        from core.zh_pipeline import zh_asr_and_translate
        return [
            ("ZH: VAD + Whisper clip + Gemini duration-aware translate", zh_asr_and_translate),
            ("Merging subtitles into the video", _7_sub_into_vid.merge_subtitles_to_video),
        ]

    return [
        ("WhisperX word-level transcription", _2_asr.transcribe),
        ("Sentence segmentation using NLP and LLM", lambda: (
            _3_1_split_nlp.split_by_spacy(),
            _3_2_split_meaning.split_sentences_by_meaning(),
        )),
        ("Summarization and multi-step translation", lambda: (
            _4_1_summarize.get_summary(),
            _4_2_translate.translate_all(),
        )),
        ("Cutting and aligning long subtitles", lambda: (
            _5_split_sub.split_for_sub_main(),
            _6_gen_sub.align_timestamp_main(),
        )),
        ("Merging subtitles into the video", _7_sub_into_vid.merge_subtitles_to_video),
    ]


def _build_audio_steps():
    """Copy nguyên logic từ st.py::_get_audio_steps() - KHÔNG import st.py."""
    lang = _get_source_language()

    if lang == "zh":
        from core.zh_pipeline import zh_gen_audio_tasks
        return [
            ("Generate audio tasks and chunks", lambda: zh_gen_audio_tasks()),
            ("Extract reference audio", _9_refer_audio.extract_refer_audio_main),
            ("Generate and merge audio files", _10_gen_audio.gen_audio),
            ("Merge full audio", _11_merge_audio.merge_full_audio),
            ("Merge final audio into video", _12_dub_to_vid.merge_video_audio),
        ]

    return [
        ("Generate audio tasks and chunks", lambda: (
            _8_1_audio_task.gen_audio_task_main(),
            _8_2_dub_chunks.gen_dub_chunks(),
        )),
        ("Extract reference audio", _9_refer_audio.extract_refer_audio_main),
        ("Generate and merge audio files", _10_gen_audio.gen_audio),
        ("Merge full audio", _11_merge_audio.merge_full_audio),
        ("Merge final audio into video", _12_dub_to_vid.merge_video_audio),
    ]


def _run_steps(steps):
    for label, fn in steps:
        print(f"[STEP] {label}", flush=True)
        fn()


def _prepare_output_dir(input_video_path: str):
    """
    Dọn sạch output/ trước khi chạy job mới, tránh dính file cũ (từ lần test Streamlit
    trước, hoặc job trước đó) làm lệch dữ liệu - đúng nỗi lo "ZH/EN file count lệch"
    đã gặp trước đây.
    """
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ext = os.path.splitext(input_video_path)[1]
    dest = os.path.join(OUTPUT_DIR, f"input{ext}")
    shutil.copy2(input_video_path, dest)
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, help="Đường dẫn video gốc (bắt buộc với --stage sub hoặc all)")
    parser.add_argument("--output", required=False, help="Đường dẫn lưu video kết quả cuối (bắt buộc với --stage dub hoặc all)")
    parser.add_argument("--job-id", default="", help="Chỉ để in log cho dễ theo dõi")
    parser.add_argument(
        "--stage", default="all", choices=["sub", "dub", "sub-only", "all"],
        help="'sub' = chạy tới bước ra phụ đề rồi dừng; "
             "'dub' = chạy tiếp từ phụ đề có sẵn trong output/ tới hết (có ghép giọng); "
             "'sub-only' = chạy tiếp từ phụ đề có sẵn, CHỈ burn sub vào video, KHÔNG ghép giọng; "
             "'all' = chạy hết 1 mạch như trước (mặc định, tương thích ngược)"
    )
    parser.add_argument(
        "--source-lang", default="",
        help="Ngôn ngữ video gốc (VD: zh, en, ja...) - rỗng thì dùng giá trị đang có sẵn "
             "trong config.yaml (whisper.language), không ghi đè gì cả."
    )
    parser.add_argument(
        "--subtitle-source", default="",
        help="'whisper' (tự nhận diện giọng nói) hoặc 'ocr' (đọc chữ có sẵn/hardsub trong video) - "
             "rỗng thì dùng giá trị đang có sẵn trong config.yaml (subtitle_source)."
    )
    parser.add_argument(
        "--voice-id", default="",
        help="Tên preset giọng CapCut TTS để dùng cho bước dub (khớp key trong "
             "VOICE_PRESETS ở core/tts_backend/capcut_tts_wrapper.py, VD: nam_minh, hoai_my...) - "
             "rỗng thì dùng đúng ACTIVE_VOICE mặc định đang set trong file đó."
    )
    parser.add_argument(
        "--ocr-top", default="", help="Toạ độ top vùng quét/che (tỉ lệ 0.0-1.0), rỗng nếu không dùng OCR."
    )
    parser.add_argument(
        "--ocr-bottom", default="", help="Toạ độ bottom vùng quét/che (tỉ lệ 0.0-1.0)."
    )
    parser.add_argument(
        "--ocr-left", default="", help="Toạ độ left vùng quét/che (tỉ lệ 0.0-1.0)."
    )
    parser.add_argument(
        "--ocr-right", default="", help="Toạ độ right vùng quét/che (tỉ lệ 0.0-1.0)."
    )
    args = parser.parse_args()

    # Ghi đè config.yaml theo đúng lựa chọn của job này TRƯỚC khi build steps -
    # _build_text_steps()/_build_audio_steps() đọc config qua load_key() nên phải set trước.
    print(
        f"[DEBUG] source_lang='{args.source_lang}' subtitle_source='{args.subtitle_source}' "
        f"voice_id='{args.voice_id}' ocr_top='{args.ocr_top}' ocr_bottom='{args.ocr_bottom}' "
        f"ocr_left='{args.ocr_left}' ocr_right='{args.ocr_right}' - rỗng nghĩa là Java KHÔNG "
        f"gửi giá trị này, config.yaml sẽ giữ nguyên như cũ.",
        flush=True
    )

    if args.source_lang:
        update_key("whisper.language", args.source_lang)
        print(f"[DEBUG] Đã ghi whisper.language = {args.source_lang} vào config.yaml", flush=True)
    if args.subtitle_source:
        update_key("subtitle_source", args.subtitle_source)
        print(f"[DEBUG] Đã ghi subtitle_source = {args.subtitle_source} vào config.yaml", flush=True)
    if args.voice_id:
        # CapCut TTS chọn giọng qua biến môi trường (đọc trong core/tts_backend/custom_tts.py),
        # KHÔNG qua config.yaml, vì ACTIVE_VOICE là hằng số cấp module trong capcut_tts_wrapper.py.
        os.environ["CAPCUT_VOICE_PRESET"] = args.voice_id

    # 4 số riêng thay vì 1 chuỗi JSON qua CLI - vì Windows tự ý "sửa" dấu ngoặc kép
    # khi ProcessBuilder dựng command line, làm hỏng JSON truyền qua argv (lỗi thật
    # đã gặp: JSONDecodeError "Expecting property name enclosed in double quotes").
    # Dựng lại dict ngay trong Python, không parse JSON qua biên CLI nữa.
    # LUÔN ghi (kể cả rỗng -> None) chứ không "if ...:" như source_lang/subtitle_source -
    # vì bước dub/sub-only sau này gọi lại merge_subtitles_to_video(), nếu job hiện tại
    # không dùng OCR mà không xoá key cũ thì sẽ vô tình che nhầm video của job này bằng
    # toạ độ của job OCR trước đó (dùng chung 1 config.yaml, không tách theo từng job).
    if args.ocr_top and args.ocr_bottom and args.ocr_left and args.ocr_right:
        ocr_region = {
            "top": float(args.ocr_top),
            "bottom": float(args.ocr_bottom),
            "left": float(args.ocr_left),
            "right": float(args.ocr_right),
        }
    else:
        ocr_region = None

    ok = update_key("ocr_region", ocr_region)
    if not ok:
        print(
            "[CẢNH BÁO] config.yaml chưa có sẵn key 'ocr_region' nên update_key() không ghi được gì "
            "(update_key chỉ SỬA key đã tồn tại, không tự tạo key mới) - cần thêm dòng "
            "'ocr_region: null' vào config.yaml 1 lần cho đủ key.",
            flush=True
        )

    print(f"[JOB {args.job_id}] Bắt đầu xử lý, stage={args.stage}", flush=True)

    try:
        if args.stage == "sub":
            if not args.input:
                raise ValueError("--stage sub cần --input")
            _prepare_output_dir(args.input)
            _run_steps(_build_text_steps())
            print("SUB_STAGE_DONE=1", flush=True)
            sys.exit(0)

        if args.stage == "dub":
            # KHÔNG gọi _prepare_output_dir() - dùng đúng output/ đang có sẵn
            # (đã được Java copy từ workDir vào trước khi gọi lệnh này)
            if not args.output:
                raise ValueError("--stage dub cần --output")

            # QUAN TRỌNG: user có thể đã sửa trans.srt ở bước sub-edit. Streamlit xử lý
            # y hệt: gọi lại merge_subtitles_to_video() để render lại hình ảnh sub trên
            # video với nội dung mới, TRƯỚC KHI chạy tiếp audio/dub - xem
            # core/st_utils/edit_sub_section.py, nút "Re-render preview".
            print("[STEP] Re-render subtitle into video with edited translation", flush=True)
            _7_sub_into_vid.merge_subtitles_to_video()

            _run_steps(_build_audio_steps())
            _finalize_result(args.output)
            sys.exit(0)

        if args.stage == "sub-only":
            # Dành cho người chỉ cần video có sub cứng, KHÔNG cần ghép giọng lồng tiếng.
            # Giống bước đầu của "dub" ở chỗ re-render sub theo bản edit mới nhất,
            # nhưng KHÔNG chạy _build_audio_steps() (bỏ qua toàn bộ TTS/ghép audio),
            # nên nhanh hơn nhiều và không tốn quota TTS.
            if not args.output:
                raise ValueError("--stage sub-only cần --output")

            print("[STEP] Re-render subtitle into video with edited translation", flush=True)
            _7_sub_into_vid.merge_subtitles_to_video()

            _finalize_sub_only_result(args.output)
            sys.exit(0)

        # stage == "all" - giữ nguyên hành vi cũ, chạy hết 1 mạch
        if not args.input or not args.output:
            raise ValueError("--stage all cần cả --input và --output")
        _prepare_output_dir(args.input)
        _run_steps(_build_text_steps())
        _run_steps(_build_audio_steps())
        _finalize_result(args.output)
        sys.exit(0)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def _finalize_result(output_path: str):
    final_video = DUB_VIDEO if os.path.exists(DUB_VIDEO) else SUB_VIDEO
    if not os.path.exists(final_video):
        raise RuntimeError(f"Pipeline chạy xong nhưng không thấy file kết quả tại {final_video}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy2(final_video, output_path)
    print(f"RESULT_PATH={output_path}", flush=True)


def _finalize_sub_only_result(output_path: str):
    # Cố tình CHỈ lấy SUB_VIDEO, không dùng logic "ưu tiên DUB_VIDEO nếu có" như
    # _finalize_result() - vì đây là mode sub-only, nếu output/ lỡ còn sót
    # output_dub.mp4 từ lần chạy trước (workDir bị tái sử dụng nhầm) thì thà báo lỗi
    # rõ ràng còn hơn âm thầm trả nhầm video có giọng dub cho người chỉ muốn video sub.
    if not os.path.exists(SUB_VIDEO):
        raise RuntimeError(f"Pipeline chạy xong nhưng không thấy file kết quả tại {SUB_VIDEO}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy2(SUB_VIDEO, output_path)
    print(f"RESULT_PATH={output_path}", flush=True)


if __name__ == "__main__":
    main()