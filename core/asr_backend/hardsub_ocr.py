"""
Hardsub OCR extractor - wrapper gọi subprocess riêng biệt (hardsub_ocr_worker.py)
để chạy PaddleOCR trên GPU mà không đụng độ DLL với torch (Whisper/OmniVoice)
trong cùng process. Output format giống hệt whisper segments để downstream
(process_transcription, save_results, zh_pipeline...) không cần đổi gì.
"""

import os
import sys
import json
import subprocess
import tempfile


def _get_ocr_python_executable():
    """
    Tìm python của venv riêng cho OCR (.venv_ocr), không cài torch, tránh
    xung đột DLL torch<->paddle trên Windows. Nếu không tìm thấy, fallback
    về sys.executable (venv chính - có thể lỗi nếu torch đã cài).
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if sys.platform == "win32":
        candidate = os.path.join(project_root, ".venv_ocr", "Scripts", "python.exe")
    else:
        candidate = os.path.join(project_root, ".venv_ocr", "bin", "python")
    if os.path.isfile(candidate):
        return candidate
    print("[OCR] CẢNH BÁO: không tìm thấy .venv_ocr, dùng venv chính (có thể lỗi do conflict với torch)")
    return sys.executable


def extract_hardsub(video_path, lang="ch", region=None):
    """
    Đọc sub cứng trong video bằng cách chạy worker OCR trong process con riêng,
    dùng venv riêng (.venv_ocr) để tránh xung đột DLL torch<->paddle.
    region: dict {top, bottom, left, right} dạng tỉ lệ 0.0-1.0, None = quét theo
    fallback mặc định (CROP_TOP_RATIO trong worker).
    Trả về {'segments': [{'start', 'end', 'text'}, ...]}
    """
    worker_path = os.path.join(os.path.dirname(__file__), "hardsub_ocr_worker.py")
    python_exe = _get_ocr_python_executable()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_json_path = tmp.name

    try:
        cmd = [python_exe, worker_path, video_path, output_json_path, lang]
        if region:
            cmd.append(json.dumps(region))
        print(f"[OCR] Chạy worker process: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=False, text=True)

        if process.returncode != 0:
            raise RuntimeError(f"OCR worker process thất bại (exit code {process.returncode})")

        with open(output_json_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        return result
    finally:
        if os.path.exists(output_json_path):
            os.remove(output_json_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hardsub_ocr.py <video_path> [lang]")
        sys.exit(1)
    video_path = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "ch"
    result = extract_hardsub(video_path, lang)
    print(json.dumps(result, ensure_ascii=False, indent=2))