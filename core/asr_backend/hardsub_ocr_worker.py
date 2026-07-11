"""
Worker chạy trong PROCESS RIÊNG BIỆT để tránh conflict DLL giữa paddle-gpu và torch-gpu
trên Windows (2 framework build CUDA/cuDNN khác nhau không thể cùng load 1 process).

Script chính (hardsub_ocr.py) sẽ gọi subprocess tới file này, không import paddle trực tiếp.
Nhận: video_path, output_json_path qua argv.
Ghi: kết quả JSON {'segments': [...]} ra output_json_path.
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def _fix_windows_dll_path():
    if sys.platform != "win32":
        return
    for pkg in ["cudnn", "cublas"]:
        try:
            mod = __import__(f"nvidia.{pkg}", fromlist=[pkg])
            bin_dir = os.path.join(os.path.dirname(mod.__file__), "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
        except ImportError:
            pass


_fix_windows_dll_path()

# QUAN TRỌNG: import paddle TRƯỚC bất cứ thứ gì khác (kể cả cv2), vì paddleocr/paddlex
# tự động thử import torch để detect backend nếu paddle chưa được import trước -> conflict
# pybind ("generic_type: type '_gpuDeviceProperties' is already registered!")
import paddle  # noqa: F401,E402

import json
import cv2
from difflib import SequenceMatcher

CROP_TOP_RATIO = 0.75
SAMPLE_INTERVAL_SEC = 0.2  # giảm từ 0.5 xuống 0.2 để bắt đúng thời điểm sub xuất hiện,
                            # tránh trễ tới ~1s do lấy mẫu quá thưa (đổi lại OCR chạy lâu hơn ~2.5x)
SIMILARITY_THRESHOLD = 0.6
MIN_TEXT_LEN = 1
MAX_OCR_WIDTH = 960


def _fix_windows_dll_path():
    if sys.platform != "win32":
        return
    for pkg in ["cudnn", "cublas"]:
        try:
            mod = __import__(f"nvidia.{pkg}", fromlist=[pkg])
            bin_dir = os.path.join(os.path.dirname(mod.__file__), "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
        except ImportError:
            pass


def _get_video_fps_and_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frame_count


def _text_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _normalize_for_compare(text):
    """Bỏ khoảng trắng và các dấu câu vụn (gạch ngang, chấm câu...) trước khi so sánh,
    vì OCR đôi khi đọc lệch mấy ký tự này (—  vs -, có/không có space...) dù cùng 1 câu."""
    import re
    return re.sub(r'[\s\-—.,;:!?…"\'"''、。！？]', '', text)


def _is_same_sentence(a, b):
    """
    Coi 2 đoạn text là CÙNG 1 câu sub gốc nếu:
    1. Similarity (sau khi chuẩn hóa) đủ cao, HOẶC
    2. 1 trong 2 là "chứa" trong cái kia (OCR đọc thiếu/dư chữ nhiễu ở đầu/cuối
       do vật thể che khuất, hiệu ứng chuyển cảnh...)
    """
    na, nb = _normalize_for_compare(a), _normalize_for_compare(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    return _text_similarity(na, nb) >= SIMILARITY_THRESHOLD


def _init_ocr_engine(lang="ch"):
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_textline_orientation=False,
        lang=lang,
        device="gpu",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
    )


def _ocr_frame(ocr_engine, frame):
    h, w = frame.shape[:2]
    crop = frame[int(h * CROP_TOP_RATIO):h, 0:w]
    ch, cw = crop.shape[:2]
    if cw > MAX_OCR_WIDTH:
        scale = MAX_OCR_WIDTH / cw
        crop = cv2.resize(crop, (MAX_OCR_WIDTH, int(ch * scale)))
    result = ocr_engine.predict(crop)
    if not result:
        return ""
    texts = []
    for res in result:
        texts.extend(res.get("rec_texts", []))
    return " ".join(texts).strip()


def extract_hardsub(video_path, lang="ch"):
    print(f"[OCR-worker] Bắt đầu đọc hardsub từ: {video_path}", flush=True)
    ocr_engine = _init_ocr_engine(lang)
    fps, frame_count = _get_video_fps_and_frame_count(video_path)
    frame_interval = max(1, int(fps * SAMPLE_INTERVAL_SEC))
    total_duration = frame_count / fps

    cap = cv2.VideoCapture(video_path)
    raw_entries = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            text = _ocr_frame(ocr_engine, frame)
            raw_entries.append((timestamp, text))
            if frame_idx % (frame_interval * 20) == 0:
                print(f"[OCR-worker] {timestamp:.1f}s / {total_duration:.1f}s...", flush=True)
        frame_idx += 1
    cap.release()

    segments = []
    current_texts = []  # list các text thô trong đoạn hiện tại, để vote chọn bản sạch nhất
    current_start = None
    last_timestamp = 0

    def _finalize_segment(texts, start, end):
        from collections import Counter
        # Vote theo bản chuẩn hóa (bỏ dấu câu/khoảng trắng vụn) để chọn nội dung
        # xuất hiện nhiều nhất, sau đó lấy bản gốc (có dấu câu) dài nhất tương ứng
        norm_counts = Counter(_normalize_for_compare(t) for t in texts)
        best_norm, _ = norm_counts.most_common(1)[0]
        candidates = [t for t in texts if _normalize_for_compare(t) == best_norm]
        best_text = max(candidates, key=len)
        segments.append({"start": start, "end": end, "text": best_text})

    for timestamp, text in raw_entries:
        if not text or len(text) < MIN_TEXT_LEN:
            if current_texts:
                _finalize_segment(current_texts, current_start, last_timestamp)
                current_texts = []
                current_start = None
            last_timestamp = timestamp
            continue

        if not current_texts:
            current_texts = [text]
            current_start = timestamp
        elif _is_same_sentence(current_texts[-1], text):
            current_texts.append(text)
        else:
            _finalize_segment(current_texts, current_start, timestamp)
            current_texts = [text]
            current_start = timestamp
        last_timestamp = timestamp

    if current_texts:
        _finalize_segment(current_texts, current_start, last_timestamp)

    print(f"[OCR-worker] Xong. Tổng {len(segments)} đoạn sub.", flush=True)
    return {"segments": segments}


if __name__ == "__main__":
    video_path = sys.argv[1]
    output_json_path = sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else "ch"
    result = extract_hardsub(video_path, lang)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OCR-worker] Đã ghi kết quả vào {output_json_path}", flush=True)