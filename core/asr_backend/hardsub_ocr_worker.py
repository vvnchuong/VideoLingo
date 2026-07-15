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

CROP_TOP_RATIO = 0.75  # fallback khi không có vùng quét tuỳ chỉnh (giữ tương thích ngược)
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


# def _ocr_frame(ocr_engine, frame, region=None):
#     """
#     region: dict {top, bottom, left, right} dạng tỉ lệ 0.0-1.0, hoặc None để dùng
#     fallback CROP_TOP_RATIO cũ (quét full chiều ngang, từ CROP_TOP_RATIO xuống đáy).
#     """
#     h, w = frame.shape[:2]
#     if region:
#         y1 = int(h * region["top"])
#         y2 = int(h * region["bottom"])
#         x1 = int(w * region["left"])
#         x2 = int(w * region["right"])
#
#         region_h = y2 - y1
#         pad_ratio = int(region_h * 0.25)
#         pad_min = 20
#         pad = max(pad_ratio, pad_min)
#         y1 = max(0, y1 - pad)
#         y2 = min(h, y2 + pad)
#
#         crop = frame[y1:y2, x1:x2]
#     else:
#         crop = frame[int(h * CROP_TOP_RATIO):h, 0:w]
#
#     ch, cw = crop.shape[:2]
#     if cw > MAX_OCR_WIDTH:
#         scale = MAX_OCR_WIDTH / cw
#         crop = cv2.resize(crop, (MAX_OCR_WIDTH, int(ch * scale)))
#     result = ocr_engine.predict(crop)
#     if not result:
#         return ""
#     texts = []
#     for res in result:
#         texts.extend(res.get("rec_texts", []))
#     return " ".join(texts).strip()

def _compute_padded_crop(frame, region, pad_ratio_val, pad_min_val):
    """Tính vùng crop đã pad, tự bù sang phía không bị chặn nếu 1 phía chạm biên frame."""
    h, w = frame.shape[:2]
    y1 = int(h * region["top"])
    y2 = int(h * region["bottom"])
    x1 = int(w * region["left"])
    x2 = int(w * region["right"])

    region_h = y2 - y1
    pad = max(int(region_h * pad_ratio_val), pad_min_val)

    new_y1 = max(0, y1 - pad)
    new_y2 = min(h, y2 + pad)

    actual_pad_top = y1 - new_y1
    actual_pad_bottom = new_y2 - y2
    deficit = (pad - actual_pad_top) + (pad - actual_pad_bottom)
    if deficit > 0:
        if actual_pad_bottom < pad:
            new_y1 = max(0, new_y1 - deficit)
        elif actual_pad_top < pad:
            new_y2 = min(h, new_y2 + deficit)

    return frame[new_y1:new_y2, x1:x2]


def _resize_for_ocr(crop):
    ch, cw = crop.shape[:2]
    if cw > MAX_OCR_WIDTH:
        scale = MAX_OCR_WIDTH / cw
        crop = cv2.resize(crop, (MAX_OCR_WIDTH, int(ch * scale)))
    return crop


def _run_ocr(ocr_engine, crop):
    result = ocr_engine.predict(crop)
    if not result:
        return ""
    texts = []
    for res in result:
        texts.extend(res.get("rec_texts", []))
    return " ".join(texts).strip()


def _ocr_frame(ocr_engine, frame, region=None):
    """
    region: dict {top, bottom, left, right} dạng tỉ lệ 0.0-1.0, hoặc None để dùng
    fallback CROP_TOP_RATIO cũ (quét full chiều ngang, từ CROP_TOP_RATIO xuống đáy).

    Khi có region: pad thêm biên trên/dưới trước khi đưa vào OCR, vì PaddleOCR
    (PP-OCRv5 detection) hay bỏ sót text khi ảnh crop quá sát/khít vào chữ, đặc
    biệt khi vùng crop rất mỏng (ratio width/height quá lớn). Nếu lần đầu (pad nhẹ)
    vẫn không đọc được gì, retry với pad rộng hơn trước khi chấp nhận bỏ trống.
    """
    h, w = frame.shape[:2]
    if region:
        crop = _compute_padded_crop(frame, region, pad_ratio_val=0.3125, pad_min_val=60)
        text = _run_ocr(ocr_engine, crop)
        if not text:
            crop = _compute_padded_crop(frame, region, pad_ratio_val=0.6, pad_min_val=120)
            text = _run_ocr(ocr_engine, crop)
        return text
    else:
        crop = frame[int(h * CROP_TOP_RATIO):h, 0:w]
        crop = _resize_for_ocr(crop)
        return _run_ocr(ocr_engine, crop)


def extract_hardsub(video_path, lang="ch", region=None):
    print(f"[OCR-worker] Bắt đầu đọc hardsub từ: {video_path}", flush=True)
    if region:
        print(f"[OCR-worker] Dùng vùng quét tuỳ chỉnh: {region}", flush=True)
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
            text = _ocr_frame(ocr_engine, frame, region)
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
    # region truyền vào dạng JSON string: '{"top":0.6,"bottom":0.8,"left":0.1,"right":0.9}'
    # rỗng/không truyền thì None -> dùng fallback CROP_TOP_RATIO cũ.
    region = None
    if len(sys.argv) > 4 and sys.argv[4]:
        region = json.loads(sys.argv[4])
    result = extract_hardsub(video_path, lang, region)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OCR-worker] Đã ghi kết quả vào {output_json_path}", flush=True)