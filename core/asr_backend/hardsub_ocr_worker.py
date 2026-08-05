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
import paddle

import json
import cv2
from difflib import SequenceMatcher

CROP_TOP_RATIO = 0.75
SAMPLE_INTERVAL_SEC = 0.2
SIMILARITY_THRESHOLD = 0.6
MIN_TEXT_LEN = 1
MAX_OCR_WIDTH = 960

REGION_SAFETY_PAD_RATIO = 0.08
REGION_SAFETY_PAD_MIN = 6


def _get_video_fps_and_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frame_count


def _text_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _normalize_for_compare(text):
    import re
    return re.sub(r'[\s\-—.,;:!?…"\'"''、。！？]', '', text)


def _is_same_sentence(a, b):
    na, nb = _normalize_for_compare(a), _normalize_for_compare(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    return _text_similarity(na, nb) >= SIMILARITY_THRESHOLD


def _init_rec_engine():
    from paddlex import create_model
    return create_model(model_name="PP-OCRv5_server_rec", device="gpu")


def _compute_region_crop(frame, region, pad_ratio_val=REGION_SAFETY_PAD_RATIO,
                          pad_min_val=REGION_SAFETY_PAD_MIN):
    h, w = frame.shape[:2]
    y1 = int(h * region["top"])
    y2 = int(h * region["bottom"])
    x1 = int(w * region["left"])
    x2 = int(w * region["right"])

    region_h = y2 - y1
    pad = max(int(region_h * pad_ratio_val), pad_min_val)

    new_y1 = max(0, y1 - pad)
    new_y2 = min(h, y2 + pad)

    return frame[new_y1:new_y2, x1:x2]


def _resize_for_ocr(crop):
    ch, cw = crop.shape[:2]
    if cw > MAX_OCR_WIDTH:
        scale = MAX_OCR_WIDTH / cw
        crop = cv2.resize(crop, (MAX_OCR_WIDTH, int(ch * scale)))
    return crop


def _run_ocr_rec_only(rec_engine, crop):
    result = rec_engine.predict(crop)
    if not result:
        return ""
    texts = []
    for res in result:
        text = res.get("rec_text", "")
        if text:
            texts.append(text)
    return " ".join(texts).strip()


def _ocr_frame(rec_engine, frame, region=None):
    h, w = frame.shape[:2]

    if region:
        crop = _compute_region_crop(frame, region)
    else:
        crop = frame[int(h * CROP_TOP_RATIO):h, 0:w]

    crop_for_ocr = _resize_for_ocr(crop)

    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(crop_for_ocr, cv2.COLOR_BGR2GRAY)

    # Cân bằng histogram
    gray = cv2.equalizeHist(gray)

    # OCR vẫn cần ảnh 3 kênh
    crop_for_ocr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    text = _run_ocr_rec_only(rec_engine, crop_for_ocr)
    return text


def extract_hardsub(video_path, lang="ch", region=None):
    print(f"[OCR-worker] Bắt đầu đọc hardsub từ: {video_path}", flush=True)
    if region:
        print(f"[OCR-worker] Dùng vùng quét tuỳ chỉnh: {region}", flush=True)
    print("[OCR-worker] Chế độ: recognition-only (bỏ qua detect)", flush=True)

    rec_engine = _init_rec_engine()

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
            text = _ocr_frame(rec_engine, frame, region)
            raw_entries.append((timestamp, text))
            if frame_idx % (frame_interval * 20) == 0:
                print(f"[OCR-worker] {timestamp:.1f}s / {total_duration:.1f}s...", flush=True)
        frame_idx += 1
    cap.release()

    segments = []
    current_texts = []
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
    region = None
    if len(sys.argv) > 4 and sys.argv[4]:
        region = json.loads(sys.argv[4])
    result = extract_hardsub(video_path, lang, region)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[OCR-worker] Đã ghi kết quả vào {output_json_path}", flush=True)