import os, re, json, shutil, tempfile, subprocess, time
import concurrent.futures
import pandas as pd
import cv2
from rich import print as rprint
from core.utils import load_key, update_key
from core.utils.models import (
    _8_1_AUDIO_TASK, _OUTPUT_DIR, _AUDIO_DIR, _RAW_AUDIO_FILE,
)

SRC_SRT_PATH    = os.path.join(_OUTPUT_DIR, "src.srt")
TRANS_SRT_PATH  = os.path.join(_OUTPUT_DIR, "trans.srt")
TRANS_AUDIO_SRT = os.path.join(_AUDIO_DIR, "trans_subs_for_audio.srt")
SRC_AUDIO_SRT   = os.path.join(_AUDIO_DIR, "src_subs_for_audio.srt")
SRC_SYNC_JSON = os.path.join(_OUTPUT_DIR, "log", "src_sync.json")
SRC_CLEANED_CHUNKS = os.path.join(_OUTPUT_DIR, "log", "src_cleaned_chunks.xlsx")
SRC_NOISE_JSON = os.path.join(_OUTPUT_DIR, "log", "src_noise_ranges.json")

def _cfg(key, fallback=None):
    try: return load_key(key)
    except Exception: return fallback

def _is_portrait_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return h > w
    except Exception:
        return False

MAX_DISPLAY_CHARS  = 200
TOLERANCE      = _cfg("tolerance", 1.5)
MIN_SUB_DUR    = _cfg("min_subtitle_duration", 2.5)

def _get_paths(session_id=None):
    if session_id:
        base    = os.path.join("output", "sessions", session_id)
        audio   = os.path.join(base, "audio")
        return {
            "output_dir":      base,
            "audio_dir":       audio,
            "raw_audio":       os.path.join(audio, "raw.mp3"),
            "cleaned_chunks":  os.path.join(base, "chunks.xlsx"),
            "translation":     os.path.join(base, "translation_results.xlsx"),
            "split_sub":       os.path.join(base, "split_sub.xlsx"),
            "remerged":        os.path.join(base, "remerged.xlsx"),
            "audio_task":      os.path.join(base, "tts_tasks.xlsx"),
            "src_srt":         os.path.join(base, "src.srt"),
            "trans_srt":       os.path.join(base, "trans.srt"),
            "trans_audio_srt": os.path.join(audio, "trans_subs_for_audio.srt"),
            "src_audio_srt":   os.path.join(audio, "src_subs_for_audio.srt"),
        }
    return {
        "output_dir":      _OUTPUT_DIR,
        "audio_dir":       _AUDIO_DIR,
        "raw_audio":       _RAW_AUDIO_FILE,
        "cleaned_chunks":  SRC_CLEANED_CHUNKS,
        "src_sync":        SRC_SYNC_JSON,
        "audio_task":      _8_1_AUDIO_TASK,
        "src_srt":         SRC_SRT_PATH,
        "trans_srt":       TRANS_SRT_PATH,
        "trans_audio_srt": TRANS_AUDIO_SRT,
        "src_audio_srt":   SRC_AUDIO_SRT,
    }

def _srt_time(s):
    h=int(s//3600); m=int((s%3600)//60); sec=s%60; ms=int(round((sec%1)*1000))
    return f"{h:02d}:{m:02d}:{int(sec):02d},{ms:03d}"

def _srt_time_dot(s):
    h=int(s//3600); m=int((s%3600)//60); sec=s%60; ms=int(round((sec%1)*1000))
    return f"{h:02d}:{m:02d}:{int(sec):02d}.{ms:03d}"

def _parse_dot_time(s):
    s=str(s).strip(); h,m,rest=s.split(":"); sec,ms=rest.split(".")
    return int(h)*3600+int(m)*60+int(sec)+int(ms)/1000

def _split_sub_for_display(text, start_s, end_s, max_chars=MAX_DISPLAY_CHARS):
    text = text.strip()
    # QUAN TRỌNG: KHÔNG được return [] khi text rỗng - src_display/vi_display
    # phải luôn sinh cùng số lượng entry 1-1 theo index, nếu không src.srt và
    # trans.srt lệch số dòng khi ghi file.
    if not text:
        return [{"start": start_s, "end": end_s, "text": " "}]
    raw_parts = re.split(r'(?<=[\.\!\?,;:…])\s+', text)
    raw_parts = [p.strip() for p in raw_parts if p.strip()] or [text]
    lines, cur = [], ""
    for p in raw_parts:
        if cur and len(cur) + 1 + len(p) > max_chars:
            lines.append(cur); cur = p
        elif len(p) > max_chars:
            if cur: lines.append(cur); cur = ""
            words, line = p.split(), ""
            for w in words:
                if line and len(line) + 1 + len(w) > max_chars:
                    lines.append(line); line = w
                else:
                    line = (line + " " + w).strip() if line else w
            if line: cur = line
        else:
            cur = (cur + " " + p).strip() if cur else p
    if cur: lines.append(cur)
    if not lines:
        lines = [text]
    display_text = "\n".join(lines[:2])
    return [{"start": start_s, "end": end_s, "text": display_text}]

def _split_text_for_sub(text, max_chars=42):
    text = text.strip()
    if not text: return [text]
    raw_parts = re.split(r'(?<=[\.\!\?,;:…])\s+', text)
    raw_parts = [p.strip() for p in raw_parts if p.strip()] or [text]
    lines, cur = [], ""
    for p in raw_parts:
        if cur and len(cur) + 1 + len(p) > max_chars:
            lines.append(cur); cur = p
        elif len(p) > max_chars:
            if cur: lines.append(cur); cur = ""
            words, line = p.split(), ""
            for w in words:
                if line and len(line) + 1 + len(w) > max_chars:
                    lines.append(line); line = w
                else:
                    line = (line + " " + w).strip() if line else w
            if line: cur = line
        else:
            cur = (cur + " " + p).strip() if cur else p
    if cur: lines.append(cur)
    return lines if lines else [text]

def _write_srt(segments, path, text_key):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines=[]
    for i,seg in enumerate(segments,1):
        lines.append(str(i)); lines.append(f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}")
        lines.append(seg[text_key]); lines.append("")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines))

def _write_cleaned_chunks(data):
    rows=[{"text":f'"{d["src"]}"',"start":d["start"],"end":d["end"]} for d in data]
    df=pd.DataFrame(rows); os.makedirs(os.path.dirname(SRC_CLEANED_CHUNKS),exist_ok=True)
    df.to_excel(SRC_CLEANED_CHUNKS,index=False)
    rprint(f"[green]src_cleaned_chunks.xlsx -> {SRC_CLEANED_CHUNKS} ({len(rows)} rows)[/green]")

def _write_src_sync(data):
    rows = [{"start": d["start"], "end": d["end"], "src": d["src"], "vi": d["vi"]} for d in data]
    os.makedirs(os.path.dirname(SRC_SYNC_JSON), exist_ok=True)
    with open(SRC_SYNC_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    rprint(f"[green]src_sync.json -> {SRC_SYNC_JSON} ({len(rows)} rows)[/green]")

def _write_subtitles(data, is_portrait=False):
    os.makedirs(_OUTPUT_DIR,exist_ok=True); os.makedirs(_AUDIO_DIR,exist_ok=True)
    display_max_chars = MAX_DISPLAY_CHARS if is_portrait else 60
    src_display,vi_display=[],[]
    for d in data:
        for chunk in _split_sub_for_display(d["src"],d["start"],d["end"],max_chars=display_max_chars): src_display.append(chunk)
        for chunk in _split_sub_for_display(d["vi"],d["start"],d["end"],max_chars=display_max_chars): vi_display.append(chunk)
    def _clamp(segs):
        for i in range(len(segs)-1):
            if segs[i]["end"]>segs[i+1]["start"]: segs[i]["end"]=max(segs[i+1]["start"],segs[i]["start"]+0.3)
    _clamp(src_display); _clamp(vi_display)
    _write_srt(src_display,SRC_SRT_PATH,"text"); _write_srt(vi_display,TRANS_SRT_PATH,"text")
    _write_srt(data,SRC_AUDIO_SRT,"src"); _write_srt(data,TRANS_AUDIO_SRT,"vi")
    rprint(f"[green]SRT files written: {SRC_SRT_PATH}, {TRANS_SRT_PATH}, {SRC_AUDIO_SRT}, {TRANS_AUDIO_SRT}[/green]")

def _merge_short_rows(data, effective_min_dur):
    rows = [{"start": d["start"], "end": d["end"], "src": d["src"], "vi": d["vi"]} for d in data]
    i = 0
    while i < len(rows):
        dur = rows[i]["end"] - rows[i]["start"]
        if dur < effective_min_dur:
            if i + 1 < len(rows):
                combined = rows[i + 1]["end"] - rows[i]["start"]
                if combined < effective_min_dur * 2:
                    rows[i]["vi"] += " " + rows[i + 1]["vi"]
                    rows[i]["src"] += " " + rows[i + 1]["src"]
                    rows[i]["end"] = rows[i + 1]["end"]
                    rows.pop(i + 1)
                    continue
                else:
                    rows[i]["end"] = rows[i]["start"] + effective_min_dur
        i += 1
    if len(rows) >= 2 and (rows[-1]["end"] - rows[-1]["start"]) < effective_min_dur:
        rows[-2]["vi"] += " " + rows[-1]["vi"]
        rows[-2]["src"] += " " + rows[-1]["src"]
        rows[-2]["end"] = rows[-1]["end"]
        rows.pop()
    return rows


from threading import Lock as _Lock
from google import genai as google_genai
from google.genai import types as google_genai_types

_GEMINI_KEY_INDEX = 0
_GEMINI_KEY_LOCK = _Lock()

def _get_gemini_keys():
    # Dùng chung key với api.key/api.keys ở đầu config (giống translate_lines) -
    # không tách 1 bộ key Gemini riêng cho watermark filter nữa.
    keys_list = _cfg("api.keys", None)
    if isinstance(keys_list, list) and keys_list:
        return [k.strip() for k in keys_list if str(k).strip()]
    raw = _cfg("api.key", "") or os.environ.get("GEMINI_API_KEY", "")
    if not raw:
        return []
    parts = re.split(r"[,\n;\s]+", raw.strip())
    return [p for p in parts if p]

def _get_next_gemini_key(keys):
    global _GEMINI_KEY_INDEX
    with _GEMINI_KEY_LOCK:
        key = keys[_GEMINI_KEY_INDEX % len(keys)]
        _GEMINI_KEY_INDEX += 1
        return key

def _gemini_generate(prompt, max_output_tokens=65536):
    keys = _get_gemini_keys()
    if not keys:
        raise RuntimeError("API key not set (api.key / api.keys)")
    last_error = None
    for _ in range(len(keys)):
        api_key = _get_next_gemini_key(keys)
        try:
            client = google_genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=google_genai_types.GenerateContentConfig(
                    response_mime_type="application/json", max_output_tokens=max_output_tokens))
            return resp.text
        except Exception as e:
            err_str = str(e)
            is_key_issue = any(s in err_str for s in
                ["API_KEY_INVALID", "429", "RESOURCE_EXHAUSTED", "quota", "rate"])
            masked = api_key[-6:] if len(api_key) >= 6 else api_key
            if is_key_issue and len(keys) > 1:
                rprint(f"[yellow]⚠ Key ...{masked} lỗi/hết quota, xoay sang key khác...[/yellow]")
                last_error = e
                continue
            raise e
    raise last_error or RuntimeError("Tất cả API key đều lỗi")

GEMINI_MODEL   = _cfg("api.model", "gemini-3.1-flash-lite")
SYLLABLE_RATE = _cfg("src_pipeline.syllable_rate", 4.0)

def _translate_one_batch(data, bstart, chunk_size):
    """Dịch 1 batch bằng translate_lines() - lõi dịch gốc của VideoLingo
    (faithfulness + reflect/expressiveness 2 bước), dùng làm task cho
    ThreadPoolExecutor trong pipeline_goc_translate."""
    from core.translate_lines import translate_lines

    batch = data[bstart:bstart + chunk_size]
    lines = "\n".join(d["src"] for d in batch)

    prev_batch = data[bstart - 3:bstart] if bstart > 0 else None
    previous_content_prompt = [d["src"] for d in prev_batch] if prev_batch else None
    next_batch = data[bstart + chunk_size:bstart + chunk_size + 2]
    after_content_prompt = [d["src"] for d in next_batch] if next_batch else None

    translation, _ = translate_lines(
        lines, previous_content_prompt, after_content_prompt,
        things_to_note_prompt=None, summary_prompt=None,
        index=bstart // chunk_size,
    )
    vi_lines = translation.split("\n")
    if len(vi_lines) != len(batch):
        raise RuntimeError(
            f"pipeline_goc_translate: batch {bstart} returned {len(vi_lines)} lines, "
            f"expected {len(batch)} - data may be misaligned."
        )

    batch_result = []
    for d, vi in zip(batch, vi_lines):
        vi = vi.strip()
        if not vi:
            # Câu dịch rỗng (thường do OCR bắt watermark/rác) -> fallback về
            # câu gốc để không bao giờ rỗng, tránh lệch dòng khi ghi srt.
            vi = d["src"]
        batch_result.append({"start": d["start"], "end": d["end"], "src": d["src"], "vi": vi})
    return bstart, batch_result


_REVIEW_PROMPT = """\
Bạn là biên tập viên lồng tiếng chuyên nghiệp Trung → Việt, đang xem lại bản
dịch nháp của cả 1 đoạn video theo đúng thứ tự thời gian (timeline).

## Nhiệm vụ
Với MỖI dòng, dựa vào câu gốc tiếng Trung, bản dịch nháp hiện tại, và ngữ cảnh
các câu xung quanh (được cho theo đúng thứ tự), hãy:

1. **Phát hiện rác/watermark - CỰC KỲ THẬN TRỌNG**: chỉ đánh dấu
   "remove": true khi bạn CHẮC CHẮN GẦN TUYỆT ĐỐI đây không phải lời thoại
   thực - ví dụ: cùng 1 chuỗi tên kênh/watermark lặp lại y hệt nhau ở nhiều
   dòng khác nhau trong video, hoặc rõ ràng là logo/URL/tên tài khoản không
   mang nghĩa gì cả.
   **MẶC ĐỊNH LÀ GIỮ LẠI ("remove": false) NẾU CÒN CHÚT NGHI NGỜ** - kể cả
   khi câu đó ngắn, có vẻ lạc quẻ, hay không rõ liên quan mạch truyện, VẪN
   GIỮ LẠI trừ khi chắc chắn tuyệt đối là rác. Xóa nhầm 1 câu thoại thật gây
   hậu quả NẶNG HƠN NHIỀU so với việc để sót 1 dòng rác (có thể xóa tay sau).
2. **Tinh chỉnh bản dịch**: viết lại "vi" cho tự nhiên, có hồn hơn, được phép
   sáng tạo thêm chút ý nếu giúp câu nói mượt mà, sinh động hơn (như cách biên
   tập viên lồng tiếng chuyên nghiệp hay làm) - miễn giữ đúng Ý CHÍNH của câu
   gốc và khớp mạch truyện với câu trước/sau. KHÔNG bịa thêm nội dung không có
   trong câu gốc, chỉ được diễn đạt lại cho hay hơn.
3. Vẫn giữ ràng buộc: duration × {syllable_rate} ≈ số âm tiết tiếng Việt cần có.

## Output — JSON list, đúng thứ tự input, đủ cho MỌI dòng:
[{{"i": 0, "vi": "...", "remove": false}}, ...]
KHÔNG thêm gì khác ngoài JSON.

## Input (câu gốc, bản dịch nháp, duration, theo đúng timeline)
{input_json}
"""

def gemini_watermark_filter_only(data, client=None, batch_size=80):
    """
    Chỉ lấy phần "phát hiện rác/watermark" của gemini_review_pass, KHÔNG lấy
    phần "viết lại vi" - dùng sau pipeline_goc_translate để giữ nguyên bản
    dịch chất lượng cao của pipeline gốc, chỉ mượn Gemini để lọc rác OCR
    (watermark/logo lặp lại) mà pipeline gốc không tự phát hiện được vì nó
    dịch từng câu độc lập, không có bước xem lại toàn video theo timeline.
    """
    rprint(f"[cyan]🔎 Lọc watermark/rác (không đổi bản dịch): xem lại {len(data)} câu...[/cyan]")
    all_remove = {}
    for bstart in range(0, len(data), batch_size):
        batch = data[bstart:bstart + batch_size]
        items = [
            {"i": bstart + j, "src": d["src"], "vi_draft": d["vi"], "duration": round(d["end"] - d["start"], 2)}
            for j, d in enumerate(batch)
        ]
        prompt = _REVIEW_PROMPT.replace("{syllable_rate}", str(SYLLABLE_RATE)).replace("{input_json}", json.dumps(items, ensure_ascii=False))
        for attempt in range(3):
            try:
                resp_text = _gemini_generate(prompt)
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp_text.strip())
                rows = json.loads(raw)
                if not isinstance(rows, list):
                    raise ValueError("Not a list")
                for row in rows:
                    if "i" not in row:
                        continue
                    if row.get("remove", False):
                        all_remove[row["i"]] = True
                break
            except Exception as e:
                rprint(f"[yellow]⚠ Lọc watermark lỗi batch (lần {attempt+1}): {e}[/yellow]")
                time.sleep(1.5)

    result, removed_count = [], 0
    for i, d in enumerate(data):
        if all_remove.get(i, False):
            removed_count += 1
            rprint(f"   [{i+1:03d}] [-] Loại bỏ (nghi watermark/rác): {d['src'][:40]}")
            continue
        result.append(d)  # giữ nguyên y hệt bản dịch pipeline gốc, không đổi "vi"

    if removed_count:
        rprint(f"[yellow]ℹ Đã loại {removed_count} câu nghi là watermark/rác[/yellow]")
    rprint(f"[green]✅ Lọc watermark hoàn tất → còn {len(result)} câu[/green]")
    return result


def pipeline_goc_translate(data):
    """Dịch OCR bằng đúng lõi dịch của pipeline gốc VideoLingo
    (translate_lines - faithfulness + reflect/expressiveness check 2 bước).
    Các batch chạy song song qua ThreadPoolExecutor(max_workers), dùng config
    "max_workers" có sẵn - giống cách _4_2_translate.py (pipeline gốc) làm."""
    chunk_size = _cfg("src_pipeline.pipeline_goc_chunk_size", 10)
    max_workers = _cfg("max_workers", 1)
    batch_starts = list(range(0, len(data), chunk_size))

    results_by_start = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_translate_one_batch, data, bstart, chunk_size): bstart
                   for bstart in batch_starts}
        for future in concurrent.futures.as_completed(futures):
            bstart, batch_result = future.result()
            results_by_start[bstart] = batch_result

    result = []
    for bstart in batch_starts:
        result.extend(results_by_start[bstart])
    return result


def src_asr_and_translate(session_id=None):
    """Nguồn transcribe: OCR hardsub (thay cho WhisperX). Dịch + lọc watermark
    đều gọi qua ask_gpt (đúng LLM gốc VideoLingo), không còn Gemini/GoogleTranslator
    hay API riêng nào khác."""
    from core._1_ytdlp import find_video_files
    video_path=find_video_files()
    tmp_dir=tempfile.mkdtemp(prefix="src_pipe_")
    try:
        os.makedirs(_AUDIO_DIR,exist_ok=True)

        if not os.path.exists(_RAW_AUDIO_FILE):
            subprocess.run(["ffmpeg","-i",video_path,"-vn","-acodec","libmp3lame","-q:a","2","-y",_RAW_AUDIO_FILE],
                           capture_output=True, check=True)

        from core.asr_backend.hardsub_ocr import extract_hardsub
        ocr_region = _cfg("ocr_region", None)
        ocr_result = extract_hardsub(video_path, lang="ch", region=ocr_region)
        raw_segments = ocr_result.get("segments", [])
        data = [
            {"start": seg["start"], "end": seg["end"], "src": seg["text"].strip()}
            for seg in raw_segments if seg.get("text", "").strip()
        ]
        if not data: raise RuntimeError("OCR did not detect any subtitle in the video")

        data = pipeline_goc_translate(data)
        if not data: raise RuntimeError("pipeline_goc_translate returned no translated lines")

        # Pipeline gốc không tự lọc watermark/rác OCR (chỉ có ở nhánh ASR gốc
        # nhờ VAD loại tiếng ồn) - bổ sung riêng cho nhánh OCR ở đây, giữ
        # nguyên bản gốc: dùng Gemini riêng, KHÔNG đổi bản dịch pipeline_goc.
        data = gemini_watermark_filter_only(data)
        if not data: raise RuntimeError("gemini_watermark_filter_only removed all lines")

        is_portrait = _is_portrait_video(video_path)

        ocr_min_dur = _cfg("src_pipeline.ocr_min_subtitle_duration", 0.8)
        data = _merge_short_rows(data, ocr_min_dur)

        _write_cleaned_chunks(data); _write_src_sync(data); _write_subtitles(data, is_portrait=is_portrait)
    finally:
        shutil.rmtree(tmp_dir,ignore_errors=True)


def src_gen_audio_tasks(session_id=None):
    from core.tts_backend.estimate_duration import init_estimator, estimate_duration
    from core.asr_backend.audio_preprocess import get_audio_duration
    with open(SRC_SYNC_JSON, encoding="utf-8") as f:
        sync_rows = json.load(f)
    estimator=init_estimator()
    accept=_cfg("speed_factor.accept",1.2); tol_cfg=_cfg("tolerance",1.5)
    whole_dur=get_audio_duration(_RAW_AUDIO_FILE)
    rows=[]
    for i,r in enumerate(sync_rows):
        rows.append({"number":i+1,"text":r["vi"],"origin":r["src"],
                     "start_time":_srt_time_dot(r["start"]),"end_time":_srt_time_dot(r["end"]),
                     "duration":round(r["end"]-r["start"],3)})
    df=pd.DataFrame(rows); n=len(df)
    df["gap"]=0.0
    for i in range(n-1):
        df.loc[i,"gap"]=max(_parse_dot_time(df.loc[i+1,"start_time"])-_parse_dot_time(df.loc[i,"end_time"]),0.0)
    df.loc[n-1,"gap"]=max(whole_dur-_parse_dot_time(df.loc[n-1,"end_time"]),0.0)
    df["tolerance"]=df["gap"].apply(lambda x:tol_cfg if x>tol_cfg else x)
    df["tol_dur"]=df["duration"]+df["tolerance"]
    df["est_dur"]=df["text"].apply(lambda t:estimate_duration(str(t),estimator))
    def _if_too_fast(row):
        est,tol,dur,tol_v=row["est_dur"],row["tol_dur"],row["duration"],row["tolerance"]
        if est/accept>tol: return 2
        elif est>tol: return 1
        elif est<dur-tol_v: return -1
        else: return 0
    df["if_too_fast"]=df.apply(_if_too_fast,axis=1)
    df["cut_off"]=1
    tts_max_chars = _cfg("src_pipeline.ocr_tts_split_max_chars", 42)
    df["lines"]=df["text"].apply(lambda t:_split_text_for_sub(str(t), max_chars=tts_max_chars))
    df["src_lines"]=df["origin"].apply(lambda t:[str(t)])
    df["real_dur"]=0.0; df["new_sub_times"]=None
    os.makedirs(os.path.dirname(_8_1_AUDIO_TASK),exist_ok=True)
    df.to_excel(_8_1_AUDIO_TASK,index=False)

    sync_data = []
    for _, row in df.iterrows():
        sync_data.append({"start": _parse_dot_time(row["start_time"]),
                          "end": _parse_dot_time(row["end_time"]),
                          "src": str(row["origin"]), "vi": str(row["text"])})
    _write_srt(sync_data, SRC_SRT_PATH, "src")
    _write_srt(sync_data, TRANS_SRT_PATH, "vi")
    src_display, vi_display = [], []
    for d in sync_data:
        for chunk in _split_sub_for_display(d["src"], d["start"], d["end"]): src_display.append(chunk)
        for chunk in _split_sub_for_display(d["vi"], d["start"], d["end"]): vi_display.append(chunk)
    _write_srt(src_display, SRC_AUDIO_SRT, "text")
    _write_srt(vi_display, TRANS_AUDIO_SRT, "text")