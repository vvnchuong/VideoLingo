import os, re, gc, json, time, asyncio, tempfile, shutil, sys, subprocess, datetime, warnings
import concurrent.futures
import torch, srt, pandas as pd
import cv2
from datetime import timedelta
from rich import print as rprint
from google import genai as google_genai
from google.genai import types as google_genai_types
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from core.utils import load_key, update_key
from core.utils.models import (
    _8_1_AUDIO_TASK, _OUTPUT_DIR, _AUDIO_DIR, _RAW_AUDIO_FILE,
)

SRC_SRT_PATH    = os.path.join(_OUTPUT_DIR, "src.srt")
TRANS_SRT_PATH  = os.path.join(_OUTPUT_DIR, "trans.srt")
TRANS_AUDIO_SRT = os.path.join(_AUDIO_DIR, "trans_subs_for_audio.srt")
SRC_AUDIO_SRT   = os.path.join(_AUDIO_DIR, "src_subs_for_audio.srt")
ZH_SYNC_JSON = os.path.join(_OUTPUT_DIR, "log", "zh_sync.json")
ZH_CLEANED_CHUNKS = os.path.join(_OUTPUT_DIR, "log", "zh_cleaned_chunks.xlsx")
ZH_NOISE_JSON = os.path.join(_OUTPUT_DIR, "log", "zh_noise_ranges.json")

def _cfg(key, fallback=None):
    try: return load_key(key)
    except Exception: return fallback

from threading import Lock as _Lock
_GEMINI_KEY_INDEX = 0
_GEMINI_KEY_LOCK = _Lock()

def _get_gemini_keys():
    keys_list = _cfg("zh_pipeline.gemini_api_keys", None)
    if isinstance(keys_list, list) and keys_list:
        return [k.strip() for k in keys_list if str(k).strip()]
    raw = _cfg("zh_pipeline.gemini_api_key", "") or os.environ.get("GEMINI_API_KEY", "")
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
        raise RuntimeError("Gemini API key not set (zh_pipeline.gemini_api_key)")
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

def _is_portrait_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return h > w
    except Exception:
        return False

VAD_THRESHOLD      = 0.35
VAD_MIN_SILENCE_MS = 300
VAD_SPEECH_PAD_MS  = 100
VAD_MERGE_GAP_S    = 0.12
MAX_VAD_SEGMENT_S  = 8.0
GAP_WARN_THRESHOLD = 8.0
MAX_DISPLAY_CHARS  = 200
SYLLABLE_RATE = _cfg("zh_pipeline.syllable_rate", 4.0)
TRANSLATE_BATCH    = 80
CONTEXT_TAIL       = 4

# ── VRAM: reload faster-whisper sau mỗi N clips ──────────────────────────────
# Tăng lên nếu VRAM lớn, giảm xuống nếu vẫn OOM (thử 30 hoặc 20)
WHISPER_RELOAD_EVERY = 50
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = _cfg("zh_pipeline.gemini_api_key", "")
GEMINI_MODEL   = _cfg("zh_pipeline.gemini_model") or _cfg("api.model", "gemini-3.1-flash-lite")
TOLERANCE      = _cfg("tolerance", 1.5)
MIN_SUB_DUR    = _cfg("min_subtitle_duration", 2.5)

_vad_cache = {"model": None, "utils": None}
_fw_cache  = {"model": None}

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
        "cleaned_chunks":  ZH_CLEANED_CHUNKS,
        "zh_sync":         ZH_SYNC_JSON,
        "audio_task":      _8_1_AUDIO_TASK,
        "src_srt":         SRC_SRT_PATH,
        "trans_srt":       TRANS_SRT_PATH,
        "trans_audio_srt": TRANS_AUDIO_SRT,
        "src_audio_srt":   SRC_AUDIO_SRT,
    }

def _probe_duration(path):
    out = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                          "-of","default=noprint_wrappers=1:nokey=1",path],
                         check=True, capture_output=True, text=True)
    return float(out.stdout.strip())

def _srt_time(s):
    h=int(s//3600); m=int((s%3600)//60); sec=s%60; ms=int(round((sec%1)*1000))
    return f"{h:02d}:{m:02d}:{int(sec):02d},{ms:03d}"

def _srt_time_dot(s):
    h=int(s//3600); m=int((s%3600)//60); sec=s%60; ms=int(round((sec%1)*1000))
    return f"{h:02d}:{m:02d}:{int(sec):02d}.{ms:03d}"

def _parse_dot_time(s):
    s=str(s).strip(); h,m,rest=s.split(":"); sec,ms=rest.split(".")
    return int(h)*3600+int(m)*60+int(sec)+int(ms)/1000

def _gemini_client():
    key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY","")
    if not key: raise RuntimeError("Gemini API key not set")
    return google_genai.Client(api_key=key)

def _is_refusal(text):
    patterns=["vui lòng cung cấp","xin vui lòng cung cấp","tôi cần thêm thông tin",
              "tôi không thể dịch","xin lỗi, tôi","vui lòng cho tôi",
              "cung cấp câu tiếng việt","cần thêm ngữ cảnh","i cannot translate","please provide"]
    return any(p in text.lower().strip() for p in patterns)

def _split_sub_for_display(text, start_s, end_s, max_chars=MAX_DISPLAY_CHARS):
    text = text.strip()
    # QUAN TRỌNG: KHÔNG được return [] khi text rỗng. zh_display và vi_display
    # (gọi hàm này riêng cho "zh" và "vi" của cùng 1 dòng) PHẢI luôn sinh ra
    # cùng số lượng entry, 1-1 theo index - nếu một bên trả về [] còn bên kia
    # trả về 1 entry, 2 file src.srt/trans.srt bị lệch số dòng, gây lỗi "File
    # phụ đề gốc và bản dịch không khớp số dòng" ở SubtitleService.readSubtitle()
    # phía Java. Luôn trả về đúng 1 entry, kể cả khi text rỗng (dùng khoảng
    # trắng để không phá format SRT).
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
    # Giới hạn tối đa 2 dòng hiển thị cùng lúc (giống chuẩn subtitle thông thường)
    display_text = "\n".join(lines[:2])
    return [{"start": start_s, "end": end_s, "text": display_text}]

def _load_vad():
    if _vad_cache["model"] is None:
        model,utils=torch.hub.load("snakers4/silero-vad",model="silero_vad",force_reload=False)
        _vad_cache["model"]=model; _vad_cache["utils"]=utils
    return _vad_cache["model"],_vad_cache["utils"]

def _split_long_segment(seg, raw_segs, max_dur=MAX_VAD_SEGMENT_S):
    dur=seg["end"]-seg["start"]
    if dur<=max_dur: return [seg]
    inner=[r for r in raw_segs if r["start"]>=seg["start"]-1e-6 and r["end"]<=seg["end"]+1e-6]
    if len(inner)<2: return [seg]
    best_gap,best_idx=-1,-1
    for i in range(len(inner)-1):
        gap=inner[i+1]["start"]-inner[i]["end"]
        if gap>best_gap: best_gap,best_idx=gap,i
    if best_idx<0 or best_gap<=0: return [seg]
    cut=(inner[best_idx]["end"]+inner[best_idx+1]["start"])/2
    return (_split_long_segment({"start":seg["start"],"end":cut},raw_segs,max_dur)+
            _split_long_segment({"start":cut,"end":seg["end"]},raw_segs,max_dur))

def _demucs_extract_vocals(audio_path, tmp_dir):
    DEMUCS_MODEL=_cfg("zh_pipeline.demucs_model","htdemucs")
    try:
        out_dir=os.path.join(tmp_dir,"demucs_out")
        import shutil as _sh
        safe=os.path.join(tmp_dir,"vad_input.wav"); _sh.copy2(audio_path,safe)
        result=subprocess.run(
            [sys.executable,"-m","demucs","-n",DEMUCS_MODEL,"--two-stems","vocals","-o",out_dir,safe],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode!=0:
            rprint(f"[yellow]⚠ Demucs lỗi: {result.stderr[:200]}[/yellow]"); return None
        out_path=os.path.join(out_dir,DEMUCS_MODEL,"vad_input","vocals.wav")
        if not os.path.exists(out_path):
            rprint("[yellow]⚠ Demucs không thấy vocals.wav[/yellow]"); return None
        return out_path
    except Exception as e:
        rprint(f"[yellow]⚠ Demucs lỗi: {e} — fallback[/yellow]"); return None

def get_vad_segments(video_path, tmp_dir):
    rprint("[cyan]🎤 [VAD] Định vị vùng nói...[/cyan]")
    audio_temp=os.path.join(tmp_dir,"vad_audio.wav")
    subprocess.run(["ffmpeg","-i",str(video_path),"-ar","16000","-ac","1","-y",audio_temp],
                   capture_output=True, check=True)
    vad_audio=audio_temp
    if _cfg("demucs",True):
        rprint("[cyan][Demucs] Tách vocal...[/cyan]"); t0=time.time()
        vocal=_demucs_extract_vocals(audio_temp,tmp_dir)
        if vocal:
            vad_audio=os.path.join(tmp_dir,"vad_vocal_16k.wav")
            subprocess.run(["ffmpeg","-i",vocal,"-ar","16000","-ac","1","-y",vad_audio],
                           capture_output=True, check=True)
            rprint(f"[green]✅ Demucs xong ({time.time()-t0:.1f}s)[/green]")
        else: vad_audio=audio_temp
    model,utils=_load_vad()
    (get_speech_timestamps,_,read_audio,_,_)=utils
    import wave as _wave, numpy as _np
    with _wave.open(vad_audio,'rb') as wf:
        frames=wf.readframes(wf.getnframes()); sr=wf.getframerate()
    wav_np=_np.frombuffer(frames,dtype=_np.int16).astype(_np.float32)/32768.0
    wav=torch.from_numpy(wav_np)
    ts=get_speech_timestamps(wav,model,threshold=VAD_THRESHOLD,
                             min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                             speech_pad_ms=VAD_SPEECH_PAD_MS)
    raw=[{"start":t["start"]/16000,"end":t["end"]/16000} for t in ts]
    merged=[]
    for seg in raw:
        if merged and seg["start"]-merged[-1]["end"]<=VAD_MERGE_GAP_S: merged[-1]["end"]=seg["end"]
        else: merged.append(dict(seg))
    rprint(f"[cyan]VAD: {len(raw)} thô → {len(merged)} sau gộp[/cyan]")
    split_result=[]
    for seg in merged: split_result.extend(_split_long_segment(seg,raw))
    if len(split_result)!=len(merged):
        rprint(f"[cyan]VAD: {len(merged)} → {len(split_result)} sau tách dài[/cyan]")
    merged=split_result
    rprint(f"[green]✅ VAD: {len(raw)} thô → {len(merged)} segment[/green]")
    for i,seg in enumerate(merged):
        gap=seg["start"]-merged[i-1]["end"] if i>0 else 0
        warn=f"  ⚠ GAP {gap:.1f}s" if gap>GAP_WARN_THRESHOLD else ""
        rprint(f"   [{i+1:03d}] {seg['start']:7.2f}s → {seg['end']:7.2f}s (dài {seg['end']-seg['start']:.1f}s){warn}")
    return merged

def _load_faster_whisper():
    if _fw_cache["model"] is None:
        model_name=_cfg("zh_pipeline.whisper_model_zh") or _cfg("whisper.model","medium")
        device="cuda" if torch.cuda.is_available() else "cpu"
        compute_type="float16" if device=="cuda" else "int8"
        rprint(f"[cyan]⏳ Load faster-whisper '{model_name}' (device={device})...[/cyan]")
        _fw_cache["model"]=WhisperModel(model_name,device=device,compute_type=compute_type)
        rprint("[green]✅ faster-whisper loaded[/green]")
    return _fw_cache["model"]

def _reload_faster_whisper():
    if _fw_cache["model"] is not None:
        del _fw_cache["model"]; _fw_cache["model"] = None
        gc.collect()
        # ── VRAM flush khi reload định kỳ ────────────────────────────────────
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # ─────────────────────────────────────────────────────────────────────
    return _load_faster_whisper()

def _unload_vad():
    if _vad_cache["model"] is not None:
        rprint("[cyan]⏳ Unload VAD...[/cyan]")
        _vad_cache["model"] = None; _vad_cache["utils"] = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        rprint("[green]✅ VAD unloaded[/green]")

def _unload_whisper():
    if _fw_cache["model"] is not None:
        rprint("[cyan]⏳ Unload faster-whisper...[/cyan]")
        del _fw_cache["model"]; _fw_cache["model"] = None
        gc.collect()
        # ── VRAM flush sau khi unload Whisper ────────────────────────────────
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # ─────────────────────────────────────────────────────────────────────
        rprint("[green]✅ Whisper unloaded[/green]")

def _unload_models():
    _unload_whisper()
    _unload_vad()

def _looks_like_noise(text, segments_list, no_speech_threshold=None):
    if no_speech_threshold is None:
        no_speech_threshold = _cfg("zh_pipeline.no_speech_threshold", 0.6)
    if not text.strip():
        return True
    has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in text)
    if not has_cjk:
        return True
    try:
        probs = [getattr(s, "no_speech_prob", 0.0) for s in segments_list]
        if probs and (sum(probs) / len(probs)) > no_speech_threshold:
            return True
    except Exception:
        pass
    return False

def transcribe_vad_clips(video_path, vad_segs, tmp_dir):
    wm = _load_faster_whisper()
    data = []
    noise_ranges = []
    rprint(f"[cyan]🎤 Whisper transcribe {len(vad_segs)} segments (reload mỗi {WHISPER_RELOAD_EVERY} clips)...[/cyan]")
    skipped = 0
    for i, seg in enumerate(vad_segs):
        # ── VRAM: reload định kỳ theo WHISPER_RELOAD_EVERY ───────────────────
        if i > 0 and i % WHISPER_RELOAD_EVERY == 0:
            rprint(f"[cyan]🔄 Reload faster-whisper tại clip {i+1}...[/cyan]")
            wm = _reload_faster_whisper()
        # ─────────────────────────────────────────────────────────────────────
        clip = os.path.join(tmp_dir, f"clip_{i:04d}.wav")
        subprocess.run(["ffmpeg","-ss",str(seg["start"]),"-to",str(seg["end"]),
                        "-i",video_path,"-ar","16000","-ac","1","-y",clip],
                       capture_output=True, check=True)
        try:
            segs_gen, _ = wm.transcribe(clip, language="zh", repetition_penalty=1.0,
                                         no_repeat_ngram_size=0, condition_on_previous_text=False)
            # Exhaust generator ngay — tránh encoder output còn treo trên VRAM
            segments_list = list(segs_gen)
            text = " ".join(s.text for s in segments_list).strip()
        finally:
            if os.path.exists(clip): os.remove(clip)
            # ── VRAM flush sau mỗi clip ───────────────────────────────────────
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            # ─────────────────────────────────────────────────────────────────
        if _looks_like_noise(text, segments_list):
            skipped += 1
            noise_ranges.append({"start": seg["start"], "end": seg["end"]})
            rprint(f"   [{i+1:03d}] {seg['start']:.2f}–{seg['end']:.2f}s | [dim]⏭️  bỏ qua (nghi tiếng động/hiệu ứng): {text[:60]}[/dim]")
            continue
        data.append({"start": seg["start"], "end": seg["end"], "zh": text})
        rprint(f"   [{i+1:03d}] {seg['start']:.2f}–{seg['end']:.2f}s | {text[:60]}")
    if skipped:
        rprint(f"[yellow]ℹ Đã bỏ qua {skipped} đoạn nghi là tiếng động/hiệu ứng (không có chữ Hán hoặc no_speech_prob cao)[/yellow]")
    os.makedirs(os.path.dirname(ZH_NOISE_JSON), exist_ok=True)
    with open(ZH_NOISE_JSON, "w", encoding="utf-8") as f:
        json.dump(noise_ranges, f, ensure_ascii=False, indent=2)
    return data



_TRANSLATE_PROMPT = """\
Bạn là phiên dịch viên lồng tiếng chuyên nghiệp Trung → Việt.

## Nhiệm vụ
Dịch mỗi câu tiếng Trung sang tiếng Việt sao cho khi đọc to, thời gian đọc
xấp xỉ với "duration" (giây) đã cho.
{context_block}
## Nguyên tắc bắt buộc
1. KHÔNG dịch kiểu từ điển, cụt lủn.
2. Nếu duration ≥ 3s: diễn đạt đầy đủ, dùng cách nói người Việt thật.
3. Nếu duration < 2s: dịch gọn.
4. duration × {syllable_rate} ≈ số âm tiết tiếng Việt cần có.
5. Kết quả nghe như người Việt đang nói, không phải đọc bản dịch.
6. NHẤT QUÁN: giữ cách xưng hô/tên riêng từ ngữ cảnh trước.

## Ví dụ
duration=1.8s | 谢谢大家 → "Xin cảm ơn mọi người"
duration=3.8s | 为你唱歌 → "Chúng tôi cất cao tiếng hát, gửi tặng đến tất cả các bạn"
duration=4.2s | 百年征程 → "Một hành trình dài trăm năm đầy gian nan và vinh quang"

## Output — JSON list, đúng thứ tự input:
[{{"i": 0, "zh_echo": "<copy y nguyên câu \\"zh\\" tương ứng ở input>", "vi": "...", "syllables": <số âm tiết>}}, ...]
QUAN TRỌNG: "zh_echo" phải copy Y NGUYÊN, KHÔNG sửa đổi, câu "zh" gốc tương
ứng đúng với "i" đó trong input - dùng để đối chiếu chống lệch thứ tự.
KHÔNG thêm gì khác ngoài JSON.

## Input
{input_json}
"""

def _normalize_for_compare(text):
    """Bỏ khoảng trắng/dấu câu vụn trước khi so sánh 2 chuỗi tiếng Trung,
    tránh báo lệch giả do model thêm/bớt dấu cách hoặc dấu câu không đáng kể."""
    return re.sub(r'[\s\-—.,;:!?…"\'"''、。！？]', '', text)


def _gemini_batch(items, client, retries=2, context_block=""):
    expected_zh = {item["i"]: item["zh"] for item in items}
    prompt=(_TRANSLATE_PROMPT.replace("{context_block}",context_block)
            .replace("{syllable_rate}",str(SYLLABLE_RATE))
            .replace("{input_json}",json.dumps(items,ensure_ascii=False)))
    for attempt in range(retries+1):
        try:
            resp_text=_gemini_generate(prompt)
            raw=re.sub(r"^```(?:json)?\s*|\s*```$","",resp_text.strip())
            rows=json.loads(raw)
            if not isinstance(rows,list): raise ValueError("Not a list")
            vi_map,syl_map,refusals,mismatched={},{},0,0
            for row in rows:
                if "i" not in row: continue
                idx = row["i"]
                # Đối chiếu zh_echo với câu gốc THẬT ở đúng vị trí idx - chống
                # lệch index khi Gemini trả JSON không khớp thứ tự input.
                expected = expected_zh.get(idx)
                echo = str(row.get("zh_echo", "")).strip()
                if expected is not None and echo and _normalize_for_compare(echo) != _normalize_for_compare(expected):
                    mismatched += 1
                    rprint(f"[red]⚠ [i={idx}] Lệch index: input='{expected[:30]}' nhưng Gemini echo='{echo[:30]}' -> bỏ qua[/red]")
                    continue
                vi=row.get("vi","").strip()
                if _is_refusal(vi): refusals+=1; continue
                vi_map[idx]=vi; syl_map[idx]=row.get("syllables","?")
            if refusals: rprint(f"[yellow]⚠ Gemini từ chối {refusals} câu[/yellow]")
            if mismatched: rprint(f"[red]⚠ {mismatched} câu bị lệch index, đã loại bỏ (sẽ xử lý riêng)[/red]")
            return vi_map,syl_map
        except Exception as e:
            rprint(f"[yellow]⚠ Gemini batch lỗi (lần {attempt+1}): {e}[/yellow]"); time.sleep(1.5)
    return {},{}

def _gemini_single(zh_text, duration_s, client, retries=1):
    items=[{"i":0,"zh":zh_text,"duration":round(duration_s,2)}]
    for _ in range(retries+1):
        try:
            prompt=(_TRANSLATE_PROMPT.replace("{context_block}","")
                    .replace("{syllable_rate}",str(SYLLABLE_RATE))
                    .replace("{input_json}",json.dumps(items,ensure_ascii=False)))
            resp_text=_gemini_generate(prompt)
            raw=re.sub(r"^```(?:json)?\s*|\s*```$","",resp_text.strip())
            rows=json.loads(raw)
            if isinstance(rows,list) and rows:
                vi=rows[0].get("vi","").strip()
                if vi and not _is_refusal(vi): return vi
        except Exception: pass
        time.sleep(1.0)
    return ""

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

def gemini_watermark_filter_only(data, client, batch_size=80):
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
            {"i": bstart + j, "zh": d["zh"], "vi_draft": d["vi"], "duration": round(d["end"] - d["start"], 2)}
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
            rprint(f"   [{i+1:03d}] [-] Loại bỏ (nghi watermark/rác): {d['zh'][:40]}")
            continue
        result.append(d)  # giữ nguyên y hệt bản dịch pipeline gốc, không đổi "vi"

    if removed_count:
        rprint(f"[yellow]ℹ Đã loại {removed_count} câu nghi là watermark/rác[/yellow]")
    rprint(f"[green]✅ Lọc watermark hoàn tất → còn {len(result)} câu[/green]")
    return result


def gemini_review_pass(data, client, batch_size=80, context_tail=4):
    """
    Xem lại TOÀN BỘ bản dịch sau khi đã dịch xong lần đầu, theo đúng timeline.
    - Tự phát hiện + loại bỏ dòng watermark/rác lẫn vào do OCR.
    - Tinh chỉnh bản dịch cho tự nhiên, sáng tạo hơn, dựa vào ngữ cảnh câu
      trước/sau (không chỉ dịch rời rạc từng câu như bước đầu).
    """
    rprint(f"[cyan]🔎 Review pass: xem lại {len(data)} câu theo ngữ cảnh toàn video...[/cyan]")
    all_vi, all_remove = {}, {}
    for bstart in range(0, len(data), batch_size):
        batch = data[bstart:bstart + batch_size]
        items = [
            {"i": bstart + j, "zh": d["zh"], "vi_draft": d["vi"], "duration": round(d["end"] - d["start"], 2)}
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
                    idx = row["i"]
                    if row.get("remove", False):
                        all_remove[idx] = True
                        continue
                    vi = row.get("vi", "").strip()
                    if vi and not _is_refusal(vi):
                        all_vi[idx] = vi
                break
            except Exception as e:
                rprint(f"[yellow]⚠ Review batch lỗi (lần {attempt+1}): {e}[/yellow]")
                time.sleep(1.5)

    result, removed_count, kept_original = [], 0, 0
    for i, d in enumerate(data):
        if all_remove.get(i, False):
            removed_count += 1
            rprint(f"   [{i+1:03d}] [-] Loại bỏ (nghi watermark/rác): {d['zh'][:40]}")
            continue
        vi = all_vi.get(i)
        if not vi:
            vi = d["vi"]  # review lỗi/không trả về -> giữ nguyên bản dịch nháp
            kept_original += 1
        result.append({"start": d["start"], "end": d["end"], "zh": d["zh"], "vi": vi})

    if removed_count:
        rprint(f"[yellow]ℹ Đã loại {removed_count} câu nghi là watermark/rác[/yellow]")
    if kept_original:
        rprint(f"[yellow]ℹ {kept_original} câu giữ nguyên bản dịch nháp (review lỗi)[/yellow]")
    rprint(f"[green]✅ Review pass hoàn tất → còn {len(result)} câu[/green]")
    return result


_FIX_BY_REAL_DUR_PROMPT = """\
Bạn là biên tập viên lồng tiếng Trung → Việt. Các câu dưới đây đã được TTS
đọc thử và ĐO THỜI GIAN THẬT (real_seconds) - không phải ước lượng. Dù đã cắt
gọn ở bước trước, TTS đọc thực tế vẫn dài hơn nhiều so với khung thời gian
gốc cho phép (tol_seconds), nên phải tăng tốc audio quá mức (nghe như tua
nhanh/rap). Hãy viết lại NGẮN GỌN HƠN NỮA, dựa trên real_seconds thật này -
đáng tin hơn con số ước lượng âm tiết trước đó.

## THỨ TỰ ƯU TIÊN (quan trọng nhất trước)
1. **Câu phải ĐÚNG NGỮ PHÁP tiếng Việt, đọc lên nghe trọn vẹn, không cụt lủn.**
   Không cắt mất từ so sánh, từ phủ định, giới từ quan trọng làm sai nghĩa.
2. **Không bịa thêm nội dung/ý không có trong câu gốc.**
3. target_ratio = real_seconds / tol_seconds cho biết cần rút ngắn bao nhiêu
   lần. Ví dụ target_ratio = 1.5 nghĩa là câu hiện tại đang dài gấp 1.5 lần
   khung cho phép -> cần cắt còn khoảng 65-70% độ dài hiện tại (theo số từ),
   không cần chính xác tuyệt đối nhưng phải rút thật sự, không chỉ sửa nhẹ.
4. Giữ khớp mạch với "context_before"/"context_after" (câu trước/sau thật sự
   trong video, chỉ để hiểu mạch truyện - KHÔNG dịch lại các câu ngữ cảnh đó).

## Output — JSON list, đúng thứ tự input:
[{{"i": 0, "vi": "..."}}, ...]
KHÔNG thêm gì khác ngoài JSON.

## Input
{input_json}
"""

def gemini_fix_by_real_dur(tasks_df, client, max_speed=1.4, batch_size=80, max_rounds=2):
    from core._10_gen_audio import TEMP_FILE_TEMPLATE
    from core.tts_backend.tts_main import tts_main
    from core.asr_backend.audio_preprocess import get_audio_duration

    for round_i in range(max_rounds):
        too_fast_idx = []
        for idx, row in tasks_df.iterrows():
            real_dur = float(row.get("real_dur", 0) or 0)
            tol_dur = float(row.get("tol_dur", 0) or 0)
            if tol_dur <= 0 or real_dur <= 0:
                continue
            if real_dur / tol_dur > max_speed:
                too_fast_idx.append(idx)

        if not too_fast_idx:
            rprint(f"[green]✅ Pass 5 (round {round_i+1}): không còn câu nào vượt speed_factor.max={max_speed} sau TTS thật[/green]")
            break

        rprint(f"[cyan]🎤 Pass 5 (round {round_i+1}): {len(too_fast_idx)} câu vẫn cần atempo > {max_speed} dựa trên real_dur thật, viết lại ngắn hơn...[/cyan]")

        all_vi = {}
        for bstart in range(0, len(too_fast_idx), batch_size):
            batch_idx = too_fast_idx[bstart:bstart + batch_size]
            items = []
            for idx in batch_idx:
                row = tasks_df.loc[idx]
                real_dur = float(row["real_dur"])
                tol_dur = float(row["tol_dur"])
                items.append({
                    "i": int(idx),
                    "zh": str(row.get("origin", "")),
                    "vi_hien_tai": str(row["text"]),
                    "real_seconds": round(real_dur, 2),
                    "tol_seconds": round(tol_dur, 2),
                    "target_ratio": round(real_dur / tol_dur, 2),
                    "context_before": str(tasks_df.loc[idx - 1, "text"]) if idx - 1 in tasks_df.index else "",
                    "context_after": str(tasks_df.loc[idx + 1, "text"]) if idx + 1 in tasks_df.index else "",
                })
            prompt = _FIX_BY_REAL_DUR_PROMPT.replace("{input_json}", json.dumps(items, ensure_ascii=False))
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
                        vi = row.get("vi", "").strip()
                        if vi and not _is_refusal(vi):
                            all_vi[int(row["i"])] = vi
                    break
                except Exception as e:
                    rprint(f"[yellow]⚠ Pass 5 batch lỗi (lần {attempt+1}): {e}[/yellow]")
                    time.sleep(1.5)

        if not all_vi:
            rprint("[yellow]⚠ Pass 5: Gemini không trả về câu nào sửa được, dừng lại[/yellow]")
            break

        fixed_count = 0
        for idx in too_fast_idx:
            if idx not in all_vi:
                continue
            new_text = all_vi[idx]
            tasks_df.at[idx, "text"] = new_text
            tasks_df.at[idx, "lines"] = _split_text_for_sub(
                new_text, max_chars=_cfg("zh_pipeline.tts_split_max_chars", 60)
            )

            number = tasks_df.at[idx, "number"]
            lines = tasks_df.at[idx, "lines"]
            new_real_dur = 0.0
            try:
                for line_index, line in enumerate(lines):
                    temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    tts_main(line, temp_file, number, tasks_df)
                    new_real_dur += get_audio_duration(temp_file)
                old_line_count = len(lines)
                stale_idx = old_line_count
                while True:
                    stale_file = TEMP_FILE_TEMPLATE.format(f"{number}_{stale_idx}")
                    if os.path.exists(stale_file):
                        os.remove(stale_file)
                        stale_idx += 1
                    else:
                        break
                old_dur = float(tasks_df.at[idx, "real_dur"])
                tasks_df.at[idx, "real_dur"] = new_real_dur
                rprint(f"   [{idx+1:03d}] real_dur {old_dur:.2f}s → {new_real_dur:.2f}s | {new_text[:50]}")
                fixed_count += 1
            except Exception as e:
                rprint(f"[red]❌ Pass 5: TTS lại câu {idx+1} lỗi: {e}, giữ nguyên bản cũ[/red]")

        rprint(f"[green]✅ Pass 5 (round {round_i+1}) hoàn tất → đã sửa+TTS lại {fixed_count}/{len(too_fast_idx)} câu[/green]")

    return tasks_df


def _count_vi_syllables(text):
    return len(text.split())


_FIX_OVERLONG_PROMPT = """\
Bạn là biên tập viên lồng tiếng Trung → Việt. Các câu dưới đây đang dịch DÀI
HƠN nhiều so với thời lượng cho phép, khiến TTS phải đọc nhanh bất thường
(nghe như tua nhanh/siêu thanh). Hãy viết lại NGẮN GỌN hơn, giữ đúng ý chính,
và PHẢI khớp mạch với "context_before"/"context_after" (câu trước/sau thật sự
trong video, chỉ để bạn hiểu mạch truyện - KHÔNG dịch lại các câu ngữ cảnh đó).

## THỨ TỰ ƯU TIÊN (quan trọng nhất trước)
1. **Câu phải ĐÚNG NGỮ PHÁP tiếng Việt, đọc lên nghe trọn vẹn, không cụt lủn.**
   Ví dụ SAI: cắt mất từ so sánh "hơn", từ phủ định "không", giới từ quan
   trọng... làm câu mất nghĩa hoặc sai ngữ pháp. Test nhanh: nếu bạn đọc to
   câu "vi" lên mà nghe ngang/thiếu/sai nghĩa so với câu gốc → SAI, viết lại.
2. **Không bịa thêm nội dung/ý không có trong câu gốc.**
3. allowed_syllables (= duration × {syllable_rate}, làm tròn) chỉ là MỤC TIÊU
   THAM KHẢO để cắt gọn - KHÔNG BẮT BUỘC phải khớp chính xác. Thà lệch vài âm
   tiết còn hơn một câu sai ngữ pháp hoặc mất nghĩa.

## Cách làm đúng
Ưu tiên cách diễn đạt cô đọng hơn (từ đồng nghĩa ngắn hơn, bỏ từ đệm không
cần thiết) thay vì cắt trụi các từ có chức năng ngữ pháp quan trọng.

## Output — JSON list, đúng thứ tự input:
[{{"i": 0, "vi": "..."}}, ...]
KHÔNG thêm gì khác ngoài JSON.

## Input
{input_json}
"""

def gemini_fix_overlong(data, client, ratio_threshold=1.3, batch_size=80):
    overlong_idx = []
    for i, d in enumerate(data):
        dur = d["end"] - d["start"]
        allowed = max(round(dur * SYLLABLE_RATE), 1)
        actual = _count_vi_syllables(d["vi"])
        if actual > allowed * ratio_threshold:
            overlong_idx.append(i)

    if not overlong_idx:
        rprint("[green]✅ Không có câu nào bị dài quá mức, bỏ qua pass 3[/green]")
        return data

    rprint(f"[cyan]✂️  Pass 3: {len(overlong_idx)}/{len(data)} câu bị dài quá mức, viết ngắn lại...[/cyan]")
    all_vi = {}
    for bstart in range(0, len(overlong_idx), batch_size):
        batch_idx = overlong_idx[bstart:bstart + batch_size]
        items = []
        for idx in batch_idx:
            d = data[idx]
            dur = d["end"] - d["start"]
            allowed = max(round(dur * SYLLABLE_RATE), 1)
            items.append({
                "i": idx,
                "zh": d["zh"],
                "vi_hien_tai": d["vi"],
                "duration": round(dur, 2),
                "allowed_syllables": allowed,
                "context_before": data[idx - 1]["vi"] if idx - 1 >= 0 else "",
                "context_after": data[idx + 1]["vi"] if idx + 1 < len(data) else "",
            })
        prompt = _FIX_OVERLONG_PROMPT.replace("{syllable_rate}", str(SYLLABLE_RATE)).replace("{input_json}", json.dumps(items, ensure_ascii=False))
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
                    vi = row.get("vi", "").strip()
                    if vi and not _is_refusal(vi):
                        all_vi[row["i"]] = vi
                break
            except Exception as e:
                rprint(f"[yellow]⚠ Fix-overlong batch lỗi (lần {attempt+1}): {e}[/yellow]")
                time.sleep(1.5)

    fixed_count = 0
    for idx in overlong_idx:
        if idx in all_vi:
            old_syl = _count_vi_syllables(data[idx]["vi"])
            new_syl = _count_vi_syllables(all_vi[idx])
            rprint(f"   [{idx+1:03d}] {old_syl}→{new_syl} âm tiết | {all_vi[idx][:50]}")
            data[idx]["vi"] = all_vi[idx]
            fixed_count += 1
    rprint(f"[green]✅ Pass 3 hoàn tất → đã sửa {fixed_count}/{len(overlong_idx)} câu quá dài[/green]")
    return data


_FIX_UNDERSHOOT_PROMPT = """\
Bạn là biên tập viên lồng tiếng Trung → Việt. Các câu dưới đây đang dịch NGẮN
HƠN nhiều so với thời lượng cho phép, khiến TTS phải đọc chậm bất thường
(nghe rề rà, kéo dài, ê a). Hãy viết lại DÀI RA tự nhiên hơn, và PHẢI khớp
mạch với "context_before"/"context_after" (câu trước/sau thật sự trong video,
chỉ để bạn hiểu mạch truyện - KHÔNG dịch lại các câu ngữ cảnh đó).

## THỨ TỰ ƯU TIÊN (quan trọng nhất trước)
1. **TUYỆT ĐỐI KHÔNG bịa thêm Ý/HÀNH ĐỘNG/CHI TIẾT không có trong câu gốc.**
   Chỉ được: lặp lại ý đã nói theo cách khác, thêm từ đệm/trợ từ cuối câu
   (à, nhé, đó, mà, thôi...), diễn giải cùng 1 ý bằng nhiều từ hơn.
   VÍ DỤ SAI (đã từng xảy ra, TUYỆT ĐỐI KHÔNG lặp lại):
     zh="知道了" (chỉ là "biết rồi") → SAI: "Tôi biết rồi, biết rồi mà, cằn
     nhằn mãi thôi" (bịa thêm ý "bị cằn nhằn" không có trong câu gốc).
     ĐÚNG hơn: "Biết rồi, biết rồi, tôi biết rồi mà" (chỉ lặp lại ý gốc).
2. **Câu phải ĐÚNG NGỮ PHÁP, nghe tự nhiên như người Việt nói**, không nhồi
   nhét từ vô nghĩa.
3. allowed_syllables (= duration × {syllable_rate}, làm tròn) chỉ là MỤC TIÊU
   THAM KHẢO - KHÔNG BẮT BUỘC khớp chính xác. Thà lệch vài âm tiết còn hơn
   bịa thêm nội dung hoặc nghe gượng ép.

## Output — JSON list, đúng thứ tự input:
[{{"i": 0, "vi": "..."}}, ...]
KHÔNG thêm gì khác ngoài JSON.

## Input
{input_json}
"""

def gemini_fix_undershoot(data, client, ratio_threshold=0.85, batch_size=80):
    short_idx = []
    for i, d in enumerate(data):
        dur = d["end"] - d["start"]
        allowed = max(round(dur * SYLLABLE_RATE), 1)
        actual = _count_vi_syllables(d["vi"])
        if actual < allowed * ratio_threshold:
            short_idx.append(i)

    if not short_idx:
        rprint("[green]✅ Không có câu nào bị ngắn quá mức, bỏ qua pass 4[/green]")
        return data

    rprint(f"[cyan]➕ Pass 4: {len(short_idx)}/{len(data)} câu bị ngắn quá mức, diễn đạt dài ra...[/cyan]")
    all_vi = {}
    for bstart in range(0, len(short_idx), batch_size):
        batch_idx = short_idx[bstart:bstart + batch_size]
        items = []
        for idx in batch_idx:
            d = data[idx]
            dur = d["end"] - d["start"]
            allowed = max(round(dur * SYLLABLE_RATE), 1)
            items.append({
                "i": idx,
                "zh": d["zh"],
                "vi_hien_tai": d["vi"],
                "duration": round(dur, 2),
                "allowed_syllables": allowed,
                "context_before": data[idx - 1]["vi"] if idx - 1 >= 0 else "",
                "context_after": data[idx + 1]["vi"] if idx + 1 < len(data) else "",
            })
        prompt = _FIX_UNDERSHOOT_PROMPT.replace("{syllable_rate}", str(SYLLABLE_RATE)).replace("{input_json}", json.dumps(items, ensure_ascii=False))
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
                    vi = row.get("vi", "").strip()
                    if vi and not _is_refusal(vi):
                        all_vi[row["i"]] = vi
                break
            except Exception as e:
                rprint(f"[yellow]⚠ Fix-undershoot batch lỗi (lần {attempt+1}): {e}[/yellow]")
                time.sleep(1.5)

    fixed_count = 0
    for idx in short_idx:
        if idx in all_vi:
            old_syl = _count_vi_syllables(data[idx]["vi"])
            new_syl = _count_vi_syllables(all_vi[idx])
            rprint(f"   [{idx+1:03d}] {old_syl}→{new_syl} âm tiết | {all_vi[idx][:50]}")
            data[idx]["vi"] = all_vi[idx]
            fixed_count += 1
    rprint(f"[green]✅ Pass 4 hoàn tất → đã sửa {fixed_count}/{len(short_idx)} câu quá ngắn[/green]")
    return data


def _translate_one_batch(data, bstart, chunk_size):
    """Dịch 1 batch (dùng làm task cho ThreadPoolExecutor trong
    pipeline_goc_translate). Trả về (bstart, list kết quả của batch này)."""
    from core.translate_lines import translate_lines

    batch = data[bstart:bstart + chunk_size]
    lines = "\n".join(d["zh"] for d in batch)

    prev_batch = data[bstart - 3:bstart] if bstart > 0 else None
    previous_content_prompt = [d["zh"] for d in prev_batch] if prev_batch else None
    next_batch = data[bstart + chunk_size:bstart + chunk_size + 2]
    after_content_prompt = [d["zh"] for d in next_batch] if next_batch else None

    translation, _ = translate_lines(
        lines, previous_content_prompt, after_content_prompt,
        things_to_note_prompt=None, summary_prompt=None,
        index=bstart // chunk_size,
    )
    vi_lines = translation.split("\n")
    if len(vi_lines) != len(batch):
        # translate_lines() đã tự retry 3 lần nếu số dòng lệch, nhưng nếu
        # vẫn lệch sau retry thì raise lỗi rõ ràng thay vì lặng lẽ ghép sai
        # dòng (dữ liệu OCR/dub sẽ bị lệch câu nếu cứ tiếp tục chạy).
        raise RuntimeError(
            f"pipeline_goc_translate: batch {bstart} dịch ra {len(vi_lines)} dòng, "
            f"cần {len(batch)} dòng - dữ liệu có thể bị lệch."
        )

    batch_result = []
    for d, vi in zip(batch, vi_lines):
        vi = vi.strip()
        if not vi:
            # Câu dịch rỗng (thường do câu gốc là rác OCR như watermark
            # "NTTS"/chữ lẻ vô nghĩa) làm lệch số dòng src.srt/trans.srt khi
            # ghi file - fallback về câu gốc để không bao giờ rỗng.
            vi = d["zh"]
            rprint(f"[yellow]⚠ Câu dịch rỗng, giữ nguyên bản gốc: {d['zh'][:30]}[/yellow]")
        batch_result.append({"start": d["start"], "end": d["end"], "zh": d["zh"], "vi": vi})
    return bstart, batch_result


def pipeline_goc_translate(data):
    """
    Dịch OCR bằng đúng lõi dịch của pipeline gốc VideoLingo (translate_lines -
    faithfulness + reflect/expressiveness check 2 bước), KHÔNG ép số âm tiết
    khớp thời lượng như gemini_translate. Dùng khi config.yaml đặt
    zh_pipeline.translate_engine: "pipeline_goc" (mặc định "gemini_4pass" giữ
    nguyên hành vi cũ).

    Vì không ép âm tiết ở đây, câu quá nhanh/chậm sẽ được zh_gen_audio_tasks
    (đã có sẵn logic if_too_fast/speed_factor) tự xử lý ở bước audio sau,
    giống hệt cách pipeline gốc Whisper vẫn làm - KHÔNG gọi
    gemini_review_pass/gemini_fix_overlong/gemini_fix_undershoot sau hàm này.

    Các batch được dịch SONG SONG qua ThreadPoolExecutor(max_workers), dùng
    đúng config "max_workers" có sẵn - giống hệt cách _4_2_translate.py
    (pipeline gốc Whisper) đang làm, để 2 luồng nhất quán với nhau. Số lần
    gọi API không đổi (vẫn 2 lần/batch: faithfulness + expressiveness) - chỉ
    đổi việc các batch chạy đồng thời hay nối đuôi nhau.
    """
    rprint(f"[cyan]📝 Dịch {len(data)} câu bằng pipeline gốc (translate_lines)...[/cyan]")

    # Đọc từ config để dễ test nhiều mức mà không cần sửa code. Batch càng lớn
    # càng ít lần gọi API, nhưng rủi ro Gemini trả lệch số dòng cũng tăng theo
    # (translate_lines() raise lỗi nếu số dòng output != input, phải dịch lại
    # cả batch đó) - 10 là mặc định an toàn kiểu cũ, thử tăng dần (vd 50, 100)
    # để cân bằng tốc độ/chi phí API với độ ổn định.
    chunk_size = _cfg("zh_pipeline.pipeline_goc_chunk_size", 10)
    max_workers = _cfg("max_workers", 1)
    batch_starts = list(range(0, len(data), chunk_size))

    results_by_start = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_translate_one_batch, data, bstart, chunk_size): bstart
                   for bstart in batch_starts}
        for future in concurrent.futures.as_completed(futures):
            bstart, batch_result = future.result()  # raise lại lỗi nếu batch đó lỗi, dừng cả job
            results_by_start[bstart] = batch_result

    result = []
    for bstart in batch_starts:  # ghép lại ĐÚNG THỨ TỰ gốc, không theo thứ tự hoàn thành
        result.extend(results_by_start[bstart])

    for i, r in enumerate(result):
        dur = r["end"] - r["start"]
        rprint(f"   [{i+1:03d}] dur={dur:.1f}s | {r['vi'][:50]}")

    return result


def gemini_translate(data, client, is_portrait=False):
    rprint(f"[cyan]📝 Gemini dịch {len(data)} câu (batch={TRANSLATE_BATCH})...[/cyan]")
    all_vi,all_syl,prev_tail={},{},[]
    for bstart in range(0,len(data),TRANSLATE_BATCH):
        batch=data[bstart:bstart+TRANSLATE_BATCH]
        items=[{"i":bstart+j,"zh":d["zh"],"duration":round(d["end"]-d["start"],2)}
               for j,d in enumerate(batch)]
        ctx=""
        if prev_tail:
            lines="\n".join(f'- "{zh}" → "{vi}"' for zh,vi in prev_tail)
            ctx=f"\n## Ngữ cảnh từ đoạn trước\n{lines}\n"
        vi_map,syl_map=_gemini_batch(items,client,context_block=ctx)
        all_vi.update(vi_map); all_syl.update(syl_map)
        tail=[]
        for j in range(len(batch)-1,-1,-1):
            idx=bstart+j
            if idx in vi_map: tail.append((batch[j]["zh"],vi_map[idx]))
            if len(tail)>=CONTEXT_TAIL: break
        prev_tail=list(reversed(tail))
    result,missing,recovered=[],0,0
    for i,d in enumerate(data):
        vi=all_vi.get(i,"").strip(); dur=d["end"]-d["start"]
        # if not vi:
        #     missing+=1; vi=_gemini_single(d["zh"],dur,client)
        #     if vi: recovered+=1
        #     else:
        #         try: vi=GoogleTranslator(source="zh-CN",target="vi").translate(d["zh"]) or ""
        #         except Exception as e: rprint(f"   [{i+1:03d}] [!] Mất câu: {e}"); continue
        # if not vi.strip(): continue
        result.append({"start":d["start"],"end":d["end"],"zh":d["zh"],"vi":vi})
        exp=round(dur*SYLLABLE_RATE)
        rprint(f"   [{i+1:03d}] dur={dur:.1f}s exp≥{exp}syl got={all_syl.get(i,'?')}syl | {vi[:50]}")
    if missing:
        rprint(f"[yellow]ℹ {missing}/{len(data)} câu xử lý riêng ({recovered} Gemini, {missing-recovered} GT)[/yellow]")
    return result

def _write_srt(segments, path, text_key):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines=[]
    for i,seg in enumerate(segments,1):
        lines.append(str(i)); lines.append(f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}")
        lines.append(seg[text_key]); lines.append("")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines))

def _write_cleaned_chunks(data):
    rows=[{"text":f'"{d["zh"]}"',"start":d["start"],"end":d["end"]} for d in data]
    df=pd.DataFrame(rows); os.makedirs(os.path.dirname(ZH_CLEANED_CHUNKS),exist_ok=True)
    df.to_excel(ZH_CLEANED_CHUNKS,index=False)
    rprint(f"[green]✅ zh_cleaned_chunks.xlsx → {ZH_CLEANED_CHUNKS} ({len(rows)} rows)[/green]")

def _write_zh_sync(data):
    rows = [{"start": d["start"], "end": d["end"], "zh": d["zh"], "vi": d["vi"]} for d in data]
    os.makedirs(os.path.dirname(ZH_SYNC_JSON), exist_ok=True)
    with open(ZH_SYNC_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    rprint(f"[green]✅ zh_sync.json → {ZH_SYNC_JSON} ({len(rows)} rows)[/green]")

def _write_subtitles(data, is_portrait=False):
    os.makedirs(_OUTPUT_DIR,exist_ok=True); os.makedirs(_AUDIO_DIR,exist_ok=True)
    display_max_chars = MAX_DISPLAY_CHARS if is_portrait else 60
    zh_display,vi_display=[],[]
    for d in data:
        for chunk in _split_sub_for_display(d["zh"],d["start"],d["end"],max_chars=display_max_chars): zh_display.append(chunk)
        for chunk in _split_sub_for_display(d["vi"],d["start"],d["end"],max_chars=display_max_chars): vi_display.append(chunk)
    def _clamp(segs):
        for i in range(len(segs)-1):
            if segs[i]["end"]>segs[i+1]["start"]: segs[i]["end"]=max(segs[i+1]["start"],segs[i]["start"]+0.3)
    _clamp(zh_display); _clamp(vi_display)
    _write_srt(zh_display,SRC_SRT_PATH,"text"); _write_srt(vi_display,TRANS_SRT_PATH,"text")
    _write_srt(data,SRC_AUDIO_SRT,"zh"); _write_srt(data,TRANS_AUDIO_SRT,"vi")
    rprint(f"[green]✅ SRT files written:[/green]")
    rprint(f"   {SRC_SRT_PATH} ({len(zh_display)} entries)")
    rprint(f"   {TRANS_SRT_PATH} ({len(vi_display)} entries)")
    rprint(f"   {SRC_AUDIO_SRT} / {TRANS_AUDIO_SRT} ({len(data)} entries)")

def _merge_short_rows(data, effective_min_dur):
    rows = [{"start": d["start"], "end": d["end"], "zh": d["zh"], "vi": d["vi"]} for d in data]
    i = 0
    while i < len(rows):
        dur = rows[i]["end"] - rows[i]["start"]
        if dur < effective_min_dur:
            if i + 1 < len(rows):
                combined = rows[i + 1]["end"] - rows[i]["start"]
                if combined < effective_min_dur * 2:
                    rows[i]["vi"] += " " + rows[i + 1]["vi"]
                    rows[i]["zh"] += " " + rows[i + 1]["zh"]
                    rows[i]["end"] = rows[i + 1]["end"]
                    rows.pop(i + 1)
                    continue
                else:
                    rows[i]["end"] = rows[i]["start"] + effective_min_dur
        i += 1
    # Câu cuối cùng nếu vẫn quá ngắn (không có câu kế tiếp để gộp xuôi) -> gộp ngược
    if len(rows) >= 2 and (rows[-1]["end"] - rows[-1]["start"]) < effective_min_dur:
        rows[-2]["vi"] += " " + rows[-1]["vi"]
        rows[-2]["zh"] += " " + rows[-1]["zh"]
        rows[-2]["end"] = rows[-1]["end"]
        rows.pop()
    return rows


def zh_asr_and_translate(session_id=None):
    rprint("[bold magenta]🚀 ZH Pipeline: ASR + Translate[/bold magenta]")
    from core._1_ytdlp import find_video_files
    video_path=find_video_files()
    rprint(f"[cyan]Video: {video_path}[/cyan]")
    tmp_dir=tempfile.mkdtemp(prefix="zh_pipe_")
    try:
        os.makedirs(_AUDIO_DIR,exist_ok=True)

        subtitle_source = _cfg("subtitle_source", "whisper")

        # Luôn extract raw audio bất kể nguồn phụ đề nào, vì bước gen audio task
        # phía sau (get_audio_duration) cần file này để tính duration
        if not os.path.exists(_RAW_AUDIO_FILE):
            rprint("[cyan]⏳ Extract raw audio...[/cyan]")
            subprocess.run(["ffmpeg","-i",video_path,"-vn","-acodec","libmp3lame","-q:a","2","-y",_RAW_AUDIO_FILE],
                           capture_output=True, check=True)
            rprint(f"[green]✅ Raw audio → {_RAW_AUDIO_FILE}[/green]")

        if subtitle_source == "ocr":
            # ── Nhánh OCR: đọc sub cứng có sẵn, bỏ qua VAD + Whisper ──────────
            from core.asr_backend.hardsub_ocr import extract_hardsub
            rprint("[cyan]🔎 Đọc phụ đề cứng (hardsub) bằng OCR...[/cyan]")
            ocr_region = _cfg("ocr_region", None)  # ← dòng MỚI thêm
            ocr_result = extract_hardsub(video_path, lang="ch", region=ocr_region)  # ← đổi từ extract_hardsub(video_path, lang="ch")
            raw_segments = ocr_result.get("segments", [])
            data = [
                {"start": seg["start"], "end": seg["end"], "zh": seg["text"].strip()}
                for seg in raw_segments if seg.get("text", "").strip()
            ]
            if not data: raise RuntimeError("OCR không đọc được sub nào từ video")
            rprint(f"[green]✅ OCR: {len(data)} câu sub được nhận diện[/green]")
        else:
            # ── Nhánh Whisper: VAD + faster-whisper như cũ ─────────────────────
            vad_segs=get_vad_segments(video_path,tmp_dir)
            if not vad_segs: raise RuntimeError("VAD không tìm thấy đoạn nói nào")
            _unload_vad()       # giải phóng VRAM trước khi load Whisper
            data=transcribe_vad_clips(video_path,vad_segs,tmp_dir)
            data=[d for d in data if d["zh"].strip()]
            if not data: raise RuntimeError("Whisper không transcribe được")
            _unload_whisper()   # giải phóng VRAM trước Gemini

        client=_gemini_client()
        is_portrait = _is_portrait_video(video_path)
        if is_portrait:
            rprint("[bold cyan]📱 Phát hiện video dọc (portrait/short) -> font sub sẽ tự nhỏ hơn khi burn[/bold cyan]")

        # Cho phép thử nghiệm 2 engine dịch song song mà không cần sửa code -
        # đổi zh_pipeline.translate_engine trong config.yaml giữa các lần chạy.
        translate_engine = _cfg("zh_pipeline.translate_engine", "gemini_4pass")

        if translate_engine == "pipeline_goc":
            data = pipeline_goc_translate(data)
            if not data: raise RuntimeError("pipeline gốc không dịch được câu nào")
            # Mượn Gemini CHỈ để lọc watermark/rác OCR (không đụng vào bản dịch
            # pipeline gốc) - pipeline gốc dịch từng câu độc lập nên không tự
            # phát hiện được watermark lặp lại xuyên suốt video như review pass.
            data = gemini_watermark_filter_only(data, client)
            if not data: raise RuntimeError("Lọc watermark loại bỏ hết câu, kiểm tra lại OCR")
            # KHÔNG gọi gemini_review_pass (sẽ ghi đè mất bản dịch pipeline gốc)
            # /fix_overlong/fix_undershoot ở đây: pipeline_goc_translate không
            # ép âm tiết nên không có khái niệm "quá dài/quá ngắn" cần Gemini
            # sửa - zh_gen_audio_tasks sẽ tự xử lý tốc độ đọc ở bước audio sau,
            # giống pipeline gốc Whisper.
        else:
            data=gemini_translate(data,client)
            if not data: raise RuntimeError("Gemini không dịch được câu nào")

            data = gemini_review_pass(data, client)
            if not data: raise RuntimeError("Review pass loại bỏ hết câu, kiểm tra lại OCR/dịch")

            data = gemini_fix_overlong(data, client)

            data = gemini_fix_undershoot(data, client)

        min_dur_whisper = _cfg("min_subtitle_duration", 2.5)
        ocr_min_dur = _cfg("zh_pipeline.ocr_min_subtitle_duration", 0.8)
        effective_min_dur = ocr_min_dur if subtitle_source == "ocr" else min_dur_whisper
        data = _merge_short_rows(data, effective_min_dur)
        rprint(f"[cyan]🔗 Đã gộp câu ngắn (ngưỡng {effective_min_dur}s) → còn {len(data)} câu[/cyan]")

        _write_cleaned_chunks(data); _write_zh_sync(data); _write_subtitles(data)
        rprint(f"[bold green]✅ ZH ASR+Translate hoàn tất — {len(data)} câu[/bold green]")
    finally:
        shutil.rmtree(tmp_dir,ignore_errors=True)

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

def zh_gen_audio_tasks(session_id=None):
    rprint("[bold magenta]🚀 ZH: Gen audio tasks[/bold magenta]")
    from core.tts_backend.estimate_duration import init_estimator, estimate_duration
    from core.asr_backend.audio_preprocess import get_audio_duration
    with open(ZH_SYNC_JSON, encoding="utf-8") as f:
        sync_rows = json.load(f)
    estimator=init_estimator()
    accept=_cfg("speed_factor.accept",1.2); tol_cfg=_cfg("tolerance",1.5)
    whole_dur=get_audio_duration(_RAW_AUDIO_FILE)
    rows=[]
    for i,r in enumerate(sync_rows):
        rows.append({"number":i+1,"text":r["vi"],"origin":r["zh"],
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
    # Tách riêng ngưỡng cắt dòng theo nguồn phụ đề - KHÔNG dùng chung 1 config
    # để tránh sửa OCR ảnh hưởng ngược lại Whisper (và ngược lại). Nhánh
    # Whisper giữ NGUYÊN key cũ + default cũ, không đổi gì cả.
    subtitle_source = _cfg("subtitle_source", "whisper")
    if subtitle_source == "ocr":
        tts_max_chars = _cfg("zh_pipeline.ocr_tts_split_max_chars", 42)
    else:
        tts_max_chars = _cfg("zh_pipeline.tts_split_max_chars", 60)
    df["lines"]=df["text"].apply(lambda t:_split_text_for_sub(str(t), max_chars=tts_max_chars))
    df["src_lines"]=df["origin"].apply(lambda t:[str(t)])
    df["real_dur"]=0.0; df["new_sub_times"]=None
    os.makedirs(os.path.dirname(_8_1_AUDIO_TASK),exist_ok=True)
    df.to_excel(_8_1_AUDIO_TASK,index=False)
    rprint(f"[bold green]✅ tts_tasks.xlsx → {_8_1_AUDIO_TASK} ({n} rows)[/bold green]")

    sync_data = []
    for _, row in df.iterrows():
        sync_data.append({"start": _parse_dot_time(row["start_time"]),
                          "end": _parse_dot_time(row["end_time"]),
                          "zh": str(row["origin"]), "vi": str(row["text"])})
    _write_srt(sync_data, SRC_SRT_PATH, "zh")
    _write_srt(sync_data, TRANS_SRT_PATH, "vi")
    zh_display, vi_display = [], []
    for d in sync_data:
        for chunk in _split_sub_for_display(d["zh"], d["start"], d["end"]): zh_display.append(chunk)
        for chunk in _split_sub_for_display(d["vi"], d["start"], d["end"]): vi_display.append(chunk)
    _write_srt(zh_display, SRC_AUDIO_SRT, "text")
    _write_srt(vi_display, TRANS_AUDIO_SRT, "text")
    rprint(f"[green]✅ Synced src.srt + trans.srt → {len(sync_data)} entries[/green]")