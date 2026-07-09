import os, re, gc, json, time, asyncio, tempfile, shutil, sys, subprocess, datetime, warnings
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
MAX_DISPLAY_CHARS  = 200  # đồng bộ với zh_pipeline.tts_split_max_chars để sub/dub hiển thị nhất quán, 1 câu luôn trọn vẹn
# Số âm tiết tiếng Việt/giây làm chuẩn khi dịch. 3.5 là tốc độ nói tự nhiên
# thuần Việt, nhưng tiếng Trung gốc thường đọc nhanh hơn tiếng Việt nên ép về
# đúng 3.5 hay làm câu dịch bị cắt cộc lốc. Nâng lên 4.0 (~1.15x) để AI có đất
# diễn đạt đầy đủ hơn, tự nhiên hơn, đỡ phải cắt bớt ý.
SYLLABLE_RATE = _cfg("zh_pipeline.syllable_rate", 4.0)
TRANSLATE_BATCH    = 300  # Gemini 3.1 Flash-Lite output tối đa 64K token, đủ chứa ~300 câu/request
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
    """
    Nếu text dài hơn max_chars, XUỐNG DÒNG (hiển thị cùng lúc trong 1 khung thời gian)
    thay vì tách thành nhiều đoạn hiển thị nối tiếp nhau theo thời gian - tránh hiện
    tượng "mất chữ rồi hiện lại" khi sub đang là 1 câu hoàn chỉnh bị chia cắt.
    """
    text = text.strip()
    if not text:
        return []
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
[{{"i": 0, "vi": "...", "syllables": <số âm tiết>}}, ...]
KHÔNG thêm gì khác ngoài JSON.

## Input
{input_json}
"""

def _gemini_batch(items, client, retries=2, context_block=""):
    prompt=(_TRANSLATE_PROMPT.replace("{context_block}",context_block)
            .replace("{syllable_rate}",str(SYLLABLE_RATE))
            .replace("{input_json}",json.dumps(items,ensure_ascii=False)))
    for attempt in range(retries+1):
        try:
            resp=client.models.generate_content(model=GEMINI_MODEL,contents=prompt,
                config=google_genai_types.GenerateContentConfig(response_mime_type="application/json"))
            raw=re.sub(r"^```(?:json)?\s*|\s*```$","",resp.text.strip())
            rows=json.loads(raw)
            if not isinstance(rows,list): raise ValueError("Not a list")
            vi_map,syl_map,refusals={},{},0
            for row in rows:
                if "i" not in row: continue
                vi=row.get("vi","").strip()
                if _is_refusal(vi): refusals+=1; continue
                vi_map[row["i"]]=vi; syl_map[row["i"]]=row.get("syllables","?")
            if refusals: rprint(f"[yellow]⚠ Gemini từ chối {refusals} câu[/yellow]")
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
            resp=client.models.generate_content(model=GEMINI_MODEL,contents=prompt,
                config=google_genai_types.GenerateContentConfig(response_mime_type="application/json"))
            raw=re.sub(r"^```(?:json)?\s*|\s*```$","",resp.text.strip())
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

def gemini_review_pass(data, client, batch_size=300, context_tail=4):
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
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt,
                    config=google_genai_types.GenerateContentConfig(response_mime_type="application/json"))
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip())
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


def _count_vi_syllables(text):
    """Đếm âm tiết tiếng Việt = số từ cách nhau bởi khoảng trắng (mỗi âm tiết
    tiếng Việt viết rời, cách nhau 1 space theo chính tả chuẩn)."""
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

def gemini_fix_overlong(data, client, ratio_threshold=1.3, batch_size=300):
    """
    Pass 3: chỉ nhắm đúng các câu dịch DÀI HƠN ratio_threshold lần so với số
    âm tiết cho phép (duration × SYLLABLE_RATE) - yêu cầu Gemini viết ngắn gọn lại,
    kèm ngữ cảnh câu trước/sau LẤY ĐỘNG từ chính data của video đang xử lý
    (không phải nội dung cố định), để giữ đúng mạch truyện khi rút gọn.
    """
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
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt,
                    config=google_genai_types.GenerateContentConfig(response_mime_type="application/json"))
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip())
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

def gemini_fix_undershoot(data, client, ratio_threshold=0.85, batch_size=300):
    """
    Đối xứng với gemini_fix_overlong: chỉ nhắm đúng các câu dịch NGẮN HƠN
    ratio_threshold lần so với số âm tiết cho phép (TTS phải đọc chậm/kéo dài
    bất thường) - yêu cầu Gemini diễn đạt dài ra tự nhiên hơn, kèm ngữ cảnh
    câu trước/sau lấy động từ chính data của video đang xử lý.
    """
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
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt,
                    config=google_genai_types.GenerateContentConfig(response_mime_type="application/json"))
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip())
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
        if not vi:
            missing+=1; vi=_gemini_single(d["zh"],dur,client)
            if vi: recovered+=1
            else:
                try: vi=GoogleTranslator(source="zh-CN",target="vi").translate(d["zh"]) or ""
                except Exception as e: rprint(f"   [{i+1:03d}] [!] Mất câu: {e}"); continue
        if not vi.strip(): continue
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
    # Video dọc (portrait/short) màn hình hẹp -> giữ ngưỡng thấp (42).
    # Video ngang (landscape) màn hình rộng hơn -> tăng ngưỡng lên 60 để đỡ bị
    # xuống dòng/cắt không cần thiết với câu dịch tiếng Việt dài hơn gốc.
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
    """
    Gộp các câu ngắn hơn effective_min_dur để TTS có đủ thời gian đọc tự nhiên.
    QUAN TRỌNG: hàm này chỉ được gọi 1 LẦN DUY NHẤT, ngay sau khi dịch xong và
    TRƯỚC KHI ghi ra file cho user edit - để đảm bảo video sub, file edit, và
    video dub đều dùng chung đúng 1 bộ dữ liệu đã merge, không lệch pha nhau.
    """
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
            ocr_result = extract_hardsub(video_path, lang="ch")
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
        data=gemini_translate(data,client)
        if not data: raise RuntimeError("Gemini không dịch được câu nào")

        # Review pass: xem lại toàn bộ theo timeline, tự lọc watermark/rác +
        # tinh chỉnh dịch cho tự nhiên/sáng tạo hơn (chạy sau dịch thô lần đầu)
        data = gemini_review_pass(data, client)
        if not data: raise RuntimeError("Review pass loại bỏ hết câu, kiểm tra lại OCR/dịch")

        # Pass 3: chỉ sửa riêng các câu dịch quá dài (TTS đọc như tua nhanh)
        data = gemini_fix_overlong(data, client)

        # Pass 4: chỉ sửa riêng các câu dịch quá ngắn (TTS đọc chậm như rùa bò)
        data = gemini_fix_undershoot(data, client)

        # Merge câu ngắn NGAY TẠI ĐÂY (1 lần duy nhất), TRƯỚC KHI ghi bất kỳ file
        # nào (kể cả file cho "video sub") - để video sub và video dub sau này
        # luôn dùng chung đúng 1 bộ dữ liệu, không lệch pha nhau nữa.
        min_dur_whisper = _cfg("min_subtitle_duration", 2.5)
        ocr_min_dur = _cfg("zh_pipeline.ocr_min_subtitle_duration", 1.2)
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
    # KHÔNG merge ở đây nữa - dữ liệu trong ZH_SYNC_JSON đã được merge 1 lần duy nhất
    # ở bước zh_asr_and_translate (_merge_short_rows), để "video sub" và "video dub"
    # luôn dùng chung đúng 1 bộ dữ liệu, không lệch pha nhau.
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