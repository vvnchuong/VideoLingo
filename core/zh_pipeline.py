"""
zh_pipeline.py — Chinese (ZH) pipeline adapter for VideoLingo
"""

import os, re, gc, json, time, asyncio, tempfile, shutil, sys, subprocess, datetime, warnings
import torch, srt, pandas as pd
from datetime import timedelta
from rich import print as rprint
from google import genai as google_genai
from google.genai import types as google_genai_types
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from core.utils import load_key, update_key
from core.utils.models import (
    _2_CLEANED_CHUNKS, _4_2_TRANSLATION, _5_SPLIT_SUB, _5_REMERGED,
    _8_1_AUDIO_TASK, _OUTPUT_DIR, _AUDIO_DIR, _RAW_AUDIO_FILE,
)

SRC_SRT_PATH    = os.path.join(_OUTPUT_DIR, "src.srt")
TRANS_SRT_PATH  = os.path.join(_OUTPUT_DIR, "trans.srt")
TRANS_AUDIO_SRT = os.path.join(_AUDIO_DIR, "trans_subs_for_audio.srt")
SRC_AUDIO_SRT   = os.path.join(_AUDIO_DIR, "src_subs_for_audio.srt")

def _cfg(key, fallback=None):
    try: return load_key(key)
    except Exception: return fallback

VAD_THRESHOLD      = 0.35
VAD_MIN_SILENCE_MS = 300
VAD_SPEECH_PAD_MS  = 100
VAD_MERGE_GAP_S    = 0.12
MAX_VAD_SEGMENT_S  = 8.0
GAP_WARN_THRESHOLD = 8.0
MAX_DISPLAY_CHARS  = 42
TRANSLATE_BATCH    = 20
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

# ═══════════════════════════════════════════════════════════════════════════════
# PATH HELPERS — session_id=None → dùng path gốc, session_id="abc" → isolate
# ═══════════════════════════════════════════════════════════════════════════════
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
        "cleaned_chunks":  _2_CLEANED_CHUNKS,
        "translation":     _4_2_TRANSLATION,
        "split_sub":       _5_SPLIT_SUB,
        "remerged":        _5_REMERGED,
        "audio_task":      _8_1_AUDIO_TASK,
        "src_srt":         SRC_SRT_PATH,
        "trans_srt":       TRANS_SRT_PATH,
        "trans_audio_srt": TRANS_AUDIO_SRT,
        "src_audio_srt":   SRC_AUDIO_SRT,
    }
# ═══════════════════════════════════════════════════════════════════════════════

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
    text=text.strip()
    if not text: return []
    raw_parts=re.split(r'(?<=[\.\!\?,;:…])\s+',text)
    raw_parts=[p.strip() for p in raw_parts if p.strip()] or [text]
    chunks,cur=[],""
    for p in raw_parts:
        if cur and len(cur)+1+len(p)>max_chars: chunks.append(cur); cur=p
        elif len(p)>max_chars:
            if cur: chunks.append(cur); cur=""
            words,line=p.split(),""
            for w in words:
                if line and len(line)+1+len(w)>max_chars: chunks.append(line); line=w
                else: line=(line+" "+w).strip() if line else w
            if line: cur=line
        else: cur=(cur+" "+p).strip() if cur else p
    if cur: chunks.append(cur)
    if len(chunks)<=1: return [{"start":start_s,"end":end_s,"text":text}]
    total_chars=sum(len(c) for c in chunks); total_dur=max(end_s-start_s,0.01)
    result,t=[],start_s
    for i,c in enumerate(chunks):
        share=len(c)/total_chars
        seg_end=end_s if i==len(chunks)-1 else min(t+total_dur*share,end_s)
        result.append({"start":t,"end":max(seg_end,t+0.3),"text":c}); t=seg_end
    return result

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

def transcribe_vad_clips(video_path, vad_segs, tmp_dir):
    wm = _load_faster_whisper()
    data = []
    rprint(f"[cyan]🎤 Whisper transcribe {len(vad_segs)} segments (reload mỗi {WHISPER_RELOAD_EVERY} clips)...[/cyan]")
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
        data.append({"start": seg["start"], "end": seg["end"], "zh": text})
        rprint(f"   [{i+1:03d}] {seg['start']:.2f}–{seg['end']:.2f}s | {text[:60]}")
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
4. duration × 3.5 ≈ số âm tiết tiếng Việt cần có.
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

def gemini_translate(data, client):
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
        exp=round(dur*3.5)
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
    df=pd.DataFrame(rows); os.makedirs(os.path.dirname(_2_CLEANED_CHUNKS),exist_ok=True)
    df.to_excel(_2_CLEANED_CHUNKS,index=False)
    rprint(f"[green]✅ cleaned_chunks.xlsx → {_2_CLEANED_CHUNKS} ({len(rows)} rows)[/green]")

def _write_translation_results(data):
    rows=[{"Source":d["zh"],"Translation":d["vi"],"start_time":_srt_time_dot(d["start"]),
           "end_time":_srt_time_dot(d["end"]),"duration":round(d["end"]-d["start"],3)} for d in data]
    df=pd.DataFrame(rows); os.makedirs(os.path.dirname(_4_2_TRANSLATION),exist_ok=True)
    df.to_excel(_4_2_TRANSLATION,index=False); df.to_excel(_5_SPLIT_SUB,index=False); df.to_excel(_5_REMERGED,index=False)
    rprint(f"[green]✅ translation_results.xlsx → {_4_2_TRANSLATION} ({len(rows)} rows)[/green]")

def _write_subtitles(data):
    os.makedirs(_OUTPUT_DIR,exist_ok=True); os.makedirs(_AUDIO_DIR,exist_ok=True)
    zh_display,vi_display=[],[]
    for d in data:
        for chunk in _split_sub_for_display(d["zh"],d["start"],d["end"]): zh_display.append(chunk)
        for chunk in _split_sub_for_display(d["vi"],d["start"],d["end"]): vi_display.append(chunk)
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

def zh_asr_and_translate(session_id=None):
    rprint("[bold magenta]🚀 ZH Pipeline: ASR + Translate[/bold magenta]")
    from core._1_ytdlp import find_video_files
    video_path=find_video_files()
    rprint(f"[cyan]Video: {video_path}[/cyan]")
    tmp_dir=tempfile.mkdtemp(prefix="zh_pipe_")
    try:
        os.makedirs(_AUDIO_DIR,exist_ok=True)
        if not os.path.exists(_RAW_AUDIO_FILE):
            rprint("[cyan]⏳ Extract raw audio...[/cyan]")
            subprocess.run(["ffmpeg","-i",video_path,"-vn","-acodec","libmp3lame","-q:a","2","-y",_RAW_AUDIO_FILE],
                           capture_output=True, check=True)
            rprint(f"[green]✅ Raw audio → {_RAW_AUDIO_FILE}[/green]")
        vad_segs=get_vad_segments(video_path,tmp_dir)
        if not vad_segs: raise RuntimeError("VAD không tìm thấy đoạn nói nào")
        _unload_vad()       # giải phóng VRAM trước khi load Whisper
        data=transcribe_vad_clips(video_path,vad_segs,tmp_dir)
        data=[d for d in data if d["zh"].strip()]
        if not data: raise RuntimeError("Whisper không transcribe được")
        _unload_whisper()   # giải phóng VRAM trước Gemini
        client=_gemini_client()
        data=gemini_translate(data,client)
        if not data: raise RuntimeError("Gemini không dịch được câu nào")
        _write_cleaned_chunks(data); _write_translation_results(data); _write_subtitles(data)
        try: update_key("whisper.detected_language","zh")
        except Exception: pass
        rprint(f"[bold green]✅ ZH ASR+Translate hoàn tất — {len(data)} câu[/bold green]")
    finally:
        shutil.rmtree(tmp_dir,ignore_errors=True)

def _split_text_for_sub(text, max_chars=42):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip() if cur else w
    if cur: lines.append(cur)
    return lines if lines else [text]

def zh_gen_audio_tasks(session_id=None):
    rprint("[bold magenta]🚀 ZH: Gen audio tasks[/bold magenta]")
    from core.tts_backend.estimate_duration import init_estimator, estimate_duration
    from core.asr_backend.audio_preprocess import get_audio_duration
    df_trans=pd.read_excel(_4_2_TRANSLATION)
    estimator=init_estimator()
    accept=_cfg("speed_factor.accept",1.2); tol_cfg=_cfg("tolerance",1.5)
    min_dur=_cfg("min_subtitle_duration",2.5); whole_dur=get_audio_duration(_RAW_AUDIO_FILE)
    rows=[]
    for i,row in df_trans.iterrows():
        rows.append({"number":i+1,"text":row["Translation"],"origin":row["Source"],
                     "start_time":row["start_time"],"end_time":row["end_time"],"duration":row["duration"]})
    i=0
    while i<len(rows):
        dur=rows[i]["duration"]
        if dur<min_dur:
            if i+1<len(rows):
                combined=_parse_dot_time(rows[i+1]["end_time"])-_parse_dot_time(rows[i]["start_time"])
                if combined<min_dur*2:
                    rprint(f"[yellow]Merge row {i+1}+{i+2}[/yellow]")
                    rows[i]["text"]+=" "+rows[i+1]["text"]; rows[i]["origin"]+=" "+rows[i+1]["origin"]
                    rows[i]["end_time"]=rows[i+1]["end_time"]; rows[i]["duration"]=combined
                    rows.pop(i+1); continue
                else:
                    rows[i]["end_time"]=_srt_time_dot(_parse_dot_time(rows[i]["start_time"])+min_dur)
                    rows[i]["duration"]=min_dur
        i+=1
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
    df["cut_off"]=0
    df.loc[df["gap"]>=tol_cfg,"cut_off"]=1; df.loc[n-1,"cut_off"]=1
    df["lines"]=df["text"].apply(lambda t:_split_text_for_sub(str(t)))
    df["src_lines"]=df["origin"].apply(lambda t:[str(t)])
    df["real_dur"]=0.0; df["new_sub_times"]=None
    os.makedirs(os.path.dirname(_8_1_AUDIO_TASK),exist_ok=True)
    df.to_excel(_8_1_AUDIO_TASK,index=False)
    rprint(f"[bold green]✅ tts_tasks.xlsx → {_8_1_AUDIO_TASK} ({n} rows)[/bold green]")

    # Sync lại src.srt + trans.srt theo rows đã merge — _8_2 match index 1-1
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