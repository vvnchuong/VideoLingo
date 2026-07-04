import os
import re
import json
import srt
import datetime
import pandas as pd

import streamlit as st
from core.utils import *
from core.utils.models import _8_1_AUDIO_TASK
from core.utils.delete_retry_dubbing import delete_dubbing_files

SRC_SRT   = "output/src.srt"
TRANS_SRT = "output/trans.srt"
SUB_VIDEO = "output/output_sub.mp4"
ZH_SYNC_JSON = "output/log/zh_sync.json"

MAX_DISPLAY_CHARS = 42

def _split_sub_for_display(text, start_s, end_s, max_chars=MAX_DISPLAY_CHARS):
    text = text.strip()
    if not text: return []
    raw_parts = re.split(r'(?<=[\.\!\?,;:…])\s+', text)
    raw_parts = [p.strip() for p in raw_parts if p.strip()] or [text]
    chunks, cur = [], ""
    for p in raw_parts:
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur); cur = p
        elif len(p) > max_chars:
            if cur: chunks.append(cur); cur = ""
            words, line = p.split(), ""
            for w in words:
                if line and len(line) + 1 + len(w) > max_chars:
                    chunks.append(line); line = w
                else:
                    line = (line + " " + w).strip() if line else w
            if line: cur = line
        else:
            cur = (cur + " " + p).strip() if cur else p
    if cur: chunks.append(cur)
    if len(chunks) <= 1:
        return [{"start": start_s, "end": end_s, "text": text}]
    total_chars = sum(len(c) for c in chunks); total_dur = max(end_s - start_s, 0.01)
    result, t = [], start_s
    for i, c in enumerate(chunks):
        share = len(c) / total_chars
        seg_end = end_s if i == len(chunks) - 1 else min(t + total_dur * share, end_s)
        result.append({"start": t, "end": max(seg_end, t + 0.3), "text": c}); t = seg_end
    return result

TS_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})$")


def _str_to_sec(s: str) -> float:
    m = TS_RE.match(str(s).strip())
    if not m:
        raise ValueError(f"Sai định dạng thời gian: '{s}' (cần HH:MM:SS.mmm)")
    h, mi, sec, ms = m.groups()
    ms = int(ms.ljust(3, "0")[:3])
    return int(h) * 3600 + int(mi) * 60 + int(sec) + ms / 1000

def _sec_to_str(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{int(s):02d}.{int(round((s % 1) * 1000)):03d}"

def _sec_to_td(sec: float) -> datetime.timedelta:
    return datetime.timedelta(seconds=sec)


def _load_zh_sync():
    """zh_sync.json: bảng 1 câu = 1 dòng, khớp 1:1 zh/vi — nguồn thật để dub.
    File này chỉ dành riêng cho ZH pipeline, KHÔNG liên quan gì translation_results.xlsx (EN/other)."""
    if not os.path.exists(ZH_SYNC_JSON):
        return None
    with open(ZH_SYNC_JSON, encoding="utf-8") as f:
        return json.load(f)

def _df_for_editor(sync_rows) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(sync_rows, 1):
        rows.append({
            "#":        i,
            "Bắt đầu":  _sec_to_str(r["start"]),
            "Kết thúc": _sec_to_str(r["end"]),
            "Gốc":      r["zh"],
            "Dịch":     r["vi"],
        })
    return pd.DataFrame(rows)

def _editor_df_to_rows(edited_df: pd.DataFrame):
    """Rebuild + sort theo thời gian, bỏ dòng Dịch rỗng."""
    parsed = []
    for _, row in edited_df.iterrows():
        vi = str(row.get("Dịch", "")).strip()
        if not vi or vi.lower() == "nan":
            continue
        try:
            start = _str_to_sec(row["Bắt đầu"])
            end = _str_to_sec(row["Kết thúc"])
        except ValueError as e:
            st.warning(f"Bỏ qua dòng lỗi: {e}")
            continue
        if end <= start:
            st.warning(f"Bỏ qua dòng có Kết thúc <= Bắt đầu: '{vi[:30]}...'")
            continue
        src = str(row.get("Gốc", "")).strip()
        parsed.append((start, end, src, vi))

    parsed.sort(key=lambda r: r[0])
    return [{"start": start, "end": end, "zh": src, "vi": vi} for start, end, src, vi in parsed]

def _write_srt_display(path, rows, key):
    """Ghi srt để BURN/XEM — tách câu dài thành nhiều dòng theo thời lượng (không phải bản 1:1
    dùng cho TTS). zh_sync.json vẫn giữ câu đầy đủ, chỉ file .srt này bị tách nhỏ để hiển thị."""
    entries = []
    for r in rows:
        entries.extend(_split_sub_for_display(r[key], r["start"], r["end"]))
    subs = [
        srt.Subtitle(index=i, start=_sec_to_td(e["start"]), end=_sec_to_td(e["end"]), content=e["text"])
        for i, e in enumerate(entries, 1)
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

def _apply_edits(rows):
    """Ghi lại zh_sync.json (nguồn thật, câu đầy đủ) + sinh lại src.srt/trans.srt bản đã tách dòng
    hiển thị để xem trước/burn, rồi xóa cache dub cũ. Không đụng gì tới excel."""
    os.makedirs(os.path.dirname(ZH_SYNC_JSON), exist_ok=True)
    with open(ZH_SYNC_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    _write_srt_display(SRC_SRT, rows, "zh")
    _write_srt_display(TRANS_SRT, rows, "vi")

    delete_dubbing_files()
    if os.path.exists(_8_1_AUDIO_TASK):
        os.remove(_8_1_AUDIO_TASK)
    if os.path.exists(SUB_VIDEO):
        os.remove(SUB_VIDEO)


def edit_sub_section():
    sync_rows = _load_zh_sync()
    if sync_rows is None:
        return False

    st.header("b.5 Review & Edit Subtitles")

    with st.container(border=True):
        st.caption(
            "✏️ Sửa **Bắt đầu / Kết thúc / Gốc / Dịch** — bảng gốc 1 câu = 1 dòng, khớp 1:1 zh/vi "
            "(khác với src.srt/trans.srt lúc burn hình, bị tách dòng hiển thị riêng theo từng ngôn ngữ "
            "nên số dòng zh/vi không khớp nhau — đừng lấy số dòng ở đó ra so sánh). "
            "Kéo xuống cuối bảng để thêm dòng mới, bấm 🗑️ đầu dòng để xóa. Định dạng thời gian: HH:MM:SS.mmm.\n\n"
            f"Lưu ý: bước dub vẫn tự gộp các dòng < {load_key('min_subtitle_duration')}s lại với dòng kế tiếp "
            "để TTS có đủ thời gian đọc — đây là hành vi mặc định của pipeline, không phải do sửa ở đây."
        )

        editor_df = _df_for_editor(sync_rows)
        edited_df = st.data_editor(
            editor_df,
            height=580,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "#":         st.column_config.NumberColumn("#", width="small", disabled=True),
                "Bắt đầu":   st.column_config.TextColumn("Bắt đầu", width="small"),
                "Kết thúc":  st.column_config.TextColumn("Kết thúc", width="small"),
                "Gốc":       st.column_config.TextColumn("Gốc", width="large"),
                "Dịch":      st.column_config.TextColumn("Dịch ✏️", width="large"),
            },
            key="sub_editor",
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save", use_container_width=True):
                rows = _editor_df_to_rows(edited_df)
                if rows:
                    _apply_edits(rows)
                    st.success(f"✅ Saved {len(rows)} câu. Bấm 'Start Audio Processing' ở bước dưới để dub lại.")
                    st.rerun()

        with b2:
            if st.button("🎬 Re-render preview", use_container_width=True, type="primary"):
                rows = _editor_df_to_rows(edited_df)
                if rows:
                    _apply_edits(rows)
                    with st.spinner("Rendering..."):
                        from core._7_sub_into_vid import merge_subtitles_to_video
                        merge_subtitles_to_video()
                    st.success("✅ Done!")
                    st.rerun()

        if os.path.exists(SUB_VIDEO):
            st.video(SUB_VIDEO)

    return True