import os
import srt
import shutil
import subprocess
import pandas as pd

import streamlit as st
from core.utils import *

TRANS_SRT  = "output/trans.srt"
SRC_SRT    = "output/src.srt"
AUDIO_SRT  = "output/audio/trans_subs_for_audio.srt"
SUB_VIDEO  = "output/output_sub.mp4"


def _load_srt(path: str):
    with open(path, encoding="utf-8") as f:
        return list(srt.parse(f.read()))

def _save_srt(path: str, subs: list):
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

def _sync_audio_srt(new_trans_subs: list):
    if not os.path.exists(AUDIO_SRT):
        return
    audio_subs = _load_srt(AUDIO_SRT)
    for a_sub in audio_subs:
        matched = []
        for t_sub in new_trans_subs:
            if t_sub.start < a_sub.end and t_sub.end > a_sub.start:
                matched.append(t_sub.content.strip())
        if matched:
            a_sub.content = " ".join(matched)
    _save_srt(AUDIO_SRT, audio_subs)

def _clear_dub_cache():
    xlsx = "output/audio/tts_tasks.xlsx"
    segs = "output/audio/segs"
    if os.path.exists(xlsx):
        os.remove(xlsx)
    if os.path.exists(segs):
        shutil.rmtree(segs)

def _rerender_sub_video():
    try:
        video_files = [f for f in os.listdir("output")
                       if f.endswith(".mp4") and "output" not in f and "preview" not in f]
        if not video_files:
            st.error("Không tìm thấy video gốc trong output/")
            return False
        src_video = os.path.join("output", video_files[0])
        srt_abs   = os.path.abspath(TRANS_SRT).replace("\\", "/").replace(":", "\\:")
        sub_style = "FontSize=18,PrimaryColour=&H00FFFFFF,Outline=3,OutlineColour=&H000000,Alignment=2,MarginV=5"
        subprocess.run([
            "ffmpeg", "-y", "-i", src_video,
            "-vf", f"subtitles='{srt_abs}':force_style='{sub_style}'",
            "-c:a", "copy", SUB_VIDEO
        ], check=True, capture_output=True)
        return True
    except Exception as e:
        st.error(f"Render lỗi: {e}")
        return False

def _subs_to_df(trans_subs, src_map):
    rows = []
    for sub in trans_subs:
        ts = f"{str(sub.start)[:-3]} → {str(sub.end)[:-3]}"
        rows.append({
            "#":        sub.index,
            "Thời gian": ts,
            "Gốc":      src_map.get(sub.index, ""),
            "Dịch":     sub.content,
        })
    return pd.DataFrame(rows)

def _df_to_subs(df, original_subs):
    result = []
    for i, sub in enumerate(original_subs):
        new_content = df.iloc[i]["Dịch"] if i < len(df) else sub.content
        result.append(srt.Subtitle(
            index=sub.index,
            start=sub.start,
            end=sub.end,
            content=str(new_content).strip(),
        ))
    return result


def edit_sub_section():
    st.header("b.5 Review & Edit Subtitles")

    if not os.path.exists(TRANS_SRT):
        return False

    with st.container(border=True):
        # Load subs với mtime check
        mtime = os.path.getmtime(TRANS_SRT)
        if "edit_subs" not in st.session_state or st.session_state.get("edit_subs_mtime") != mtime:
            st.session_state.edit_subs = _load_srt(TRANS_SRT)
            st.session_state.edit_subs_mtime = mtime

        trans_subs = st.session_state.edit_subs

        src_map = {}
        if os.path.exists(SRC_SRT):
            for s in _load_srt(SRC_SRT):
                src_map[s.index] = s.content

        df = _subs_to_df(trans_subs, src_map)

        col_edit, col_video = st.columns([1, 1])

        with col_edit:
            st.caption("✏️ Sửa cột **Dịch** — cột khác readonly")
            edited_df = st.data_editor(
                df,
                height=580,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "#":         st.column_config.NumberColumn("#", width="small", disabled=True),
                    "Thời gian": st.column_config.TextColumn("Thời gian", width="medium", disabled=True),
                    "Gốc":       st.column_config.TextColumn("Gốc", width="large", disabled=True),
                    "Dịch":      st.column_config.TextColumn("Dịch ✏️", width="large"),
                },
                key="sub_editor",
            )

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("💾 Save", use_container_width=True):
                    new_subs = _df_to_subs(edited_df, trans_subs)
                    _save_srt(TRANS_SRT, new_subs)
                    _sync_audio_srt(new_subs)
                    _clear_dub_cache()
                    st.session_state.edit_subs = new_subs
                    st.success("✅ Saved!")

            with b2:
                if st.button("🎬 Re-render", use_container_width=True, type="primary"):
                    new_subs = _df_to_subs(edited_df, trans_subs)
                    _save_srt(TRANS_SRT, new_subs)
                    _sync_audio_srt(new_subs)
                    _clear_dub_cache()
                    st.session_state.edit_subs = new_subs
                    with st.spinner("Rendering..."):
                        ok = _rerender_sub_video()
                    if ok:
                        st.success("✅ Done!")
                        st.rerun()

            with b3:
                if st.button("🔄 Reload", use_container_width=True):
                    del st.session_state["edit_subs"]
                    st.rerun()

        with col_video:
            if os.path.exists(SUB_VIDEO):
                st.video(SUB_VIDEO)
            else:
                try:
                    vf = [f for f in os.listdir("output")
                          if f.endswith(".mp4") and "output" not in f]
                    if vf:
                        st.video(os.path.join("output", vf[0]))
                except Exception:
                    pass

    return True