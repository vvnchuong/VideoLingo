import os
import datetime
import pandas as pd

import streamlit as st
from core.utils import *
from core.utils.models import _4_2_TRANSLATION, _5_SPLIT_SUB, _5_REMERGED, _8_1_AUDIO_TASK
from core.utils.delete_retry_dubbing import delete_dubbing_files

SRC_SRT        = "output/src.srt"
TRANS_SRT      = "output/trans.srt"
SUB_VIDEO      = "output/output_sub.mp4"
TRANS_AUDIO_SRT = "output/audio/trans_subs_for_audio.srt"
SRC_AUDIO_SRT   = "output/audio/src_subs_for_audio.srt"


def _parse_srt_time(s: str) -> float:
    """'HH:MM:SS,mmm' -> giây (float)"""
    s = s.strip()
    h, m, rest = s.split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000

def _format_srt_time(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"

def _split_timestamp(ts: str):
    """'HH:MM:SS,mmm --> HH:MM:SS,mmm' -> (start_str, end_str)"""
    start, end = ts.split(" --> ")
    return start.strip(), end.strip()


def _load_translation_df():
    if not os.path.exists(_4_2_TRANSLATION):
        return None
    return pd.read_excel(_4_2_TRANSLATION)

def _df_for_editor(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, row in raw_df.iterrows():
        start_s, end_s = _split_timestamp(str(row["timestamp"]))
        rows.append({
            "#":        i + 1,
            "Bắt đầu":  start_s,
            "Kết thúc": end_s,
            "Gốc":      str(row["Source"]),
            "Dịch":     str(row["Translation"]),
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
            start = _parse_srt_time(row["Bắt đầu"])
            end = _parse_srt_time(row["Kết thúc"])
        except Exception as e:
            st.warning(f"Bỏ qua dòng lỗi định dạng thời gian (cần HH:MM:SS,mmm): {e}")
            continue
        if end <= start:
            st.warning(f"Bỏ qua dòng có Kết thúc <= Bắt đầu: '{vi[:30]}...'")
            continue
        src = str(row.get("Gốc", "")).strip()
        parsed.append((start, end, src, vi))

    parsed.sort(key=lambda r: r[0])
    return [
        {"Source": src, "Translation": vi,
         "timestamp": f"{_format_srt_time(start)} --> {_format_srt_time(end)}",
         "duration": round(end - start, 3)}
        for start, end, src, vi in parsed
    ]

def _clear_downstream_cache():
    """Xóa các file sinh ra từ translation_results.xlsx để split_for_sub_main() +
    align_timestamp_main() tái tạo lại đúng theo data mới, không dùng cache cũ."""
    for f in [_5_SPLIT_SUB, _5_REMERGED, _8_1_AUDIO_TASK,
              SRC_SRT, TRANS_SRT, TRANS_AUDIO_SRT, SRC_AUDIO_SRT, SUB_VIDEO]:
        if os.path.exists(f):
            os.remove(f)
    delete_dubbing_files()

def _apply_edits(rows):
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(_4_2_TRANSLATION), exist_ok=True)
    df.to_excel(_4_2_TRANSLATION, index=False)

    _clear_downstream_cache()

    from core._5_split_sub import split_for_sub_main
    from core._6_gen_sub import align_timestamp_main
    split_for_sub_main()
    align_timestamp_main()


def edit_sub_en_section():
    raw_df = _load_translation_df()
    if raw_df is None:
        return False

    st.header("b.5 Review & Edit Subtitles")

    with st.container(border=True):
        st.caption(
            "✏️ Sửa **Bắt đầu / Kết thúc / Gốc / Dịch** — bảng gốc 1 câu = 1 dòng (translation_results.xlsx). "
            "Kéo xuống cuối bảng để thêm dòng mới, bấm 🗑️ đầu dòng để xóa. Định dạng thời gian: HH:MM:SS,mmm "
            "(dùng dấu phẩy, kiểu SRT).\n\n"
            "⚠️ Save/Re-render sẽ chạy lại bước tách câu dài (có gọi LLM nếu câu vượt ngưỡng "
            f"`subtitle.max_length`={load_key('subtitle.max_length')}) và tái tạo lại toàn bộ sub/audio "
            "phía sau — có thể mất chút thời gian."
        )

        editor_df = _df_for_editor(raw_df)
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
            key="sub_editor_en",
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save", use_container_width=True, key="save_en"):
                rows = _editor_df_to_rows(edited_df)
                if rows:
                    with st.spinner("Đang tái tạo sub/audio..."):
                        _apply_edits(rows)
                    st.success(f"✅ Saved {len(rows)} câu. Bấm 'Start Audio Processing' ở bước dưới để dub lại.")
                    st.rerun()

        with b2:
            if st.button("🎬 Re-render preview", use_container_width=True, type="primary", key="rerender_en"):
                rows = _editor_df_to_rows(edited_df)
                if rows:
                    with st.spinner("Đang tái tạo sub/audio + render..."):
                        _apply_edits(rows)
                        from core._7_sub_into_vid import merge_subtitles_to_video
                        merge_subtitles_to_video()
                    st.success("✅ Done!")
                    st.rerun()

        if os.path.exists(SUB_VIDEO):
            st.video(SUB_VIDEO)

    return True