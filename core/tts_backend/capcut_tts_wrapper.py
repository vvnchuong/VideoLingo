"""
capcut_tts_wrapper.py

Wrapper khớp đúng interface custom_tts(text, save_path) -> None của VideoLingo,
gọi module capcut_tts.py (đã test kỹ ở bước trước) để sinh audio qua CapCut TTS.

Cách dùng trong core/all_tts_functions/custom_tts.py của VideoLingo:

    from .capcut_tts_wrapper import capcut_custom_tts

    def custom_tts(text: str, save_path: str) -> None:
        capcut_custom_tts(text, save_path)

YÊU CẦU:
- Đặt file này CÙNG THƯ MỤC với capcut_tts.py và capcut_common_task_client.py
  (hoặc đặt cả 3 file vào core/all_tts_functions/ rồi import tương ứng).
"""

from __future__ import annotations

from .capcut_tts import synthesize, CapCutTTSError

# ---------------------------------------------------------------------------
# Danh sách giọng tiếng Việt có sẵn (lấy từ Voice.json của repo K07VN).
# Đổi giọng đang dùng bằng cách sửa ACTIVE_VOICE bên dưới -- không cần tra
# lại resource_id mỗi lần.
# ---------------------------------------------------------------------------
VOICE_PRESETS: dict[str, tuple[str, str]] = {
    # key: (voice_type, resource_id)
    "nho_ngot_ngao":      ("BV421_vivn_streaming", "7252594014782755330"),
    "nu_pho_thong":       ("vi_female_huong", "7264854897953083905"),
    "giong_be":           ("BV074_streaming_dsp", "7550087831092251920"),
    "co_gai_hoat_ngon":   ("BV074_streaming", "7102355709945188865"),  # đã test, chất lượng tốt
    "hoai_my":            ("vi-VN-HoaiMyNeural", "7371666434650280464"),
    "nam_minh":           ("vi-VN-NamMinhNeural", "7371666524727153168"),
    "viet_meo":           ("BV075_streaming_vibrato_dsp", "7569450639810465040"),
    "mai":                ("BV562_streaming", "7483736254694035984"),
    "ban_mai":            ("multi_female_yangguangnv_uranus_bigtts", "7637456432522218773"),
    "review_phim_new":    ("multi_female_richgirl_uranus_bigtts", "7637460351541447956"),
    "ban_tin_1":          ("multi_female_quanweinv_uranus_bigtts", "7637458743197732117"),
    "review_phim_4":      ("multi_female_stokie_uranus_bigtts", "7637456729696996628"),
    "ban_tin_nu":         ("multi_female_sisi_uranus_bigtts", "7637455857285860629"),
    "review_phim_3":      ("multi_female_daqi_uranus_bigtts", "7637451983389019409"),
    "review_phim_2":      ("multi_female_xyf04auto_uranus_bigtts", "7637458743197732117"),
    "sunny_idol":         ("multi_female_kiwi_uranus_bigtts", "7637457995882089749"),
    "kenny_dai_de":       ("BV075_streaming_demon_dsp", "7569442422665661712"),
    "robot_vn":           ("BV075_streaming_robot_dsp", "7538698409633516816"),  # giọng robot, nghe hợp gu
    "giong_nam_tram":     ("multi_male_felipe_uranus_bigtts", "7637456729696996628"),
    "gai_moi_lon":        ("multi_female_peiqi_uranus_bigtts", "7637458789033151751"),
    "nam_ban_tin":        ("multi_female_xinwenjieshuo_uranus_bigtts", "7637455039719640327"),
    "quen_ten_tu_test":   ("multi_female_tianmeijieshuo_uranus_bigtts", "7637460417295469832"),
    "thanh_nien_tu_tin":  ("BV075_streaming", "7102355803792740865"),
    "alex_dai_de":        ("BV560_streaming", "7483736167565758992"),
}

# Đổi tại đây để chọn giọng đang dùng cho toàn bộ pipeline.
ACTIVE_VOICE = "co_gai_hoat_ngon"

DEFAULT_VOICE, DEFAULT_RESOURCE_ID = VOICE_PRESETS[ACTIVE_VOICE]
DEFAULT_RATE = "1.0"


def capcut_custom_tts(
    text: str,
    save_path: str,
    voice: str = DEFAULT_VOICE,
    resource_id: str = DEFAULT_RESOURCE_ID,
    rate: str = DEFAULT_RATE,
) -> None:
    """
    Sinh audio qua CapCut TTS và lưu vào save_path.
    Khớp đúng interface custom_tts(text, save_path) -> None của VideoLingo.

    Raises:
        CapCutTTSError nếu thất bại sau khi đã thử hết device pool + retry.
        (VideoLingo tự bắt exception ở tầng gọi, không cần return status.)
    """
    synthesize(
        text=text,
        voice=voice,
        resource_id=resource_id,
        out_path=save_path,
        rate=rate,
    )


def capcut_custom_tts_by_name(
    text: str,
    save_path: str,
    preset_name: str,
    rate: str = DEFAULT_RATE,
) -> None:
    """
    Giống capcut_custom_tts() nhưng chọn giọng theo tên trong VOICE_PRESETS,
    không cần nhớ voice_type/resource_id.

    Ví dụ:
        capcut_custom_tts_by_name(text, save_path, preset_name="robot_vn")

    Raises:
        KeyError nếu preset_name không tồn tại trong VOICE_PRESETS.
        CapCutTTSError nếu tạo audio thất bại.
    """
    if preset_name not in VOICE_PRESETS:
        available = ", ".join(sorted(VOICE_PRESETS.keys()))
        raise KeyError(f"Không tìm thấy giọng '{preset_name}'. Các giọng có sẵn: {available}")
    voice, resource_id = VOICE_PRESETS[preset_name]
    capcut_custom_tts(text, save_path, voice=voice, resource_id=resource_id, rate=rate)


if __name__ == "__main__":
    # Test nhanh, giống hệt cách VideoLingo sẽ gọi
    capcut_custom_tts("Xin chào, đây là bài kiểm tra wrapper.", "test_output/wrapper_demo.mp3")
    print("OK — đã tạo test_output/wrapper_demo.mp3")