"""
3 style phụ đề TĨNH (không phải karaoke - đó là hướng riêng, dùng .ass, làm sau).
Dùng chung cho cả _7_sub_into_vid.py (sub-only) và _12_dub_to_vid.py (dub) - style
áp dụng cho CẢ 2 nút "Chỉ Sub mới" lẫn "Sub mới + Lồng tiếng" (đã xác nhận với user).

Format ASS style string (dùng trong ffmpeg force_style=...), theo đúng field:
Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour,
BackColour, Bold, Alignment, MarginL, MarginR, MarginV, Encoding

Ở đây chỉ cần các field ffmpeg force_style hỗ trợ trực tiếp - không cần Name/
SecondaryColour/Alignment/Margin* (những cái đó code đã tự thêm riêng ở nơi gọi,
ví dụ Alignment=2,MarginV=... được nối thêm sau, xem SUBTITLE_STYLE_PRESETS usage).
"""

# key khớp đúng giá trị lưu trong VideoJob.subtitleStyle (Java) / --subtitle-style
# (CLI) - xem run_job.py và VideoJob.java.
SUBTITLE_STYLE_PRESETS = {
    # Dạng 1: Nền vàng tươi, chữ đen
    "yellow_black": {
        "font_color": "&H00000000",
        "outline_color": "&H00000000",
        "back_color": "&H0000D7FF",
        "border_style": 4,  # 4 = có nền hộp (opaque box), khớp cách BorderStyle cũ đang dùng
    },
    # Dạng 2 (MẶC ĐỊNH): Nền trắng, chữ đen
    "white_black": {
        "font_color": "&H00000000",
        "outline_color": "&H00000000",
        "back_color": "&H00FFFFFF",
        "border_style": 4,
    },
    # Dạng 3: Nền đen mờ (50%), chữ trắng
    "black_opaque_white": {
        "font_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "back_color": "&H80000000",
        "border_style": 4,
    },
}

DEFAULT_SUBTITLE_STYLE = "white_black"


def get_subtitle_style_colors(style_key: str) -> dict:
    """Trả về preset màu theo key, fallback về mặc định (trắng-đen) nếu key lạ/rỗng -
    không raise lỗi để job cũ (chưa có subtitle_style trong config.yaml/DB) vẫn
    chạy đúng với hành vi mặc định, không bị crash."""
    return SUBTITLE_STYLE_PRESETS.get(style_key, SUBTITLE_STYLE_PRESETS[DEFAULT_SUBTITLE_STYLE])