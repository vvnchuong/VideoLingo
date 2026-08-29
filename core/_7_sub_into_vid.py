import os, subprocess, time
import cv2
import numpy as np
import platform
import re
from PIL import Image, ImageDraw, ImageFont
from core._1_ytdlp import find_video_files
from core.utils import *
from core.subtitle_style_presets import get_subtitle_style_colors, DEFAULT_SUBTITLE_STYLE

SRC_FONT_SIZE = 15
TRANS_FONT_SIZE = 17
FONT_NAME = 'arial.ttf'
TRANS_FONT_NAME = 'arial.ttf'

OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/output_sub.mp4"
TRANS_SRT = f"{OUTPUT_DIR}/trans.srt"
CONCAT_TXT = f"{OUTPUT_DIR}/concat_subs.txt"
SUBS_PNG_DIR = f"{OUTPUT_DIR}/subs_png"

LEGACY_PLAYRES_HEIGHT = 288

def _compute_ocr_margin_v(ocr_region: dict, target_height: int) -> int:
    if ocr_region and "bottom" in ocr_region:
        # Tính khoảng cách từ đáy vùng OCR đến đáy video
        margin_bottom = target_height - int(ocr_region["bottom"] * target_height)
        return margin_bottom
    return 10


def time_to_sec(t_str):
    t_str = t_str.strip().replace(',', '.')
    parts = t_str.split(':')
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])


def get_system_font(font_name, size):
    try:
        return ImageFont.truetype(font_name, size)
    except IOError:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except IOError:
            # Fallback for Mac/Linux
            for f in ["/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "NotoSansCJK-Regular.ttc"]:
                try:
                    return ImageFont.truetype(f, size)
                except IOError:
                    continue
    rprint("[bold red]CẢNH BÁO: Không tìm thấy font TTF chuẩn, đang dùng font mặc định (sẽ bị lỗi size)![/bold red]")
    return ImageFont.load_default()


def wrap_text(text, font, max_width, draw):
    lines = []
    for raw_line in text.split('\n'):
        words = raw_line.split()
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            # Đo bề ngang của dòng chữ
            if draw.textlength(test_line, font=font) <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
    return "\n".join(lines)

def _get_media_duration_sec(path: str) -> float:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True, timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def generate_png_sequence(srt_path, target_w, target_h, font_size_legacy, style_mode, use_ocr, ocr_region, subtitle_position=None, total_duration_sec=0.0):
    if not os.path.exists(srt_path):
        return None

    os.makedirs(SUBS_PNG_DIR, exist_ok=True)

    # 1. Setup Màu sắc
    if style_mode == "yellow_black":
        text_color = (0, 0, 0, 255)
        bg_color = (255, 215, 0, 255)
    elif style_mode == "white_black":
        text_color = (0, 0, 0, 255)
        bg_color = (255, 255, 255, 255)
    else:  # white_opaque
        text_color = (255, 255, 255, 255)
        bg_color = (0, 0, 0, 150)

    real_font_size = int((font_size_legacy / LEGACY_PLAYRES_HEIGHT) * target_h)
    font = get_system_font(TRANS_FONT_NAME, real_font_size)

    # 3. Tính toán Tọa độ Tâm Y của Subtitle
    if use_ocr:
        center_y = int((ocr_region["top"] + ocr_region["bottom"]) / 2 * target_h)
    elif subtitle_position and subtitle_position.get("top") is not None and subtitle_position.get("bottom") is not None:
        center_y = int((subtitle_position["top"] + subtitle_position["bottom"]) / 2 * target_h)
    else:
        # Default: Căn cách đáy màn hình 12%
        center_y = int(target_h - (target_h * 0.12))

    # Tạo file ảnh rỗng (Blank)
    blank_path = os.path.join(SUBS_PNG_DIR, "blank.png")
    Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0)).save(blank_path)

    # Đọc SRT
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    sub_data = []

    for idx, block in enumerate(blocks):
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            time_line = lines[1]
            text = "\n".join(lines[2:]).strip()  # Giữ nguyên cấu trúc xuống dòng

            times = time_line.split(' --> ')
            start_sec = time_to_sec(times[0])
            end_sec = time_to_sec(times[1])

            # --- RENDER ẢNH PNG ---
            img = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # GIỚI HẠN CHIỀU RỘNG: Chữ chỉ được chiếm tối đa 90% chiều rộng màn hình
            max_text_w = target_w * 0.90
            wrapped_text = wrap_text(text, font, max_text_w, draw)

            # Tính toán kích thước Box bọc trọn vẹn dựa trên text ĐÃ XUỐNG DÒNG
            bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            pad_x = int(real_font_size * 0.8)  # Padding ngang
            pad_y = int(real_font_size * 0.5)  # Padding dọc
            box_w = text_w + pad_x * 2
            box_h = text_h + pad_y * 2

            box_x1 = (target_w - box_w) // 2
            box_y1 = center_y - (box_h // 2)
            box_x2 = box_x1 + box_w
            box_y2 = box_y1 + box_h

            # VẼ HÌNH CHỮ NHẬT BO GÓC
            radius = int(box_h * 0.2)
            draw.rounded_rectangle([box_x1, box_y1, box_x2, box_y2], radius=radius, fill=bg_color)

            # Viết chữ đè lên (nhớ truyền wrapped_text thay vì text gốc)
            draw.multiline_text((target_w // 2, center_y), wrapped_text, font=font, fill=text_color, anchor="mm",
                                align="center")

            img_filename = f"sub_{idx}.png"
            img.save(os.path.join(SUBS_PNG_DIR, img_filename))

            sub_data.append({"start": start_sec, "end": end_sec, "file": img_filename})

    # --- TẠO FILE KỊCH BẢN CONCAT DEMUXER CHO FFMPEG ---
    with open(CONCAT_TXT, 'w', encoding='utf-8') as f:
        f.write("ffconcat version 1.0\n")
        current_time = 0.0

        for sub in sub_data:
            if sub["start"] > current_time:
                f.write(f"file 'subs_png/blank.png'\n")
                f.write(f"duration {sub['start'] - current_time:.3f}\n")

            f.write(f"file 'subs_png/{sub['file']}'\n")
            f.write(f"duration {sub['end'] - sub['start']:.3f}\n")
            current_time = sub["end"]

        remaining = total_duration_sec - current_time
        if remaining > 0.01:
            f.write(f"file 'subs_png/blank.png'\n")
            f.write(f"duration {remaining:.3f}\n")
            f.write(f"file 'subs_png/blank.png'\n")
        else:
            f.write(f"file 'subs_png/blank.png'\n")

    return CONCAT_TXT


def _compute_safe_blur_strength(crop_w: int, crop_h: int) -> str:
    max_radius = max(3, min(20, min(crop_w, crop_h) * 4 // 10))
    chroma_radius = min(max_radius, 9)
    return f"{max_radius}:5:{chroma_radius}:5"


def _region_to_pixels(region: dict, target_width: int, target_height: int):
    x1 = int(target_width * region["left"])
    y1 = int(target_height * region["top"])
    x2 = int(target_width * region["right"])
    y2 = int(target_height * region["bottom"])
    w = max(2, (x2 - x1) // 2 * 2)
    h = max(2, (y2 - y1) // 2 * 2)
    return x1, y1, w, h


def merge_subtitles_to_video():
    video_file = find_video_files()
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    if not load_key("burn_subtitles"):
        return

    if not os.path.exists(TRANS_SRT):
        rprint("Subtitle files not found.")
        exit(1)

    video = cv2.VideoCapture(video_file)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")

    is_portrait = TARGET_HEIGHT > TARGET_WIDTH
    trans_font_size = load_key("src_pipeline.portrait_font_size") if is_portrait else TRANS_FONT_SIZE

    subtitle_source = load_key("subtitle_source")
    ocr_region = load_key("ocr_region")
    use_ocr_crop_style = (subtitle_source == "ocr" and ocr_region and ocr_region.get("bottom") is not None)

    subtitle_position = None
    if not use_ocr_crop_style:
        try:
            subtitle_position = load_key("subtitle_position")
        except KeyError:
            subtitle_position = None

    try:
        style_key = load_key("subtitle_style")
    except KeyError:
        style_key = 'yellow_black'

    rprint(f"[bold green]🎨 Generating perfectly rounded PNG sequence...[/bold green]")
    total_duration_sec = _get_media_duration_sec(video_file)
    generate_png_sequence(TRANS_SRT, TARGET_WIDTH, TARGET_HEIGHT, trans_font_size, style_key, use_ocr_crop_style,
                          ocr_region, subtitle_position, total_duration_sec)

    # LỆNH FFMPEG (Overlay ảnh Subtitle lên Video gốc)
    base_video_vf = f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2"

    if use_ocr_crop_style:
        x1, y1, w, h = _region_to_pixels(ocr_region, TARGET_WIDTH, TARGET_HEIGHT)
        vf_chain = (
            f"[0:v]{base_video_vf}[scaled_vid];"
            f"[scaled_vid]split[base][forblur];"
            f"[forblur]crop={w}:{h}:{x1}:{y1},boxblur={_compute_safe_blur_strength(w, h)}[blurred];"
            f"[base][blurred]overlay={x1}:{y1}[withblur];"
            f"[withblur][1:v]overlay=0:0[final]"
        )
    else:
        vf_chain = (
            f"[0:v]{base_video_vf}[scaled_vid];"
            f"[scaled_vid][1:v]overlay=0:0[final]"
        )

    ffmpeg_cmd = [
        'ffmpeg', '-i', video_file,
        '-f', 'concat', '-safe', '0', '-i', CONCAT_TXT,
        '-filter_complex', vf_chain,
        '-map', '[final]', '-map', '0:a?'
    ]

    ffmpeg_gpu = load_key("ffmpeg_gpu")
    if ffmpeg_gpu:
        ffmpeg_cmd.extend(['-c:v', 'h264_nvenc'])
    else:
        ffmpeg_threads = load_key("ffmpeg_threads")
        if ffmpeg_threads and ffmpeg_threads > 0:
            ffmpeg_cmd.extend(['-threads', str(ffmpeg_threads)])

    ffmpeg_cmd.extend(['-y', OUTPUT_VIDEO])

    rprint("🎬 Start overlaying PNG sequence to video...")
    start_time = time.time()
    process = subprocess.Popen(ffmpeg_cmd)

    try:
        process.wait()
        if process.returncode == 0:
            rprint(f"\n✅ Done! Khung bo góc chuẩn CapCut. Time taken: {time.time() - start_time:.2f} seconds")
        else:
            rprint("\n❌ FFmpeg execution error")
    except Exception as e:
        rprint(f"\n❌ Error occurred: {e}")
        if process.poll() is None:
            process.kill()


if __name__ == "__main__":
    merge_subtitles_to_video()