import os, subprocess, time
from core._1_ytdlp import find_video_files
import cv2
import numpy as np
import platform
from core.utils import *

SRC_FONT_SIZE = 15
TRANS_FONT_SIZE = 17
FONT_NAME = 'Arial'
TRANS_FONT_NAME = 'Arial'

# Linux need to install google noto fonts: apt-get install fonts-noto
if platform.system() == 'Linux':
    FONT_NAME = 'NotoSansCJK-Regular'
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
# Mac OS has different font names
elif platform.system() == 'Darwin':
    FONT_NAME = 'Arial Unicode MS'
    TRANS_FONT_NAME = 'Arial Unicode MS'

SRC_FONT_COLOR = '&HFFFFFF'
SRC_OUTLINE_COLOR = '&H000000'
SRC_OUTLINE_WIDTH = 1
SRC_SHADOW_COLOR = '&H80000000'
TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1
TRANS_BACK_COLOR = '&H33000000'

OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/output_sub.mp4"
SRC_SRT = f"{OUTPUT_DIR}/src.srt"
TRANS_SRT = f"{OUTPUT_DIR}/trans.srt"

def _compute_safe_blur_strength(crop_w: int, crop_h: int) -> str:
    max_radius = max(3, min(20, min(crop_w, crop_h) * 4 // 10))
    chroma_radius = min(max_radius, 9)  # 9 luôn an toàn, thấp hơn hẳn ngưỡng 10-11 đã gặp
    return f"{max_radius}:5:{chroma_radius}:5"
LEGACY_PLAYRES_HEIGHT = 288


def _region_to_pixels(region: dict, target_width: int, target_height: int):
    x1 = int(target_width * region["left"])
    y1 = int(target_height * region["top"])
    x2 = int(target_width * region["right"])
    y2 = int(target_height * region["bottom"])
    w = max(2, (x2 - x1) // 2 * 2)
    h = max(2, (y2 - y1) // 2 * 2)
    return x1, y1, w, h


def _region_center_y_ratio(region: dict) -> float:
    """Tỉ lệ 0.0-1.0 của điểm giữa vùng crop theo trục dọc, tính từ đỉnh khung hình."""
    return (region["top"] + region["bottom"]) / 2


def _compute_ocr_margin_v(target_height: int, region: dict, font_size: int) -> int:
    """
    QUAN TRỌNG: khi ffmpeg convert .srt sang ASS nội bộ (filter "subtitles="),
    nó LUÔN tạo script với PlayResY=288 cố định (mặc định của ffmpeg), bất kể
    video thật cao bao nhiêu pixel, và "original_size" KHÔNG thay đổi điều này
    (option đó chỉ sửa lỗi tỉ lệ khung hình/font-scale, không đổi PlayResY).
    MarginV trong ASS luôn chạy theo hệ toạ độ PlayResY=288 này, KHÔNG PHẢI
    pixel thật của video. Nếu tính MarginV theo pixel thật (vd video 1920px
    cao) sẽ ra số > 288, khiến libass đẩy chữ ra ngoài khung ảo -> mất chữ
    hoàn toàn (không lệch vị trí, mà biến mất luôn, không vẽ gì cả).

    Vậy phải tính theo TỈ LỆ (0.0-1.0) rồi nhân với 288, không phải target_height.

    Yêu cầu: DÒNG ĐẦU (dòng duy nhất nếu 1 dòng) của sub phải nằm giữa vùng
    crop theo chiều dọc. Dòng 2 (nếu sub bị wrap 2 dòng) cứ xếp xuống dưới dòng
    1 bình thường, tràn ra ngoài crop cũng không sao.

    Alignment=2 (bottom-center) neo theo ĐÁY của dòng cuối cùng, cách đáy khung
    ẢO (cao 288) MarginV đơn vị ảo. font_size cũng chạy theo cùng thang ảo này
    (đó là lý do portrait_font_size=10 vẫn đọc được - nó không phải pixel thật,
    mà là đơn vị theo thang PlayResY=288).
    """
    line_height = font_size * 1.3
    center_crop_ratio = _region_center_y_ratio(region)  # 0.0-1.0
    center_crop_y_virtual = LEGACY_PLAYRES_HEIGHT * center_crop_ratio
    bottom_of_first_line = center_crop_y_virtual + line_height / 2
    margin_v = LEGACY_PLAYRES_HEIGHT - bottom_of_first_line
    return max(0, round(margin_v))


def _measure_subtitle_center_y(video_file, target_width, target_height, subtitles_style,
                                margin_v_probe, use_original_size=False):
    """
    use_original_size PHẢI khớp chính xác với filter dùng lúc burn sub thật sự
    (subtitles=...:original_size=WxH có mặt hay không). MarginV trong ASS chạy
    theo hệ toạ độ PlayResY - nếu không có original_size, libass tự suy PlayResY
    từ script (thường ra một giá trị ảo, không phải chiều cao pixel thật của
    video). Đo ở một hệ toạ độ rồi burn ở hệ toạ độ khác sẽ làm sub lệch hẳn
    khỏi vùng crop mong muốn.
    """
    probe_srt = os.path.join(OUTPUT_DIR, "_probe.srt")
    probe_png = os.path.join(OUTPUT_DIR, "_probe.png")
    with open(probe_srt, "w", encoding="utf-8") as f:
        f.write("1\n00:00:00,000 --> 00:00:05,000\nA\n")

    probe_srt_escaped = probe_srt.replace("\\", "/").replace(":", "\\:")
    original_size_part = f":original_size={target_width}x{target_height}" if use_original_size else ""

    cmd = [
        'ffmpeg', '-y', '-i', video_file,
        '-vf', (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
            f"subtitles={probe_srt_escaped}{original_size_part}:"
            f"force_style='{subtitles_style},Alignment=2,MarginV={margin_v_probe}'"
        ).encode('utf-8'),
        '-frames:v', '1', '-update', '1', probe_png,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(probe_png):
        print(f"[probe-debug] ffmpeg stderr: {result.stderr[-800:]}", flush=True)

    img = cv2.imread(probe_png)
    os.remove(probe_srt)
    if os.path.exists(probe_png):
        os.remove(probe_png)
    if img is None:
        return None

    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    yellow_mask = (r > 180) & (g > 180) & (b < 120)
    ys, _ = np.where(yellow_mask)
    if len(ys) == 0:
        return None
    return int((ys.min() + ys.max()) / 2)


def _compute_ocr_margin_v_by_measurement(video_file, target_width, target_height,
                                          subtitles_style, region, use_original_size=False):
    probe_margin_1, probe_margin_2 = 20, 200
    y1 = _measure_subtitle_center_y(video_file, target_width, target_height, subtitles_style,
                                     probe_margin_1, use_original_size)
    y2 = _measure_subtitle_center_y(video_file, target_width, target_height, subtitles_style,
                                     probe_margin_2, use_original_size)

    target_y = int(target_height * _region_center_y_ratio(region))

    if y1 is None or y2 is None or y1 == y2:
        # Đo thất bại (hiếm khi xảy ra, vd font không load được) - fallback về giữa
        # khung ảo 288 như cách cũ, còn hơn không có gì.
        rprint("[bold yellow]Không đo được thực nghiệm vị trí sub, dùng công thức ước lượng dự phòng.[/bold yellow]")
        return max(0, min(260, round((1 - _region_center_y_ratio(region)) * LEGACY_PLAYRES_HEIGHT)))

    a = (y2 - y1) / (probe_margin_2 - probe_margin_1)
    b = y1 - a * probe_margin_1
    margin_v = (target_y - b) / a
    return max(0, round(margin_v))


def check_gpu_available():
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        return 'h264_nvenc' in result.stdout
    except:
        return False


def merge_subtitles_to_video():
    video_file = find_video_files()
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    # Check resolution
    if not load_key("burn_subtitles"):
        rprint(
            "[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    if not os.path.exists(TRANS_SRT):
        rprint("Subtitle files not found in the 'output' directory.")
        exit(1)

    video = cv2.VideoCapture(video_file)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")

    # Video dọc (portrait/short: cao > rộng) -> chữ nhỏ lại cho đỡ che video
    is_portrait = TARGET_HEIGHT > TARGET_WIDTH
    trans_font_size = load_key("zh_pipeline.portrait_font_size") if is_portrait else TRANS_FONT_SIZE
    if is_portrait:
        rprint(f"[bold cyan]Video dọc (portrait) -> giảm font sub còn {trans_font_size}[/bold cyan]")

    subtitle_source = load_key("subtitle_source")
    ocr_region = load_key("ocr_region")
    use_ocr_crop_style = (
        subtitle_source == "ocr" and ocr_region and ocr_region.get("bottom") is not None
    )

    subtitles_style = (
        f"FontSize={trans_font_size},FontName={TRANS_FONT_NAME},"
        f"PrimaryColour={TRANS_FONT_COLOR},OutlineColour={TRANS_OUTLINE_COLOR},"
        f"OutlineWidth={TRANS_OUTLINE_WIDTH},BackColour={TRANS_BACK_COLOR},BorderStyle=4"
    )

    if use_ocr_crop_style:
        rprint(f"[bold cyan]OCR có vùng crop -> làm mờ vùng {ocr_region}, đưa sub lên giữa vùng đó[/bold cyan]")
        x1, y1, w, h = _region_to_pixels(ocr_region, TARGET_WIDTH, TARGET_HEIGHT)
        blur_filter_str = (
            f"split[base][forblur];"
            f"[forblur]crop={w}:{h}:{x1}:{y1},boxblur={_compute_safe_blur_strength(w, h)}[blurred];"
            f"[base][blurred]overlay={x1}:{y1}[withblur]"
        )
        # Tính MarginV trực tiếp theo pixel thật của vùng crop, không render thử
        # gì cả (xem ghi chú trong _compute_ocr_margin_v).
        margin_v = _compute_ocr_margin_v(TARGET_HEIGHT, ocr_region, trans_font_size)
        rprint(f"[bold cyan]MarginV tính theo vùng crop: {margin_v}[/bold cyan]")
        # QUAN TRỌNG: trong filter_complex, dấu ':' và '\' trong đường dẫn phải
        # được escape, nếu không ffmpeg parse sai cú pháp filter (path Windows
        # kiểu D:\video\... có dấu ':' sẽ làm vỡ toàn bộ filter_complex, khiến
        # burn sub thất bại âm thầm tuỳ máy/tuỳ đường dẫn video).
        trans_srt_escaped = TRANS_SRT.replace("\\", "/").replace(":", "\\:")
        vf_chain = (
            f"{blur_filter_str};"
            f"[withblur]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"subtitles={trans_srt_escaped}:original_size={TARGET_WIDTH}x{TARGET_HEIGHT}:"
            f"force_style='{subtitles_style},Alignment=2,MarginV={margin_v}'[final]"
        )
        ffmpeg_cmd = [
            'ffmpeg', '-i', video_file,
            '-filter_complex', vf_chain.encode('utf-8'),
            '-map', '[final]', '-map', '0:a?',
        ]
    else:
        margin_v = compute_sub_margin_v(TARGET_HEIGHT)
        trans_srt_escaped = TRANS_SRT.replace("\\", "/").replace(":", "\\:")
        ffmpeg_cmd = [
            'ffmpeg', '-i', video_file,
            '-vf', (
                f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
                f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
                f"subtitles={trans_srt_escaped}:force_style='{subtitles_style},Alignment=2,MarginV={margin_v}'"
            ).encode('utf-8'),
        ]

    ffmpeg_gpu = load_key("ffmpeg_gpu")
    if ffmpeg_gpu:
        rprint("[bold green]will use GPU acceleration.[/bold green]")
        ffmpeg_cmd.extend(['-c:v', 'h264_nvenc'])
    ffmpeg_cmd.extend(['-y', OUTPUT_VIDEO])

    rprint("🎬 Start merging subtitles to video...")
    start_time = time.time()
    process = subprocess.Popen(ffmpeg_cmd)

    try:
        process.wait()
        if process.returncode == 0:
            rprint(f"\n✅ Done! Time taken: {time.time() - start_time:.2f} seconds")
        else:
            rprint("\n❌ FFmpeg execution error")
    except Exception as e:
        rprint(f"\n❌ Error occurred: {e}")
        if process.poll() is None:
            process.kill()


if __name__ == "__main__":
    merge_subtitles_to_video()