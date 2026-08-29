import os

import platform
import subprocess

import cv2
import numpy as np
from rich.console import Console

from core._1_ytdlp import find_video_files
from core.asr_backend.audio_preprocess import normalize_audio_volume
from core.utils import *
from core.utils.models import *

# IMPORT HÀM RENDER ẢNH BO GÓC (PILLOW) TỪ FILE SỐ 7
from core._7_sub_into_vid import (
    _region_to_pixels, _compute_safe_blur_strength, generate_png_sequence
)
from core.subtitle_style_presets import get_subtitle_style_colors, DEFAULT_SUBTITLE_STYLE

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = 'output/dub.srt'
DUB_AUDIO = 'output/dub.mp3'
_BACKGROUND_AUDIO_FILE = 'output/audio/background.mp3'

TRANS_FONT_SIZE = 17
TRANS_FONT_NAME = 'Arial'
if platform.system() == 'Linux':
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    TRANS_FONT_NAME = 'Arial Unicode MS'

BG_AUDIO_PREFILTER = (
    "atempo=1.04,asetrate=44100*1.02,aresample=44100,"
    "highpass=f=100,lowpass=f=15000"
)


def _build_audio_filter(background_music_volume: float, has_bg: bool) -> str:
    """Xử lý đúng chỉ số input của FFmpeg tùy theo việc video có nhạc nền hay không.

    QUAN TRỌNG - lỗi thật đã gặp (filtergraph "Invalid argument"): input 0 là
    VIDEO_FILE, input 1 là concat_txt_path (CŨNG LÀ VIDEO, chuỗi PNG overlay) -
    audio ĐẦU TIÊN chỉ bắt đầu từ input 2, không phải input 1 như code cũ giả
    định. Nếu has_bg=True: nhạc nền=[2:a], dub=[3:a]. Nếu has_bg=False:
    dub=[2:a] (không phải [1:a]).

    Nhạc nền luôn đi qua BG_AUDIO_PREFILTER (tempo/pitch/EQ) trước khi mix.

    QUAN TRỌNG #2 - lỗi thật đã gặp (mất tiếng giữa chừng, cả dub lẫn nhạc
    nền): amix dùng duration=first trước đây, nghĩa là độ dài audio output
    bị ép cứng theo INPUT ĐẦU TIÊN của amix, tức [bg] (nhạc nền) - và nhạc
    nền đã qua atempo=1.04 trong BG_AUDIO_PREFILTER nên NGẮN HƠN bản gốc
    ~4%. Nếu nhạc nền ngắn hơn giọng dub (rất hay xảy ra vì dub th ường dài
    hơn do speed_factor không co đủ, hoặc nhạc nền chỉ là 1 đoạn lặp ngắn
    hơn tổng video) -> toàn bộ audio output (cả dub + nhạc nền) bị cắt cụt
    đúng theo độ dài nhạc nền, dù dub vẫn còn nội dung. Đổi sang
    duration=longest để amix kéo dài theo track dài nhất trong 2 track,
    không cắt mất phần dub còn lại khi nhạc nền ngắn hơn.
    """
    dub_audio_idx = 3 if has_bg else 2
    if not has_bg or background_music_volume <= 0:
        # Không có nhạc nền hoặc chọn xóa (0) -> Chỉ lấy giọng dub
        return f"[{dub_audio_idx}:a]anull[a]"
    if background_music_volume >= 1.0:
        # Giữ nguyên âm lượng nhạc nền -> prefilter rồi mix [2:a] với giọng dub [3:a]
        return (
            f"[2:a]{BG_AUDIO_PREFILTER}[bg];"
            f"[bg][{dub_audio_idx}:a]amix=inputs=2:duration=longest:dropout_transition=3:normalize=0[a]"
        )

    # Giữ nhỏ (vd: 0.05) -> prefilter, giảm âm lượng nhạc nền [2:a] rồi mix với giọng dub [3:a]
    return (
        f"[2:a]{BG_AUDIO_PREFILTER},volume={background_music_volume}[bg_reduced];"
        f"[bg_reduced][{dub_audio_idx}:a]amix=inputs=2:duration=longest:dropout_transition=3:normalize=0[a]"
    )


def merge_video_audio():
    """Merge video and audio, and overlay Pillow Vector Subtitles"""
    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE

    try:
        background_music_volume = float(load_key("background_music_volume"))
    except KeyError:
        background_music_volume = 1.0

    if not load_key("burn_subtitles"):
        rprint(
            "[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(DUB_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()
        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    # Normalize dub audio
    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)

    # Retrieve Resolution
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")

    is_portrait = TARGET_HEIGHT > TARGET_WIDTH
    trans_font_size = load_key("src_pipeline.portrait_font_size") if is_portrait else TRANS_FONT_SIZE

    subtitle_source = load_key("subtitle_source")
    ocr_region = load_key("ocr_region")
    use_ocr_crop_style = (subtitle_source == "ocr" and ocr_region and ocr_region.get("bottom") is not None)

    # Whisper source: user có thể tự chọn vị trí sub qua thanh ngang kéo dọc - chỉ
    # đọc khi KHÔNG dùng ocr crop style. Bọc try/except vì key mới thêm, config.yaml
    # job cũ chưa có sẵn.
    subtitle_position = None
    if not use_ocr_crop_style:
        try:
            subtitle_position = load_key("subtitle_position")
        except KeyError:
            subtitle_position = None

    try:
        style_key = load_key("subtitle_style")
    except KeyError:
        style_key = DEFAULT_SUBTITLE_STYLE

    # ĐIỂM CỐT LÕI: Gọi Engine Pillow để vẽ hàng loạt khung Sub bo góc
    rprint(f"[bold green]🎨 Generating perfectly rounded PNG sequence for DUB...[/bold green]")
    # QUAN TRỌNG: chuỗi PNG phải kéo dài tới hết audio DUB thật (không phải
    # chỉ tới câu sub cuối cùng), nếu không overlay sẽ cắt cụt cả video+audio
    # output theo đúng khoảng chênh này - đã gặp thực tế (mất tiếng ở cuối).
    # Lấy MAX(video gốc, audio dub) vì dub có thể dài hơn video gốc một chút
    # dù đã áp speed_factor (không phải lúc nào cũng co vừa khít).
    from core._7_sub_into_vid import _get_media_duration_sec
    video_duration_sec = _get_media_duration_sec(VIDEO_FILE)
    dub_duration_sec = _get_media_duration_sec(normalized_dub_audio)
    total_duration_sec = max(video_duration_sec, dub_duration_sec)
    concat_txt_path = generate_png_sequence(
        DUB_SUB_FILE, TARGET_WIDTH, TARGET_HEIGHT, trans_font_size, style_key, use_ocr_crop_style, ocr_region,
        subtitle_position, total_duration_sec
    )

    base_video_vf = f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2"

    if use_ocr_crop_style:
        rprint(f"[bold cyan]OCR có vùng crop -> làm mờ vùng {ocr_region}, đưa sub lên giữa vùng đó[/bold cyan]")
        x1, y1, w, h = _region_to_pixels(ocr_region, TARGET_WIDTH, TARGET_HEIGHT)
        vf_chain = (
            f"[0:v]{base_video_vf}[scaled_vid];"
            f"[scaled_vid]split[base][forblur];"
            f"[forblur]crop={w}:{h}:{x1}:{y1},boxblur={_compute_safe_blur_strength(w, h)}[blurred];"
            f"[base][blurred]overlay={x1}:{y1}[withblur];"
            f"[withblur][1:v]overlay=0:0[v];"
        )
    else:
        vf_chain = (
            f"[0:v]{base_video_vf}[scaled_vid];"
            f"[scaled_vid][1:v]overlay=0:0[v];"
        )

    cmd = [
        'ffmpeg', '-y',
        '-i', VIDEO_FILE,  # [0:v] Video Gốc
        '-f', 'concat', '-safe', '0', '-i', concat_txt_path  # [1:v] Chuỗi hình ảnh Subtitle PNG
    ]

    # Kiểm tra xem file background.mp3 có thực tế tồn tại không rồi mới đưa vào FFmpeg
    bg_audio_idx = -1
    dub_audio_idx = -1
    # Kiểm tra xem file background.mp3 có thực tế tồn tại không
    has_background = background_music_volume > 0 and os.path.exists(background_file)

    cmd = [
        'ffmpeg', '-y',
        '-i', VIDEO_FILE,  # [0:v] Video Gốc
        '-f', 'concat', '-safe', '0', '-i', concat_txt_path  # [1:v] Chuỗi hình ảnh Subtitle PNG
    ]

    if has_background:
        cmd.extend(['-i', background_file])  # [1:a] Nhạc nền gốc
        cmd.extend(['-i', normalized_dub_audio])  # [2:a] Giọng đọc Dub
    else:
        cmd.extend(['-i', normalized_dub_audio])  # [1:a] Chỉ có Giọng đọc Dub

    filter_complex = vf_chain + _build_audio_filter(background_music_volume, has_background)
    cmd.extend(['-filter_complex', filter_complex])
    if load_key("ffmpeg_gpu"):
        rprint("[bold green]Using GPU acceleration...[/bold green]")
        cmd.extend(['-map', '[v]', '-map', '[a]', '-c:v', 'h264_nvenc'])
    else:
        cmd.extend(['-map', '[v]', '-map', '[a]'])
        ffmpeg_threads = load_key("ffmpeg_threads")
        if ffmpeg_threads and ffmpeg_threads > 0:
            cmd.extend(['-threads', str(ffmpeg_threads)])

    cmd.extend(['-c:a', 'aac', '-b:a', '96k', DUB_VIDEO])

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg merge_video_audio thất bại, exit code={result.returncode}")
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")


if __name__ == '__main__':
    merge_video_audio()