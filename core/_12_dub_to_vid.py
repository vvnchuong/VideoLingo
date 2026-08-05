import platform
import subprocess

import cv2
import numpy as np
from rich.console import Console

from core._1_ytdlp import find_video_files
from core.asr_backend.audio_preprocess import normalize_audio_volume
from core.utils import *
from core.utils.models import *
from core._7_sub_into_vid import (
    _region_to_pixels, _compute_safe_blur_strength, _compute_ocr_margin_v_by_measurement,
)

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = 'output/dub.srt'
DUB_AUDIO = 'output/dub.mp3'

TRANS_FONT_SIZE = 17
TRANS_FONT_NAME = 'Arial'
if platform.system() == 'Linux':
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    TRANS_FONT_NAME = 'Arial Unicode MS'

TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1
TRANS_BACK_COLOR = '&H33000000'


def merge_video_audio():
    """Merge video and audio, and reduce video volume"""
    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE

    if not load_key("burn_subtitles"):
        rprint(
            "[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
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

    # Merge video and audio with translated subtitles
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")

    is_portrait = TARGET_HEIGHT > TARGET_WIDTH
    trans_font_size = load_key("zh_pipeline.portrait_font_size") if is_portrait else TRANS_FONT_SIZE
    if is_portrait:
        rprint(f"[bold cyan]Video dọc (portrait) -> giảm font sub còn {trans_font_size}[/bold cyan]")

    margin_v = compute_sub_margin_v(TARGET_HEIGHT)

    subtitle_source = load_key("subtitle_source")
    ocr_region = load_key("ocr_region")
    use_ocr_crop_style = (
        subtitle_source == "ocr" and ocr_region and ocr_region.get("bottom") is not None
    )

    subtitles_style = (
        f"FontSize={trans_font_size},FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
        f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
        f"BackColour={TRANS_BACK_COLOR},BorderStyle=4"
    )

    if use_ocr_crop_style:
        rprint(f"[bold cyan]OCR có vùng crop -> làm mờ vùng {ocr_region}, đưa sub lên giữa vùng đó[/bold cyan]")
        x1, y1, w, h = _region_to_pixels(ocr_region, TARGET_WIDTH, TARGET_HEIGHT)
        ocr_margin_v = _compute_ocr_margin_v_by_measurement(
            VIDEO_FILE, TARGET_WIDTH, TARGET_HEIGHT, subtitles_style, ocr_region
        )
        rprint(f"[bold cyan]Đo thực nghiệm: MarginV={ocr_margin_v}[/bold cyan]")
        subtitle_filter = (
            f"subtitles={DUB_SUB_FILE}:original_size={TARGET_WIDTH}x{TARGET_HEIGHT}:"
            f"force_style='{subtitles_style},Alignment=2,MarginV={ocr_margin_v}'"
        )
        filter_complex = (
            f"[0:v]split[base][forblur];"
            f"[forblur]crop={w}:{h}:{x1}:{y1},boxblur={_compute_safe_blur_strength(w, h)}[blurred];"
            f"[base][blurred]overlay={x1}:{y1}[withblur];"
            f"[withblur]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"{subtitle_filter}[v];"
            f"[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3:normalize=0[a]"
        )
    else:
        # Hành vi CŨ, giữ nguyên y hệt trước - áp dụng cho Whisper hoặc OCR không
        # có vùng crop tuỳ chỉnh.
        subtitle_filter = (
            f"subtitles={DUB_SUB_FILE}:original_size={TARGET_WIDTH}x{TARGET_HEIGHT}:"
            f"force_style='{subtitles_style},Alignment=2,MarginV={margin_v}'"
        )
        filter_complex = (
            f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
            f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
            f'{subtitle_filter}[v];'
            f'[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3:normalize=0[a]'
        )

    cmd = [
        'ffmpeg', '-y', '-i', VIDEO_FILE, '-i', background_file, '-i', normalized_dub_audio,
        '-filter_complex', filter_complex,
    ]

    if load_key("ffmpeg_gpu"):
        rprint("[bold green]Using GPU acceleration...[/bold green]")
        cmd.extend(['-map', '[v]', '-map', '[a]', '-c:v', 'h264_nvenc'])
    else:
        cmd.extend(['-map', '[v]', '-map', '[a]'])

    cmd.extend(['-c:a', 'aac', '-b:a', '96k', DUB_VIDEO])

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg merge_video_audio thất bại, exit code={result.returncode}")
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")


if __name__ == '__main__':
    merge_video_audio()