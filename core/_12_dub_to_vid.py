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
    _region_to_pixels, _compute_safe_blur_strength, _compute_ocr_margin_v,
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

# Xử lý nhạc nền trước khi mix: tăng nhẹ tốc độ + cao độ, lọc bớt tần số quá
# trầm/quá cao. KHÔNG hạ volume ở đây nữa - giữ nguyên âm lượng gốc, việc giảm
# âm lượng (nếu có) do background_music_volume trong config.yaml quyết định.
BG_AUDIO_PREFILTER = (
    "atempo=1.04,asetrate=44100*1.02,aresample=44100,"
    "highpass=f=100,lowpass=f=15000"
)


def _build_audio_filter(background_music_volume: float) -> str:
    """Trộn giọng dub với nhạc nền gốc theo đúng % âm lượng chỉ định (0.0-1.0):
    0 = xoá hoàn toàn (bỏ hẳn input background, không mix gì), 1.0 = giữ nguyên như
    cũ, giá trị giữa (vd 0.05) = giảm âm lượng nhạc nền trước khi mix. Input index
    đổi theo có background_file hay không - xem chỗ gọi cmd.extend ở
    merge_video_audio(). Trước khi mix, nhạc nền luôn đi qua BG_AUDIO_PREFILTER
    (tempo/pitch/EQ)."""
    if background_music_volume <= 0:
        return "[1:a]anull[a]"
    if background_music_volume >= 1.0:
        return (
            f"[1:a]{BG_AUDIO_PREFILTER}[bg];"
            f"[bg][2:a]amix=inputs=2:duration=first:dropout_transition=3:normalize=0[a]"
        )
    return (
        f"[1:a]{BG_AUDIO_PREFILTER},volume={background_music_volume}[bg_reduced];"
        f"[bg_reduced][2:a]amix=inputs=2:duration=first:dropout_transition=3:normalize=0[a]"
    )


def merge_video_audio():
    """Merge video and audio, and reduce video volume"""
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

    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)

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

    dub_sub_file_escaped = DUB_SUB_FILE.replace("\\", "/").replace(":", "\\:")

    if use_ocr_crop_style:
        rprint(f"[bold cyan]OCR có vùng crop -> làm mờ vùng {ocr_region}, đưa sub lên giữa vùng đó[/bold cyan]")
        x1, y1, w, h = _region_to_pixels(ocr_region, TARGET_WIDTH, TARGET_HEIGHT)
        ocr_margin_v = _compute_ocr_margin_v(TARGET_HEIGHT, ocr_region, trans_font_size)
        rprint(f"[bold cyan]MarginV tính theo vùng crop: {ocr_margin_v}[/bold cyan]")
        subtitle_filter = (
            f"subtitles={dub_sub_file_escaped}:original_size={TARGET_WIDTH}x{TARGET_HEIGHT}:"
            f"force_style='{subtitles_style},Alignment=2,MarginV={ocr_margin_v}'"
        )
        filter_complex = (
            f"[0:v]split[base][forblur];"
            f"[forblur]crop={w}:{h}:{x1}:{y1},boxblur={_compute_safe_blur_strength(w, h)}[blurred];"
            f"[base][blurred]overlay={x1}:{y1}[withblur];"
            f"[withblur]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"{subtitle_filter}[v];"
            + _build_audio_filter(background_music_volume)
        )
    else:
        subtitle_filter = (
            f"subtitles={dub_sub_file_escaped}:original_size={TARGET_WIDTH}x{TARGET_HEIGHT}:"
            f"force_style='{subtitles_style},Alignment=2,MarginV={margin_v}'"
        )
        filter_complex = (
            f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
            f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
            f'{subtitle_filter}[v];'
            + _build_audio_filter(background_music_volume)
        )

    cmd = [
        'ffmpeg', '-y', '-i', VIDEO_FILE,
    ]
    if background_music_volume > 0:
        cmd.extend(['-i', background_file])
    cmd.extend(['-i', normalized_dub_audio, '-filter_complex', filter_complex])

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