import os
import subprocess

import cv2


def get_video_size(video_path: str):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def is_landscape(video_path: str) -> bool:
    w, h = get_video_size(video_path)
    return w > h


def convert_to_short(video_path: str, target_w: int = 1080, target_h: int = 1920,
                      blur_sigma: int = 20) -> str:
    """Chuyển video_path (khổ ngang) sang khổ dọc target_w x target_h, nền blur từ chính video đó.
    Ghi đè lên đúng video_path (video gốc bị thay thế bởi bản 9:16 mới)."""
    tmp_out = video_path + ".short_tmp.mp4"

    filter_complex = (
        f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},gblur=sigma={blur_sigma}[bg];"
        f"[0:v]scale={target_w}:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
    )

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
        tmp_out,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    os.replace(tmp_out, video_path)
    return video_path