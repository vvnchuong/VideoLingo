"""
Wrapper CLI để Java gọi tải video từ link, dùng lại ĐÚNG logic có sẵn trong
core/_1_ytdlp.py (bao gồm nhánh Douyin riêng qua Playwright) - không viết lại
logic tải video ở phía Java để tránh trùng công và dễ sai (đặc biệt là Douyin,
vốn cần Playwright + cookie mà yt-dlp thường không tải được).

Dùng:
    python download_video.py --url <URL> --output <đường dẫn file .mp4 muốn lưu>
"""
import argparse
import os
import shutil
import sys
import tempfile

from core._1_ytdlp import download_video_ytdlp, find_video_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, help="Đường dẫn file .mp4 muốn lưu kết quả")
    args = parser.parse_args()

    # Tải vào thư mục tạm riêng, tránh đụng tới output/ đang dùng cho job khác
    tmp_dir = tempfile.mkdtemp(prefix="url_download_")

    try:
        download_video_ytdlp(args.url, save_path=tmp_dir)
        downloaded_path = find_video_files(save_path=tmp_dir)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        shutil.move(downloaded_path, args.output)

        print(f"RESULT_PATH={args.output}", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()