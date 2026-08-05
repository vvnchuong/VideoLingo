import os, sys, re, glob, subprocess, asyncio, requests
from core.utils import *

def sanitize_filename(filename):
    filename = filename.replace('\n', ' ').replace('\r', ' ')
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f#]', '', filename)
    filename = filename.strip('. ')
    filename = re.sub(r'\s+', '_', filename)
    filename = filename[:80]
    return filename if filename else 'video'

def update_ytdlp():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
        if 'yt_dlp' in sys.modules:
            del sys.modules['yt_dlp']
        rprint("[green]yt-dlp updated[/green]")
    except subprocess.CalledProcessError as e:
        rprint(f"[yellow]Warning: Failed to update yt-dlp: {e}[/yellow]")
    from yt_dlp import YoutubeDL
    return YoutubeDL


# ══════════════════════════════════════════════════════════════════════════════
# DOUYIN
# ══════════════════════════════════════════════════════════════════════════════

def _douyin_extract_id(url: str):
    """Lấy aweme_id từ mọi dạng URL Douyin."""
    for pattern in [
        r'modal_id=(\d+)',
        r'/video/(\d+)',
        r'aweme_id=(\d+)',
    ]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None

def _douyin_get_direct_url_attempt(aweme_id: str):
    import json, tempfile, sys
    debug_dir = os.path.join(tempfile.gettempdir(), "douyin_debug")
    os.makedirs(debug_dir, exist_ok=True)
    script = f"""
import json, sys
from playwright.sync_api import sync_playwright

dl_url, title, cookie_str = None, "{aweme_id}", ""
matched_response_seen = False
matched_response_status = None
matched_response_body_snippet = None
response_count = 0

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    def on_response(response):
        global dl_url, title, matched_response_seen, matched_response_status, matched_response_body_snippet, response_count
        response_count += 1
        if "aweme/v1/web/aweme/detail" not in response.url: return
        matched_response_seen = True
        matched_response_status = response.status
        if dl_url: return
        try:
            data = response.json()
            matched_response_body_snippet = json.dumps(data)[:500]
            detail = data.get("aweme_detail", {{}})
            title = detail.get("desc", "{aweme_id}")[:80]
            video = detail.get("video", {{}})
            urls = (video.get("play_addr", {{}}).get("url_list", []) or
                    video.get("play_addr_h264", {{}}).get("url_list", []))
            for u in urls:
                if "douyin.com" in u or "tiktokv.com" in u:
                    dl_url = u; return
            if urls: dl_url = urls[0]
        except Exception as e:
            matched_response_body_snippet = f"LỖI parse JSON: {{e}}"

    page.on("response", on_response)
    page.goto("https://www.douyin.com/video/{aweme_id}", wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(7000)
    cookies = context.cookies()
    cookie_str = "; ".join(f"{{c['name']}}={{c['value']}}" for c in cookies)

    # ── DEBUG: luôn lưu lại bằng chứng để so sánh giữa lần gọi Streamlit vs
    # React - chụp màn hình + url cuối cùng trình duyệt đang đứng (có thể bị
    # Douyin redirect sang trang captcha/chặn khác hẳn URL gốc) + có bắt được
    # đúng response API mong đợi hay không (tách biệt với việc parse ra dl_url
    # có thành công hay không - 2 vấn đề khác nhau).
    debug_info = {{
        "final_page_url": page.url,
        "response_count_total": response_count,
        "matched_response_seen": matched_response_seen,
        "matched_response_status": matched_response_status,
        "matched_response_body_snippet": matched_response_body_snippet,
        "dl_url_extracted": dl_url,
    }}
    with open(r"{debug_dir}\\{aweme_id}_debug.json", "w", encoding="utf-8") as f:
        json.dump(debug_info, f, ensure_ascii=False, indent=2)
    page.screenshot(path=r"{debug_dir}\\{aweme_id}_screenshot.png", full_page=True)

    browser.close()

print(json.dumps({{"dl_url": dl_url, "title": title, "cookie_str": cookie_str}}))
"""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    tmp.write(script)
    tmp.close()

    try:
        result = subprocess.run(
            [sys.executable, tmp.name],
            capture_output=True, text=True, timeout=90
        )
        if result.returncode != 0:
            raise RuntimeError(f"Playwright subprocess lỗi:\n{result.stderr}")
        data = json.loads(result.stdout.strip())
        return data["dl_url"], data["title"], data["cookie_str"]
    finally:
        os.unlink(tmp.name)


def _douyin_get_direct_url(aweme_id: str, max_attempts: int = 3):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            dl_url, title, cookie_str = _douyin_get_direct_url_attempt(aweme_id)
            if dl_url:
                return dl_url, title, cookie_str
            print(f"[Douyin] Lần thử {attempt}/{max_attempts} không lấy được dl_url, thử lại...")
        except Exception as e:
            last_error = e
            print(f"[Douyin] Lần thử {attempt}/{max_attempts} lỗi: {e}, thử lại...")

    if last_error:
        raise last_error
    return None, aweme_id, ""

def _douyin_download_file(dl_url: str, cookie_str: str, save_path: str, aweme_id: str, title: str):
    """Download file mp4 từ direct URL."""
    os.makedirs(save_path, exist_ok=True)
    safe_title = sanitize_filename(title)
    out_path   = os.path.join(save_path, f"{safe_title}.mp4")

    headers = {
        "Referer":         "https://www.douyin.com/",
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":          "video/webm,video/ogg,video/*;*/*;q=0.9",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Range":           "bytes=0-",
        "Cookie":          cookie_str,
    }

    rprint(f"[cyan]Downloading Douyin video: {title[:50]}...[/cyan]")
    with requests.get(dl_url, headers=headers, stream=True, timeout=60, verify=False) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r      {done/total*100:.1f}%...", end="", flush=True)
    print()
    rprint(f"[green]✓ Douyin saved → {out_path}[/green]")
    return out_path

def download_douyin(url: str, save_path: str = 'output'):
    """Entry point cho Douyin — detect ID, Playwright, download."""
    aweme_id = _douyin_extract_id(url)
    if not aweme_id:
        raise ValueError(f"Không tìm được aweme_id từ URL: {url}")

    rprint(f"[cyan]Douyin ID: {aweme_id}[/cyan]")
    dl_url, title, cookie_str = _douyin_get_direct_url(aweme_id)

    if not dl_url:
        raise RuntimeError(f"Không lấy được download URL cho video {aweme_id}")

    return _douyin_download_file(dl_url, cookie_str, save_path, aweme_id, title)

def download_video_ytdlp(url, save_path='output', resolution='1080'):
    # Douyin → dùng Playwright
    if "douyin.com" in url:
        return download_douyin(url, save_path)

    # Các platform khác → yt-dlp như cũ
    os.makedirs(save_path, exist_ok=True)
    ydl_opts = {
        'format': ('bestvideo+bestaudio/best' if resolution == 'best'
                   else f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]'),
        'outtmpl': f'{save_path}/%(title)s.%(ext)s',
        'noplaylist': True,
        'writethumbnail': True,
        'merge_output_format': 'mp4',
        'postprocessors': [{'key': 'FFmpegThumbnailsConvertor', 'format': 'jpg'}],
    }

    cookies_path = load_key("youtube.cookies_path")
    if os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = str(cookies_path)

    YoutubeDL = update_ytdlp()
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for file in os.listdir(save_path):
        if os.path.isfile(os.path.join(save_path, file)):
            filename, ext = os.path.splitext(file)
            new_filename  = sanitize_filename(filename)
            if new_filename != filename:
                os.rename(
                    os.path.join(save_path, file),
                    os.path.join(save_path, new_filename + ext),
                )


def find_video_files(save_path='output'):
    video_files = [
        file for file in glob.glob(save_path + "/*")
        if os.path.splitext(file)[1][1:].lower() in load_key("allowed_video_formats")
    ]
    if sys.platform.startswith('win'):
        video_files = [file.replace("\\", "/") for file in video_files]
    video_files = [file for file in video_files if not file.startswith("output/output")]
    if len(video_files) != 1:
        raise ValueError(f"Number of videos found {len(video_files)} is not unique. Please check.")
    return video_files[0]


if __name__ == '__main__':
    url        = input('Please enter the URL of the video you want to download: ')
    resolution = input('Please enter the desired resolution (360/480/720/1080, default 1080): ')
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")