import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import subprocess
import tempfile
from pathlib import Path

from core.tts_backend.capcut_tts_wrapper import VOICE_PRESETS
from core.tts_backend.capcut_tts import synthesize

MAX_CHARS_PER_REQUEST = 8000  # CapCut giới hạn ~10.000, chừa dư an toàn


def split_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_REQUEST) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    remaining = text
    sentence_enders = (".", "!", "?", "…", "\n")

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut_at = max(window.rfind(ch) for ch in sentence_enders)

        if cut_at == -1:
            cut_at = max_chars - 1

        chunks.append(remaining[: cut_at + 1].strip())
        remaining = remaining[cut_at + 1:].strip()

    if remaining:
        chunks.append(remaining)

    return [c for c in chunks if c]


def concat_mp3s(paths: list[str], final_output: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for p in paths:
            escaped = p.replace("'", r"'\''")
            f.write(f"file '{escaped}'\n")
        list_file = f.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file, "-c", "copy", final_output,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg concat lỗi: {e.stderr}") from e
    finally:
        Path(list_file).unlink(missing_ok=True)


def apply_speed(input_path: str, output_path: str, rate: float) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-filter:a", f"atempo={rate}", output_path],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg atempo lỗi: {e.stderr}") from e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=False,
                         help="Văn bản trực tiếp - CHỈ nên dùng cho text ngắn/test tay. "
                              "Với text dài hoặc có ký tự đặc biệt, dùng --text-file thay thế "
                              "vì truyền qua command-line dễ vỡ trên Windows.")
    parser.add_argument("--text-file", required=False,
                         help="Đường dẫn file .txt chứa văn bản (UTF-8) - cách AN TOÀN để "
                              "truyền text dài, tránh lỗi escape/giới hạn độ dài dòng lệnh "
                              "của Windows khi text dài hoặc có ký tự đặc biệt.")
    parser.add_argument("--voice", required=True, help="Tên preset trong VOICE_PRESETS")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rate", default="1.0",
                         help="Tốc độ đọc 0.5-2.0 - xử lý bằng ffmpeg atempo SAU khi có "
                              "audio gốc, không dùng SSML rate của CapCut (không hoạt động).")
    parser.add_argument("--list-voices", action="store_true",
                         help="Chỉ in danh sách voice ra JSON rồi thoát, không tạo audio")
    args = parser.parse_args()

    if args.list_voices:
        import json
        print(json.dumps(sorted(VOICE_PRESETS.keys())))
        return

    if not args.text and not args.text_file:
        print("LỖI: Phải truyền --text hoặc --text-file.", file=sys.stderr)
        sys.exit(1)

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = args.text

    if args.voice not in VOICE_PRESETS:
        available = ", ".join(sorted(VOICE_PRESETS.keys()))
        print(f"LỖI: Không tìm thấy giọng '{args.voice}'. Các giọng có sẵn: {available}",
              file=sys.stderr)
        sys.exit(1)

    voice, resource_id = VOICE_PRESETS[args.voice]

    try:
        chunks = split_into_chunks(text)
        if not chunks:
            raise RuntimeError("Văn bản rỗng sau khi strip.")

        out_path = Path(args.output)
        rate = float(args.rate)
        needs_speed_change = abs(rate - 1.0) > 0.001
        raw_output = (out_path.parent / f".raw_{out_path.name}") if needs_speed_change else out_path

        if len(chunks) == 1:
            synthesize(chunks[0], voice=voice, resource_id=resource_id,
                       out_path=str(raw_output))
        else:
            batch_dir = out_path.parent / f".tts_batch_{out_path.stem}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            chunk_paths = []
            try:
                for i, chunk in enumerate(chunks):
                    chunk_path = batch_dir / f"chunk_{i:03d}.mp3"
                    synthesize(chunk, voice=voice, resource_id=resource_id,
                               out_path=str(chunk_path))
                    chunk_paths.append(str(chunk_path))

                concat_mp3s(chunk_paths, str(raw_output))
            finally:
                for p in chunk_paths:
                    Path(p).unlink(missing_ok=True)
                try:
                    batch_dir.rmdir()
                except OSError:
                    pass

        if needs_speed_change:
            try:
                apply_speed(str(raw_output), str(out_path), rate)
            finally:
                Path(raw_output).unlink(missing_ok=True)

        print("OK")
    except Exception as e:
        print(f"LỖI: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()