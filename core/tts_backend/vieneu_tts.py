import os
import re
import yaml
from pathlib import Path

_vieneu_cache = {"model": None}

# ── Config gộp câu ngắn ───────────────────────────────────────────────────────
MIN_WORDS_THRESHOLD = 4
PAD_PREFIX = "Này, tôi muốn nói là"  # cụm đệm trung tính, không mang nghĩa đặc biệt
PAD_TRIM_BUFFER_SEC = 0.15           # lùi lại 1 chút để tránh cắt lẹm từ đầu câu thật


def _load_config():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_vieneu():
    if _vieneu_cache["model"] is not None:
        return _vieneu_cache["model"]

    cfg = _load_config().get("vieneu", {})
    hf_home = cfg.get("hf_home", "")
    mode    = cfg.get("mode", "v3turbo")

    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)

    from vieneu import Vieneu
    print(f"[VieNeu] Loading model (mode={mode})...")
    tts = Vieneu(mode=mode) if mode else Vieneu()
    _vieneu_cache["model"] = tts
    print("[VieNeu] Model loaded.")
    return tts


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _extract_wave_and_sr(audio):
    """
    Trích xuất (waveform, sample_rate) từ object audio trả về bởi tts.infer().

    CHÚ Ý: đoán cấu trúc phổ biến (attr .samples/.sample_rate, hoặc tuple,
    hoặc numpy array thô mặc định 24000Hz). Nếu API thật của thư viện `vieneu`
    khác, cần chỉnh lại hàm này — in thử `type(audio)` và `dir(audio)` để kiểm tra.
    """
    import numpy as np

    if hasattr(audio, "samples") and hasattr(audio, "sample_rate"):
        return np.asarray(audio.samples), audio.sample_rate
    if isinstance(audio, (tuple, list)) and len(audio) == 2:
        return np.asarray(audio[0]), audio[1]
    # fallback: giả định là numpy array thô, 24kHz
    return np.asarray(audio), 24000


def vieneu_tts(text: str, save_path: str, voice: str = None) -> None:
    """
    voice=None -> lấy từ config.yaml (hành vi CŨ, dùng cho dub - KHÔNG đổi gì
    để không ảnh hưởng pipeline dub đang chạy ổn định).
    voice="Tên giọng" -> override, dùng cho tính năng TTS đứng lẻ (user tự chọn
    1 trong 10 giọng preset qua UI).
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cfg   = _load_config().get("vieneu", {})
    voice = voice or cfg.get("voice", "Trọng Hữu")
    tts   = _load_vieneu()

    if _word_count(text) < MIN_WORDS_THRESHOLD:
        # Câu quá ngắn/đơn -> đệm thêm cụm mở đầu để model có đủ "nhịp nói" ổn định
        padded_text = f"{PAD_PREFIX} {text}"
        audio = tts.infer(text=padded_text, voice=voice)

        wav, sr = _extract_wave_and_sr(audio)

        total_chars  = len(padded_text)
        prefix_chars = len(PAD_PREFIX) + 1  # +1 cho khoảng trắng nối
        cut_ratio    = prefix_chars / total_chars

        cut_sample     = int(len(wav) * cut_ratio)
        buffer_samples = int(sr * PAD_TRIM_BUFFER_SEC)
        cut_sample     = max(0, cut_sample - buffer_samples)

        trimmed_wav = wav[cut_sample:]

        import soundfile as sf
        sf.write(save_path, trimmed_wav, samplerate=sr)
        print(f"[VieNeu] (padded, {_word_count(text)} từ) Saved → {save_path}")
    else:
        audio = tts.infer(text=text, voice=voice)
        tts.save(audio, save_path)
        print(f"[VieNeu] Saved → {save_path}")


def custom_tts(text: str, save_path: str) -> None:
    vieneu_tts(text, save_path)


# ── Test standalone ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        "Ừ.",
        "Không.",
        "Được rồi.",
        "Đây là một câu dài hơn để so sánh, xem giọng đọc có tự nhiên và ổn định hay không.",
    ]

    os.makedirs("test_output_vieneu", exist_ok=True)
    for i, t in enumerate(test_cases):
        out = f"test_output_vieneu/test_{i+1}.wav"
        print(f"[{i+1}] '{t}'")
        vieneu_tts(t, out)  