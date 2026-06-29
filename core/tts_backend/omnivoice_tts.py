"""
OmniVoice custom TTS cho VideoLingo
Nhét vào core/all_tts_functions/custom_tts.py

Ưu điểm so với VieNeu:
- Có speed param thật → sync tốt hơn, ít cần atempo
- GPU mode với RTX 4050 → nhanh hơn CPU OmniVoice cũ
- Fine-tune tiếng Việt: splendor1811/omnivoice-vietnamese

Cài đặt:
  pip install omnivoice

Lần đầu chạy sẽ tự download model ~vài GB về HF_HOME
"""

import os
import random
import time

import numpy as np
import soundfile as sf
import torch

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_ID = "splendor1811/omnivoice-vietnamese"  # VI fine-tune
# MODEL_ID = "k2-fsa/OmniVoice"                # base multilingual

DEFAULT_VOICE_INSTRUCT = "female, young adult, moderate pitch"
DEFAULT_SPEED = 1.0
DEFAULT_SEED = 42

# ── Seed helper ──────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ── Model singleton ──────────────────────────────────────────────────────────

_model = None

def _get_model():
    global _model
    if _model is None:
        from omnivoice import OmniVoice
        print(f"[OmniVoice] Loading model: {MODEL_ID} on cuda...")
        _model = OmniVoice.from_pretrained(
            MODEL_ID,
            device_map="cuda:0",
            dtype=torch.float16,
            load_asr=False,
        )
        print("[OmniVoice] Model loaded ✓")
    return _model


# ── Main function VideoLingo gọi ─────────────────────────────────────────────

def omnivoice_tts(text: str, output_path: str, speed: float = DEFAULT_SPEED, seed: int = DEFAULT_SEED) -> str:
    """
    Generate TTS với OmniVoice.

    Args:
        text: văn bản tiếng Việt cần đọc
        output_path: đường dẫn file output (.wav)
        speed: tốc độ nói (0.8 chậm ~ 1.3 nhanh), default 1.0
        seed: random seed để output ổn định, default 42

    Returns:
        output_path nếu thành công
    """
    model = _get_model()

    # Set seed trước mỗi lần generate — không pass vào model.generate()
    # vì OmniVoice không có param seed trong API
    _set_seed(seed)

    t0 = time.time()

    # Thử noise_scale=0 nếu model hỗ trợ (VITS-based), fallback nếu không
    try:
        audio = model.generate(
            text=text,
            instruct=DEFAULT_VOICE_INSTRUCT,
            speed=speed,
            noise_scale=0.0,
            noise_scale_w=0.0,
        )
    except TypeError:
        # Model không có noise_scale param → dùng generate thường
        audio = model.generate(
            text=text,
            instruct=DEFAULT_VOICE_INSTRUCT,
            speed=speed,
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, audio[0], samplerate=24000)

    elapsed = time.time() - t0
    audio_duration = len(audio[0]) / 24000
    rtf = elapsed / audio_duration if audio_duration > 0 else 0
    print(f"[OmniVoice] '{text[:40]}...' → {audio_duration:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f})")

    return output_path


# ── Ref audio variant ────────────────────────────────────────────────────────

REF_AUDIO = os.path.join(os.path.dirname(__file__), "reference.wav")

def omnivoice_tts_with_ref(text: str, save_path: str, seed: int = DEFAULT_SEED) -> None:
    import soundfile as sf_reader
    model = _get_model()

    _set_seed(seed)

    ref_waveform, ref_sr = sf_reader.read(REF_AUDIO, dtype='float32')

    audio = model.generate(
        text=text,
        ref_audio=ref_waveform,
        ref_sr=ref_sr,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    sf.write(save_path, audio[0], samplerate=24000)


# ── Tích hợp vào VideoLingo custom_tts.py ────────────────────────────────────
#
# Trong file core/all_tts_functions/custom_tts.py của VideoLingo, thêm:
#
#   from omnivoice_tts import omnivoice_tts
#
#   def custom_tts(text, output_path, **kwargs):
#       speed = kwargs.get("speed", 1.0)
#       return omnivoice_tts(text, output_path, speed=speed)
#
# Trong config.yaml set:
#   tts_method: "custom"
#
# ─────────────────────────────────────────────────────────────────────────────


# ── Test standalone ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_sentences = [
        "Xin chào, đây là bài kiểm tra giọng nói tiếng Việt.",
        "Hôm nay thời tiết Đà Lạt rất đẹp, trời mát mẻ và có gió nhẹ.",
        "Pipeline xử lý video tự động đang chạy thử nghiệm.",
    ]

    os.makedirs("test_output", exist_ok=True)

    print("=== Test OmniVoice TTS (seed=42) ===\n")

    for i, text in enumerate(test_sentences):
        out = f"test_output/test_{i+1}.wav"
        print(f"[{i+1}] {text}")
        omnivoice_tts(text, out, speed=1.0)
        print(f"     → Saved: {out}\n")

    # Test reproducibility — generate 2 lần cùng text, output phải giống nhau
    print("=== Test reproducibility ===\n")
    text = "Kiểm tra tính ổn định của seed."
    omnivoice_tts(text, "test_output/repro_1.wav")
    omnivoice_tts(text, "test_output/repro_2.wav")
    print("Nếu repro_1.wav và repro_2.wav nghe giống nhau → seed hoạt động ✓\n")

    # Test speed control
    print("=== Test speed control ===\n")
    text = "Đây là câu kiểm tra tốc độ nói, thử ở các mức khác nhau."
    for spd in [0.9, 1.0, 1.1, 1.2]:
        out = f"test_output/speed_{spd}.wav"
        print(f"Speed {spd}x:")
        omnivoice_tts(text, out, speed=spd)
        print(f"  → {out}\n")

    print("Done! Check test_output/ folder.")