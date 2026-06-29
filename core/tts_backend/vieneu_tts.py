import os
import yaml
from pathlib import Path

_vieneu_cache = {"model": None}

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

def vieneu_tts(text: str, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cfg   = _load_config().get("vieneu", {})
    voice = cfg.get("voice", "Trọng Hữu")
    tts   = _load_vieneu()
    audio = tts.infer(text=text, voice=voice)
    tts.save(audio, save_path)
    print(f"[VieNeu] Saved → {save_path}")

def custom_tts(text: str, save_path: str) -> None:
    vieneu_tts(text, save_path)