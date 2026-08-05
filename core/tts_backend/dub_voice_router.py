import os

from core.tts_backend.capcut_tts_wrapper import VOICE_PRESETS, capcut_custom_tts_by_name, capcut_custom_tts

VIENEU_VOICES = {
    "Ngọc Lan", "Gia Bảo", "Thái Sơn", "Đức Trí", "Mỹ Duyên",
    "Trúc Ly", "Xuân Vĩnh", "Trọng Hữu", "Bình An", "Ngọc Linh",
}


def get_selected_voice_id() -> str:
    return os.environ.get("DUB_VOICE_ID", "").strip()


def backend_of(voice_id: str) -> str:
    if not voice_id:
        return "capcut"
    if voice_id in VOICE_PRESETS:
        return "capcut"
    if voice_id in VIENEU_VOICES:
        return "vieneu"
    return "capcut"


def is_dynamic_routing_active() -> bool:
    return bool(get_selected_voice_id())


def generate_dub_audio(text: str, save_path: str) -> None:
    voice_id = get_selected_voice_id()
    backend = backend_of(voice_id)

    if backend == "vieneu":
        from core.tts_backend.vieneu_tts import vieneu_tts
        vieneu_tts(text, save_path, voice=voice_id)  # ghi thẳng .wav, không cần convert
        return

    import tempfile
    from pydub import AudioSegment

    tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    try:
        if voice_id in VOICE_PRESETS:
            capcut_custom_tts_by_name(text, tmp_mp3, preset_name=voice_id)
        else:
            capcut_custom_tts(text, tmp_mp3)

        AudioSegment.from_mp3(tmp_mp3).export(save_path, format="wav")
    finally:
        if os.path.exists(tmp_mp3):
            os.remove(tmp_mp3)

    # thêm backend mới tại đây khi cần, ví dụ:
    # elif backend == "f5tts":
    #     from core.tts_backend._302_f5tts import f5_tts_for_videolingo
    #     f5_tts_for_videolingo(text, save_path, ...)
    # elif backend == "omnivoice_clone":
    #     from core.tts_backend.omnivoice_tts import omnivoice_tts_with_ref
    #     omnivoice_tts_with_ref(text, save_path, reference_wav=...)