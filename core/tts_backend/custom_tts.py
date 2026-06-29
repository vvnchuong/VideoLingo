# from .vieneu_tts import vieneu_tts
#
# def custom_tts(text: str, save_path: str) -> None:
#     vieneu_tts(text, save_path)

# from .omnivoice_tts import omnivoice_tts_with_ref
#
# def custom_tts(text: str, save_path: str) -> None:
#     omnivoice_tts_with_ref(text, save_path)

from .omnivoice_tts import omnivoice_tts

def custom_tts(text: str, save_path: str) -> None:
    omnivoice_tts(text, save_path)