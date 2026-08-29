import pandas as pd
from typing import List, Tuple
import concurrent.futures

from core._3_2_split_meaning import split_sentence, split_sentence_batch
from core.prompts import get_align_prompt, get_align_prompt_batch
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.models import *

console = Console()


# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text)  # force convert

    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)


def align_subs_batch(items: List[dict]) -> dict:
    """Gộp nhiều dòng cần align vào 1 lần gọi ask_gpt duy nhất (thay vì 1 dòng =
    1 request như align_subs cũ). items: list dict {index, src_sub, tr_sub, src_part}.
    Trả về dict {index: (src_parts, tr_parts, tr_remerged)}."""
    align_prompt = get_align_prompt_batch(items)

    def valid_align_batch(response_data):
        for item in items:
            key = str(item["index"])
            if key not in response_data:
                return {"status": "error", "message": f"Missing line {key} in response"}
            if 'align' not in response_data[key]:
                return {"status": "error", "message": f"Missing `align` for line {key}"}
            if len(response_data[key]['align']) < 2:
                return {"status": "error", "message": f"Line {key}: align does not contain more than 1 part"}
        return {"status": "success", "message": "Align completed"}

    parsed = ask_gpt(align_prompt, resp_type='json', valid_def=valid_align_batch, log_title='align_subs_batch')

    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)

    result = {}
    for item in items:
        idx = item["index"]
        align_data = parsed[str(idx)]['align']
        src_parts = item["src_part"].split('\n')
        tr_parts = [entry[f'target_part_{i + 1}'].strip() for i, entry in enumerate(align_data)]
        tr_remerged = joiner.join(tr_parts)

        table = Table(title=f"🔗 Aligned parts (line {idx})")
        table.add_column("Language", style="cyan")
        table.add_column("Parts", style="magenta")
        table.add_row("SRC_LANG", "\n".join(src_parts))
        table.add_row("TARGET_LANG", "\n".join(tr_parts))
        console.print(table)

        result[idx] = (src_parts, tr_parts, tr_remerged)
    return result


def split_align_subs(src_lines: List[str], tr_lines: List[str]):
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_tr_lines = tr_lines.copy()

    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if len(src) > MAX_SUB_LENGTH or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH:
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            console.print(table)

    # Chuẩn bị dữ liệu để gộp batch: split câu gốc theo batch (split_sentence_batch),
    # rồi gộp nhiều dòng vào từng batch ~split_align_batch_size dòng/lần cho
    # bước align - thay vì 2 request riêng (split + align) cho MỖI dòng như cũ.
    batch_size = load_key("split_align_batch_size") or 150
    split_items = [
        {"index": i, "sentence": src_lines[i], "num_parts": 2, "word_limit": 20}
        for i in to_split
    ]
    split_batches = [split_items[b:b + batch_size] for b in range(0, len(split_items), batch_size)]

    prepared = {}

    @except_handler("Error in split_sentence_batch")
    def prepare_batch(batch_items):
        batch_result = split_sentence_batch(batch_items)
        for item in batch_items:
            idx = item["index"]
            if idx in batch_result:
                prepared[idx] = batch_result[idx].strip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        executor.map(prepare_batch, split_batches)

    ready_indices = [i for i in to_split if i in prepared]
    align_batches = [ready_indices[b:b + batch_size] for b in range(0, len(ready_indices), batch_size)]

    @except_handler("Error in align_subs_batch")
    def process_batch(batch_indices):
        items = [
            {"index": i, "src_sub": src_lines[i], "tr_sub": tr_lines[i], "src_part": prepared[i]}
            for i in batch_indices
        ]
        batch_result = align_subs_batch(items)
        for i, (src_parts, tr_parts, tr_remerged) in batch_result.items():
            src_lines[i] = src_parts
            tr_lines[i] = tr_parts
            remerged_tr_lines[i] = tr_remerged

    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        executor.map(process_batch, align_batches)

    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]

    return src_lines, tr_lines, remerged_tr_lines


def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")

    df = pd.read_excel(_4_2_TRANSLATION)
    src = df['Source'].tolist()
    trans = df['Translation'].tolist()

    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]

    for attempt in range(3):  # 多次切割
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        split_src, split_trans, remerged = split_align_subs(src.copy(), trans)

        # 检查是否所有字幕都符合长度要求
        if all(len(src) <= MAX_SUB_LENGTH for src in split_src) and \
                all(calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans):
            break

        # 更新源数据继续下一轮分割
        src, trans = split_src, split_trans

    # 确保二者有相同的长度，防止报错
    if len(src) > len(remerged):
        remerged += [None] * (len(src) - len(remerged))
    elif len(remerged) > len(src):
        src += [None] * (len(remerged) - len(src))

    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_excel(_5_SPLIT_SUB, index=False)
    pd.DataFrame({'Source': src, 'Translation': remerged}).to_excel(_5_REMERGED, index=False)


if __name__ == '__main__':
    split_for_sub_main()