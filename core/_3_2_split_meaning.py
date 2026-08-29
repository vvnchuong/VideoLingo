import concurrent.futures
from difflib import SequenceMatcher
import math
from core.prompts import get_split_prompt, get_split_prompt_batch
from core.spacy_utils.load_nlp_model import init_nlp
from core.utils import *
from rich.console import Console
from rich.table import Table
from core.utils.models import _3_1_SPLIT_BY_NLP, _3_2_SPLIT_BY_MEANING

console = Console()


def tokenize_sentence(sentence, nlp):
    doc = nlp(sentence)
    return [token.text for token in doc]


def find_split_positions(original, modified):
    split_positions = []
    parts = modified.split('[br]')
    start = 0
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)

    for i in range(len(parts) - 1):
        max_similarity = 0
        best_split = None

        for j in range(start, len(original)):
            original_left = original[start:j]
            modified_left = joiner.join(parts[i].split())

            left_similarity = SequenceMatcher(None, original_left, modified_left).ratio()

            if left_similarity > max_similarity:
                max_similarity = left_similarity
                best_split = j

        if max_similarity < 0.9:
            console.print(f"[yellow]Warning: low similarity found at the best split point: {max_similarity}[/yellow]")
        if best_split is not None:
            split_positions.append(best_split)
            start = best_split
        else:
            console.print(f"[yellow]Warning: Unable to find a suitable split point for the {i + 1}th part.[/yellow]")

    return split_positions


def split_sentence(sentence, num_parts, word_limit=20, index=-1, retry_attempt=0):
    """Split a long sentence using GPT and return the result as a string."""
    split_prompt = get_split_prompt(sentence, num_parts, word_limit)

    def valid_split(response_data):
        choice = response_data["choice"]
        if f'split{choice}' not in response_data:
            return {"status": "error", "message": "Missing required key: `split`"}
        if "[br]" not in response_data[f"split{choice}"]:
            return {"status": "error", "message": "Split failed, no [br] found"}
        return {"status": "success", "message": "Split completed"}

    response_data = ask_gpt(split_prompt + " " * retry_attempt, resp_type='json', valid_def=valid_split,
                            log_title='split_by_meaning')
    choice = response_data["choice"]
    best_split = response_data[f"split{choice}"]
    split_points = find_split_positions(sentence, best_split)
    # split the sentence based on the split points
    for i, split_point in enumerate(split_points):
        if i == 0:
            best_split = sentence[:split_point] + '\n' + sentence[split_point:]
        else:
            parts = best_split.split('\n')
            last_part = parts[-1]
            parts[-1] = last_part[:split_point - split_points[i - 1]] + '\n' + last_part[
                                                                               split_point - split_points[i - 1]:]
            best_split = '\n'.join(parts)
    if index != -1:
        console.print(f'[green]✅ Sentence {index} has been successfully split[/green]')
    table = Table(title="")
    table.add_column("Type", style="cyan")
    table.add_column("Sentence")
    table.add_row("Original", sentence, style="yellow")
    table.add_row("Split", best_split.replace('\n', ' ||'), style="yellow")
    console.print(table)

    return best_split


def split_sentence_batch(items, retry_attempt=0):
    """Gộp nhiều câu cần split vào 1 lần gọi ask_gpt duy nhất (thay vì 1 câu =
    1 request như split_sentence cũ). items: list dict {index, sentence,
    num_parts, word_limit}. Trả về dict {index: best_split_string}."""
    split_prompt = get_split_prompt_batch(items)

    def valid_split_batch(response_data):
        for item in items:
            key = str(item["index"])
            if key not in response_data:
                return {"status": "error", "message": f"Missing line {key} in response"}
            entry = response_data[key]
            if "choice" not in entry:
                return {"status": "error", "message": f"Line {key}: missing `choice`"}
            choice = entry["choice"]
            if f'split{choice}' not in entry:
                return {"status": "error", "message": f"Line {key}: missing `split{choice}`"}
            if "[br]" not in entry[f'split{choice}']:
                return {"status": "error", "message": f"Line {key}: split failed, no [br] found"}
        return {"status": "success", "message": "Split completed"}

    response_data = ask_gpt(split_prompt + " " * retry_attempt, resp_type='json',
                            valid_def=valid_split_batch, log_title='split_by_meaning_batch')

    result = {}
    for item in items:
        idx = item["index"]
        sentence = item["sentence"]
        entry = response_data[str(idx)]
        choice = entry["choice"]
        best_split = entry[f'split{choice}']
        split_points = find_split_positions(sentence, best_split)
        for i, split_point in enumerate(split_points):
            if i == 0:
                best_split = sentence[:split_point] + '\n' + sentence[split_point:]
            else:
                parts = best_split.split('\n')
                last_part = parts[-1]
                parts[-1] = last_part[:split_point - split_points[i - 1]] + '\n' + last_part[
                                                                                   split_point - split_points[i - 1]:]
                best_split = '\n'.join(parts)

        console.print(f'[green]✅ Sentence {idx} has been successfully split[/green]')
        table = Table(title="")
        table.add_column("Type", style="cyan")
        table.add_column("Sentence")
        table.add_row("Original", sentence, style="yellow")
        table.add_row("Split", best_split.replace('\n', ' ||'), style="yellow")
        console.print(table)

        result[idx] = best_split
    return result


def parallel_split_sentences(sentences, max_length, max_workers, nlp, retry_attempt=0):
    """Split sentences in batches (gộp nhiều câu/lần gọi LLM thay vì 1 câu/request)."""
    new_sentences = [None] * len(sentences)

    to_split = []
    for index, sentence in enumerate(sentences):
        tokens = tokenize_sentence(sentence, nlp)
        num_parts = math.ceil(len(tokens) / max_length)
        if len(tokens) > max_length:
            to_split.append({"index": index, "sentence": sentence, "num_parts": num_parts, "word_limit": max_length})
        else:
            new_sentences[index] = [sentence]

    if not to_split:
        return [sentence for sublist in new_sentences for sentence in sublist]

    batch_size = load_key("split_align_batch_size") or 150
    batches = [to_split[b:b + batch_size] for b in range(0, len(to_split), batch_size)]

    @except_handler("Error in split_sentence_batch")
    def process_batch(batch_items):
        batch_result = split_sentence_batch(batch_items, retry_attempt=retry_attempt)
        for item in batch_items:
            idx = item["index"]
            split_result = batch_result.get(idx)
            if split_result:
                split_lines = split_result.strip().split('\n')
                new_sentences[idx] = [line.strip() for line in split_lines]
            else:
                new_sentences[idx] = [item["sentence"]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_batch, batches)

    # An toàn: nếu batch nào lỗi (except_handler nuốt exception) mà chưa gán
    # kết quả, fallback về câu gốc để không bao giờ để None lọt qua.
    for item in to_split:
        if new_sentences[item["index"]] is None:
            new_sentences[item["index"]] = [item["sentence"]]

    return [sentence for sublist in new_sentences for sentence in sublist]


@check_file_exists(_3_2_SPLIT_BY_MEANING)
def split_sentences_by_meaning():
    """The main function to split sentences by meaning."""
    # read input sentences
    with open(_3_1_SPLIT_BY_NLP, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]

    nlp = init_nlp()
    # 🔄 process sentences multiple times to ensure all are split
    for retry_attempt in range(3):
        sentences = parallel_split_sentences(sentences, max_length=load_key("max_split_length"),
                                             max_workers=load_key("max_workers"), nlp=nlp, retry_attempt=retry_attempt)

    # 💾 save results
    with open(_3_2_SPLIT_BY_MEANING, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences))
    console.print('[green]✅ All sentences have been successfully split![/green]')


if __name__ == '__main__':
    # print(split_sentence('Which makes no sense to the... average guy who always pushes the character creation slider all the way to the right.', 2, 22))
    split_sentences_by_meaning()