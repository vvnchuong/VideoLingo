import os
import numpy as np
import pandas as pd
import subprocess
from pydub import AudioSegment
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from core.utils import *
from core.utils.models import *

console = Console()

DUB_VOCAL_FILE = 'output/dub.mp3'

DUB_SUB_FILE = 'output/dub.srt'
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"


def load_and_flatten_data(excel_file):
    """Load and flatten Excel data"""
    df = pd.read_excel(excel_file)
    lines = [eval(line) if isinstance(line, str) else line for line in df['lines'].tolist()]
    lines = [item for sublist in lines for item in sublist]

    new_sub_times = [eval(time) if isinstance(time, str) else time for time in df['new_sub_times'].tolist()]
    new_sub_times = [item for sublist in new_sub_times for item in sublist]

    return df, lines, new_sub_times


def get_audio_files(df):
    """Generate a list of audio file paths"""
    audios = []
    for index, row in df.iterrows():
        number = row['number']
        line_count = len(eval(row['lines']) if isinstance(row['lines'], str) else row['lines'])
        for line_index in range(line_count):
            temp_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
            audios.append(temp_file)
    return audios


def process_audio_segment(audio_file):
    """Process a single audio segment with MP3 compression"""
    temp_file = f"{audio_file}_temp.mp3"
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', audio_file,
        '-ar', '16000',
        '-ac', '1',
        '-b:a', '64k',
        temp_file
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio_segment = AudioSegment.from_mp3(temp_file)
    os.remove(temp_file)
    return audio_segment


def merge_audio_segments(audios, new_sub_times, sample_rate):
    """
    Ghép các file audio lại, đồng thời tính ra timeline THẬT (actual_sub_times)
    dựa trên độ dài audio thật đã ghép - KHÔNG dùng new_sub_times gốc để suy ra
    vị trí tiếp theo (vì audio thật sau khi chỉnh tốc độ luôn lệch vài chục ms
    so với dự tính, sai số này CỘNG DỒN qua từng câu, khiến sub và audio trôi
    lệch dần theo thời gian - càng về cuối video càng lệch nhiều).
    Trả về (merged_audio, actual_sub_times) để dùng CHUNG 1 timeline duy nhất
    khi ghi dub.srt, đảm bảo sub luôn khớp audio thật.
    """
    merged_audio = AudioSegment.silent(duration=0, frame_rate=sample_rate)
    actual_sub_times = []
    real_prev_end = 0.0  # vị trí THẬT tính bằng giây, sau khi ghép audio trước đó

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TaskProgressColumn()) as progress:
        merge_task = progress.add_task("🎵 Merging audio segments...", total=len(audios))

        for i, (audio_file, time_range) in enumerate(zip(audios, new_sub_times)):
            start_time, end_time = time_range
            if not os.path.exists(audio_file):
                console.print(
                    f"[bold yellow]⚠️  Warning: File {audio_file} does not exist, skipping audio (giữ nguyên thời lượng sub dự tính)...[/bold yellow]")
                # Audio thiếu -> vẫn giữ ĐÚNG thời lượng dự tính ban đầu cho sub
                # (chỉ mỗi phần audio bị câm/im lặng), KHÔNG co khung sub về 0
                # giây - nếu không sub sẽ biến mất khỏi màn hình dù data vẫn có.
                if i > 0:
                    target_prev_end = new_sub_times[i - 1][1]
                    planned_gap = max(start_time - target_prev_end, 0.0)
                else:
                    planned_gap = max(start_time, 0.0)
                if planned_gap > 0:
                    merged_audio += AudioSegment.silent(duration=int(planned_gap * 1000), frame_rate=sample_rate)
                    real_prev_end += planned_gap
                planned_dur = max(end_time - start_time, 0.0)
                if planned_dur > 0:
                    silence = AudioSegment.silent(duration=int(planned_dur * 1000), frame_rate=sample_rate)
                    merged_audio += silence
                real_start = real_prev_end
                real_end = real_start + planned_dur
                actual_sub_times.append([real_start, real_end])
                real_prev_end = real_end
                progress.advance(merge_task)
                continue

            audio_segment = process_audio_segment(audio_file)

            # Khoảng lặng dự tính giữa 2 câu (dựa theo target ban đầu), nhưng
            # điểm neo là VỊ TRÍ THẬT (real_prev_end), không phải target trước đó
            if i > 0:
                target_prev_end = new_sub_times[i - 1][1]
                planned_gap = max(start_time - target_prev_end, 0.0)
            else:
                planned_gap = max(start_time, 0.0)

            if planned_gap > 0:
                silence = AudioSegment.silent(duration=int(planned_gap * 1000), frame_rate=sample_rate)
                merged_audio += silence
                real_prev_end += planned_gap

            real_start = real_prev_end
            merged_audio += audio_segment
            real_actual_dur = len(audio_segment) / 1000.0  # độ dài audio THẬT (giây)
            real_end = real_start + real_actual_dur

            actual_sub_times.append([real_start, real_end])
            real_prev_end = real_end
            progress.advance(merge_task)

    return merged_audio, actual_sub_times


def create_srt_subtitle(sub_times=None):
    df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)
    times_to_use = sub_times if sub_times is not None else new_sub_times

    # An toàn cuối cùng: dù nguyên nhân gốc (sai số làm tròn khi trim audio,
    # tolerance tính lệch...) là gì, KHÔNG BAO GIỜ để 1 cue overlap cue kế tiếp -
    # nếu end_time của cue này > start_time của cue sau, cắt ngắn lại cho khớp.
    for i in range(len(times_to_use) - 1):
        this_end = times_to_use[i][1]
        next_start = times_to_use[i + 1][0]
        if this_end > next_start:
            times_to_use[i][1] = next_start

    with open(DUB_SUB_FILE, 'w', encoding='utf-8') as f:
        for i, ((start_time, end_time), line) in enumerate(zip(times_to_use, lines), 1):
            start_str = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d},{int((start_time * 1000) % 1000):03d}"
            end_str = f"{int(end_time // 3600):02d}:{int((end_time % 3600) // 60):02d}:{int(end_time % 60):02d},{int((end_time * 1000) % 1000):03d}"

            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{line}\n\n")

    rprint(f"[bold green]✅ Subtitle file created: {DUB_SUB_FILE}[/bold green]")


def merge_full_audio():
    """Main function: Process the complete audio merging process"""
    console.print("\n[bold cyan]🎬 Starting audio merging process...[/bold cyan]")

    with console.status("[bold cyan]📊 Loading data from Excel...[/bold cyan]"):
        df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)
    console.print("[bold green]✅ Data loaded successfully[/bold green]")

    with console.status("[bold cyan]🔍 Getting audio file list...[/bold cyan]"):
        audios = get_audio_files(df)
    console.print(f"[bold green]✅ Found {len(audios)} audio segments[/bold green]")

    if not os.path.exists(audios[0]):
        console.print(f"[bold red]❌ Error: First audio file {audios[0]} does not exist![/bold red]")
        return

    sample_rate = 16000
    console.print(f"[bold green]✅ Sample rate: {sample_rate}Hz[/bold green]")

    # Ghép audio TRƯỚC để lấy được timeline THẬT (actual_sub_times) - tránh
    # trôi lệch dần giữa sub và audio qua từng câu (xem giải thích trong
    # merge_audio_segments). Ghi sub SAU, dùng đúng timeline thật này.
    console.print("[bold cyan]🔄 Starting audio merge process...[/bold cyan]")
    merged_audio, actual_sub_times = merge_audio_segments(audios, new_sub_times, sample_rate)

    with console.status("[bold cyan]📝 Generating subtitle file (theo timeline audio thật)...[/bold cyan]"):
        create_srt_subtitle(sub_times=actual_sub_times)

    with console.status("[bold cyan]💾 Exporting final audio file...[/bold cyan]"):
        merged_audio = merged_audio.set_frame_rate(16000).set_channels(1)
        merged_audio.export(DUB_VOCAL_FILE, format="mp3", parameters=["-b:a", "64k"])
    console.print(f"[bold green]✅ Audio file successfully merged![/bold green]")
    console.print(f"[bold green]📁 Output file: {DUB_VOCAL_FILE}[/bold green]")


if __name__ == "__main__":
    merge_full_audio()