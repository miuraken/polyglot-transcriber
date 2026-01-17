import whisper
import os
import numpy as np
import torch
import librosa
import soundfile as sf
import webrtcvad
from collections import deque
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import argparse

# VAD aggressiveness setting (0=least aggressive, 3=most aggressive)

VAD_AGGRESSIVENESS = 1

SAMPLE_RATE = 16000  # webrtcvad requires 16kHz

FRAME_LENGTH_MS = 30  # webrtcvad requires 10, 20, or 30ms frame lengths


def read_wave(path):
    """Reads a .wav file.

    Takes the path, returns (sample_rate, audio_data).

    """

    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    # webrtcvad requires 16bit PCM

    audio_int16 = (audio * 32767).astype(np.int16)

    return sr, audio_int16.tobytes()


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes the path, audio data, and sample rate.

    """

    sf.write(path, audio, sample_rate)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):

        self.bytes = bytes

        self.timestamp = timestamp

        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data."""

    n = int(
        sample_rate * (frame_duration_ms / 1000.0) * 2
    )  # 2 bytes per sample for int16

    offset = 0

    timestamp = 0.0

    duration = (n / 2) / sample_rate

    while offset + n <= len(audio):

        yield Frame(audio[offset : offset + n], timestamp, duration)

        offset += n

        timestamp += duration


def vad_segment_generator(
    wave_file, aggressiveness, min_speech_duration_ms=300, min_silence_duration_ms=600
):
    """Generates segments of audio based on voice activity detection.

    Yields (start_time, end_time, segment_bytes) tuples.

    """

    sample_rate, audio = wave_file

    vad = webrtcvad.Vad(aggressiveness)

    frames = frame_generator(FRAME_LENGTH_MS, audio, sample_rate)

    frames = list(frames)

    segments = []

    # VADのサンプルコードを参考に、より洗練されたセグメント抽出ロジックを実装

    # https://github.com/wiseman/py-webrtcvad/blob/master/example.py

    # 各フレームが発話かどうかを判定

    is_speech_flags = [vad.is_speech(frame.bytes, sample_rate) for frame in frames]

    # 連続する発話/無音の長さを追跡

    current_speech_frames = 0

    current_silence_frames = 0

    segment_start_frame_index = 0

    for i, is_speech in enumerate(is_speech_flags):

        if is_speech:

            current_speech_frames += 1

            current_silence_frames = 0

        else:

            current_silence_frames += 1

            current_speech_frames = 0  # 無音であれば発話カウントをリセット

        # 発話が一定時間続いたらセグメント開始 (または継続)

        if (
            current_speech_frames * FRAME_LENGTH_MS >= min_speech_duration_ms
            and segment_start_frame_index == 0
        ):

            segment_start_frame_index = (
                i - current_speech_frames + 1
            )  # 発話開始フレームのインデックス

        # 無音が一定時間続いたらセグメント終了

        if (
            current_silence_frames * FRAME_LENGTH_MS >= min_silence_duration_ms
            and segment_start_frame_index != 0
        ):

            segment_end_frame_index = (
                i - current_silence_frames + 1
            )  # 無音開始フレームのインデックス

            # セグメントを抽出

            segment_bytes_list = []

            for j in range(segment_start_frame_index, segment_end_frame_index):

                segment_bytes_list.append(frames[j].bytes)

            start_time = frames[segment_start_frame_index].timestamp

            end_time = (
                frames[segment_end_frame_index - 1].timestamp
                + frames[segment_end_frame_index - 1].duration
            )

            yield (start_time, end_time, b"".join(segment_bytes_list))

            segment_start_frame_index = 0

            current_speech_frames = 0

            current_silence_frames = 0

    # 最後のセグメントを処理

    if segment_start_frame_index != 0:

        segment_end_frame_index = len(frames)

        segment_bytes_list = []

        for j in range(segment_start_frame_index, segment_end_frame_index):

            segment_bytes_list.append(frames[j].bytes)

        start_time = frames[segment_start_frame_index].timestamp

        end_time = (
            frames[segment_end_frame_index - 1].timestamp
            + frames[segment_end_frame_index - 1].duration
        )

        yield (start_time, end_time, b"".join(segment_bytes_list))


def format_time(seconds):
    """Formats seconds into MM:SS.ms string."""

    minutes = int(seconds // 60)

    remaining_seconds = seconds % 60

    return f"{minutes:02d}:{remaining_seconds:05.2f}"


def transcribe_audio(
    input_audio_path,
    output_text_path,
    primary_lang,
    secondary_lang,
    min_speech_duration_ms,
    min_silence_duration_ms,
    lang_prob_threshold,
    vad_aggressiveness,
):

    print("Loading Whisper model...")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    model = whisper.load_model("large", device=device)

    print(f"Whisper model loaded. Using device: {device}")

    sample_rate, audio_data = read_wave(input_audio_path)

    print("Detecting speech segments with VAD...")

    pid = os.getpid()

    input_filename_hash = hash(input_audio_path) % (10**8)

    with open(output_text_path, "w", encoding="utf-8") as f:

        for i, (
            segment_start_time_abs,
            segment_end_time_abs,
            segment_bytes,
        ) in enumerate(
            vad_segment_generator(
                (sample_rate, audio_data),
                vad_aggressiveness,
                min_speech_duration_ms,
                min_silence_duration_ms,
            )
        ):

            segment_audio_float = (
                np.frombuffer(segment_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

            segment_duration_vad = len(segment_audio_float) / sample_rate

            if segment_duration_vad < 1.0:  # Skip segments shorter than 1 second

                continue

            temp_audio_path = f"temp_vad_segment_{pid}_{input_filename_hash}_{i}.wav"

            sf.write(temp_audio_path, segment_audio_float, SAMPLE_RATE)

            segment_audio_whisper_for_lang_detect = whisper.load_audio(temp_audio_path)

            segment_audio_padded_for_lang_detect = whisper.pad_or_trim(
                segment_audio_whisper_for_lang_detect
            )

            segment_mel_for_lang_detect = whisper.log_mel_spectrogram(
                segment_audio_padded_for_lang_detect, n_mels=128
            ).to(model.device)

            _, probs = model.detect_language(segment_mel_for_lang_detect)

            most_likely_lang = max(probs, key=probs.get)

            primary_lang_prob = probs.get(primary_lang, 0)

            lang_forced = False  # Initialize flag

            if (
                most_likely_lang == primary_lang
                and primary_lang_prob < lang_prob_threshold
            ):

                detected_language_for_transcribe = secondary_lang

                lang_forced = True  # Mark as forced

            elif most_likely_lang != primary_lang:

                detected_language_for_transcribe = secondary_lang

            else:

                detected_language_for_transcribe = primary_lang

            # Get the probability of the most likely language that was initially detected by Whisper

            # regardless of whether it was forced to secondary_lang.

            # This is to show the *model's* confidence for the most likely language it found.

            # If the user wants the probability of the *final chosen* language, it would be different.

            # For "hi (80%)", it implies the probability of HI if HI was detected, or the prob of EN if EN was detected.

            # Let's use the probability of the `most_likely_lang` from Whisper's original detection.

            final_display_prob_value = probs.get(most_likely_lang, 0)

            # Determine the language string for logging (e.g., "hi" or "hi*")

            display_lang_str = detected_language_for_transcribe

            if lang_forced:

                display_lang_str += "*"

            # Format the confidence percentage

            confidence_str = f" ({int(final_display_prob_value * 100)}%)"

            transcribe_result = whisper.transcribe(
                model,
                temp_audio_path,
                fp16=False,
                task="transcribe",
                language=detected_language_for_transcribe,
            )

            for segment_whisper in transcribe_result["segments"]:

                start_time_final = segment_start_time_abs + segment_whisper["start"]

                end_time_final = segment_start_time_abs + segment_whisper["end"]

                text_final = segment_whisper["text"]

                language_final = detected_language_for_transcribe

                output_line = f"{format_time(start_time_final)} [{language_final}] {text_final.strip()}"

                if language_final == secondary_lang:

                    if secondary_lang == "hi":

                        roman_text = transliterate(
                            text_final.strip(), sanscript.DEVANAGARI, sanscript.IAST
                        )

                        output_line += f" ({roman_text})"

                    else:

                        output_line += (
                            " (Transliteration not available for this language)"
                        )

                f.write(output_line + "\n")

                f.flush()  # Flush to disk immediately

                # VADセグメントの書き起こし結果の最初の部分をログに追記

                first_30_chars = text_final[:30].replace("\n", " ") + (
                    "..." if len(text_final) > 30 else ""
                )

                print(
                    f"  {format_time(segment_start_time_abs)} - {format_time(segment_end_time_abs)}: {display_lang_str}{confidence_str} - {first_30_chars}"
                )

            os.remove(temp_audio_path)  # Delete temporary file

    print("Transcription complete.")

    print(f"--- Process finished ---")

    print(f"Results saved to {output_text_path}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Transcribe multi-language audio using Whisper and VAD."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input audio file path (e.g., input_source.mp3)",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=None,
        help="Output text file path (e.g., output_text.txt) or a directory. If not specified, output will be saved in the same directory as the input audio file.",
    )

    parser.add_argument(
        "-l",
        "--lang_codes",
        required=True,
        help="Comma-separated language codes (e.g., en,hi). First is primary, second is fallback.",
    )

    parser.add_argument(
        "--min_speech_duration_ms",
        type=int,
        default=300,
        help="Minimum duration of speech to consider as a segment (ms). Default: 300ms.",
    )

    parser.add_argument(
        "--min_silence_duration_ms",
        type=int,
        default=600,
        help="Minimum duration of silence to consider as a segment boundary (ms). Default: 600ms.",
    )

    parser.add_argument(
        "--lang_prob_threshold",
        type=float,
        default=0.90,
        help="Probability threshold to force secondary language. If primary lang prob is below this, use secondary. Default: 0.90.",
    )

    parser.add_argument(
        "--vad_aggressiveness",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Set VAD aggressiveness mode (0-3). 0 is least aggressive (most sensitive), 3 is most aggressive. Default: 1.",
    )

    args = parser.parse_args()

    output_path = args.output

    if output_path is None:

        # If -o is not provided, use the input file's directory

        input_dir = os.path.dirname(args.input)

        input_basename = os.path.basename(args.input)

        filename_without_ext = os.path.splitext(input_basename)[0]

        output_filename = f"{filename_without_ext}.txt"

        output_path = os.path.join(input_dir, output_filename)

        print(
            f"Output path not specified. Writing to: {output_path} (same as input audio)"
        )

    elif os.path.isdir(output_path):

        # If -o is a directory, use it as before

        input_basename = os.path.basename(args.input)

        filename_without_ext = os.path.splitext(input_basename)[0]

        output_filename = f"{filename_without_ext}.txt"

        output_path = os.path.join(output_path, output_filename)

        print(f"Output path is a directory. Writing to: {output_path}")

    # Else (output_path is a file path), use it as is.

    lang_codes = args.lang_codes.split(",")

    if len(lang_codes) < 2:

        print("Error: Please provide at least two language codes (primary, secondary).")

        exit(1)

    primary_lang = lang_codes[0].strip()

    secondary_lang = lang_codes[1].strip()

    transcribe_audio(
        args.input,
        output_path,
        primary_lang,
        secondary_lang,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        lang_prob_threshold=args.lang_prob_threshold,
        vad_aggressiveness=args.vad_aggressiveness,
    )
