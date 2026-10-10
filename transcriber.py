import argparse
import os
import sys

import librosa
import numpy as np
import torch
import webrtcvad
import whisper
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# --- Global Constants ---
VAD_AGGRESSIVENESS = 1  # Aggressiveness mode for VAD: 0 (gentle) to 3 (aggressive)
SAMPLE_RATE = 16000     # WebRTCVAD requires exactly 16kHz
FRAME_LENGTH_MS = 30    # Frame duration: Must be 10, 20, or 30ms

# Languages (Whisper codes) whose native script can be transliterated to IAST.
# IAST is designed for Indic (Brahmic) scripts, so only these are supported.
IAST_SOURCE_SCRIPTS = {
    "hi": sanscript.DEVANAGARI,  # Hindi
    "mr": sanscript.DEVANAGARI,  # Marathi
    "ne": sanscript.DEVANAGARI,  # Nepali
    "sa": sanscript.DEVANAGARI,  # Sanskrit
    "bn": sanscript.BENGALI,     # Bengali
    "gu": sanscript.GUJARATI,    # Gujarati
    "pa": sanscript.GURMUKHI,    # Punjabi
    "ta": sanscript.TAMIL,       # Tamil
    "te": sanscript.TELUGU,      # Telugu
    "kn": sanscript.KANNADA,     # Kannada
    "ml": sanscript.MALAYALAM,   # Malayalam
}


def read_wave(path):
    """
    Reads an audio file and converts it to 16-bit PCM format.
    Required for WebRTCVAD compatibility.
    """
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    # Clip to avoid int16 overflow, then convert float audio to 16-bit integer PCM
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return sr, audio_int16.tobytes()


class Frame:
    """Represents a short segment (frame) of audio data for VAD processing."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    Yields audio frames of specified duration from raw PCM audio data.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset, timestamp = 0, 0.0
    duration = (n / 2) / sample_rate
    while offset + n <= len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        offset += n
        timestamp += duration


def vad_segment_generator(wave_file, aggressiveness, min_speech_duration_ms=200, min_silence_duration_ms=300):
    """
    Analyzes audio frames and yields contiguous segments of speech.
    Uses a state-machine logic to filter out short noises or brief pauses.
    """
    sample_rate, audio = wave_file
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(frame_generator(FRAME_LENGTH_MS, audio, sample_rate))

    is_speech_flags = [vad.is_speech(f.bytes, sample_rate) for f in frames]
    current_speech_frames = 0
    current_silence_frames = 0
    start_index = None  # None = not currently inside a speech segment

    for i, is_speech in enumerate(is_speech_flags):
        if is_speech:
            current_speech_frames += 1
            current_silence_frames = 0
        else:
            current_silence_frames += 1
            current_speech_frames = 0

        # Trigger start of a segment if speech persists longer than threshold
        if start_index is None and current_speech_frames * FRAME_LENGTH_MS >= min_speech_duration_ms:
            start_index = i - current_speech_frames + 1

        # Trigger end of a segment if silence persists longer than threshold
        if start_index is not None and current_silence_frames * FRAME_LENGTH_MS >= min_silence_duration_ms:
            end_index = i - current_silence_frames + 1
            yield _create_segment(frames, start_index, end_index)
            start_index, current_speech_frames, current_silence_frames = None, 0, 0

    # Handle the final remaining segment if the file ends during speech
    if start_index is not None:
        yield _create_segment(frames, start_index, len(frames))


def _create_segment(frames, start, end):
    """Combines a range of frames into a single binary audio segment."""
    data = b"".join(f.bytes for f in frames[start:end])
    return (frames[start].timestamp, frames[end-1].timestamp + frames[end-1].duration, data)


def format_time(seconds):
    """Formats time in seconds to a human-readable MM:SS.ms string."""
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def transcribe_audio(input_audio_path, output_text_path, primary_lang, secondary_lang,
                     min_speech_duration_ms, min_silence_duration_ms, lang_prob_threshold, vad_aggressiveness,
                     min_segment_duration_ms):
    """
    Main transcription pipeline:
    1. Loads Whisper Large model.
    2. Segments audio using VAD.
    3. Detects language for each segment.
    4. Transcribes and applies IAST transliteration for Indic-script languages.
    """
    print("Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = whisper.load_model("large", device=device)

    sr, audio_data = read_wave(input_audio_path)

    with open(output_text_path, "w", encoding="utf-8") as f:
        for start_abs, end_abs, segment_bytes in vad_segment_generator(
            (sr, audio_data), vad_aggressiveness, min_speech_duration_ms, min_silence_duration_ms
        ):
            # Convert PCM bytes to float32 for Whisper (passed directly; no temp files)
            audio_f32 = np.frombuffer(segment_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_f32) / sr * 1000 < min_segment_duration_ms:
                continue

            mel = whisper.log_mel_spectrogram(
                whisper.pad_or_trim(audio_f32), n_mels=model.dims.n_mels
            ).to(model.device)

            # Language Detection Logic
            _, probs = model.detect_language(mel)
            most_likely = max(probs, key=probs.get)

            # Determine target language: Force secondary if primary confidence is too low
            lang_forced = (most_likely == primary_lang and probs.get(primary_lang, 0) < lang_prob_threshold)
            target_lang = secondary_lang if (lang_forced or most_likely != primary_lang) else primary_lang

            # Perform Transcription
            result = whisper.transcribe(model, audio_f32, fp16=False, language=target_lang)

            for seg in result["segments"]:
                text = seg["text"].strip()
                if not text:
                    continue
                output = f"{format_time(start_abs + seg['start'])} [{target_lang}] {text}"

                # Apply IAST transliteration for supported Indic-script languages
                if target_lang in IAST_SOURCE_SCRIPTS:
                    output += f" ({transliterate(text, IAST_SOURCE_SCRIPTS[target_lang], sanscript.IAST)})"

                f.write(output + "\n")
                f.flush()
                # Console logging for progress tracking
                print(f"  {format_time(start_abs)}: {target_lang}{'*' if lang_forced else ''} ({int(probs[most_likely]*100)}%) - {text[:30]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe multi-language audio using VAD & Whisper.")
    parser.add_argument("-i", "--input", required=True, help="Input audio path.")
    parser.add_argument("-o", "--output", help="Output path (file or directory).")
    parser.add_argument("-l", "--lang_codes", required=True, help="Codes e.g., 'en,hi'.")
    parser.add_argument("--min_speech_duration_ms", type=int, default=200)
    parser.add_argument("--min_silence_duration_ms", type=int, default=300)
    parser.add_argument("--min_segment_duration_ms", type=int, default=500,
                        help="Skip VAD segments shorter than this (ms).")
    parser.add_argument("--lang_prob_threshold", type=float, default=0.90)
    parser.add_argument("--vad_aggressiveness", type=int, default=VAD_AGGRESSIVENESS, choices=[0, 1, 2, 3])

    args = parser.parse_args()

    # Secure Path Handling: Correctly handles both directory and file outputs
    input_base = os.path.splitext(os.path.basename(args.input))[0]
    if args.output is None:
        out = os.path.splitext(args.input)[0] + ".txt"
    elif os.path.isdir(args.output):
        out = os.path.join(args.output, input_base + ".txt")
    else:
        out = args.output

    langs = [l.strip() for l in args.lang_codes.split(",")]
    if len(langs) < 2:
        print("Error: Please provide both primary and secondary language codes.")
        sys.exit(1)

    transcribe_audio(args.input, out, langs[0], langs[1], args.min_speech_duration_ms,
                     args.min_silence_duration_ms, args.lang_prob_threshold, args.vad_aggressiveness,
                     args.min_segment_duration_ms)
