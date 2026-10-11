import argparse
import os
import platform
import sys

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from readings import READING_STYLES, make_reading, relabel_by_script

# --- Global Constants ---
SAMPLE_RATE = 16000                 # Whisper and Silero VAD both expect 16kHz mono
DEFAULT_MODEL = "large-v3-turbo"
LOW_CONFIDENCE = 0.75               # Language decisions below this share are marked with '*'


# --- ASR backends ----------------------------------------------------------------
class MLXWhisperBackend:
    """Whisper on Apple Silicon GPU via MLX (fast on M-series Macs)."""
    name = "mlx"

    def __init__(self, model):
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper import audio as mlx_audio
        from mlx_whisper.transcribe import ModelHolder

        self._mx, self._mlx_whisper, self._audio = mx, mlx_whisper, mlx_audio
        self.repo = model if "/" in model else f"mlx-community/whisper-{model}"
        # Same dtype as mlx_whisper.transcribe's default (fp16) so the cached model is reused
        self.model = ModelHolder.get_model(self.repo, mx.float16)

    def load_audio(self, path):
        return np.asarray(self._audio.load_audio(path), dtype=np.float32)

    def language_probs(self, audio):
        a = self._audio
        mel = a.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels, padding=a.N_SAMPLES)
        mel = a.pad_or_trim(mel, a.N_FRAMES, axis=-2).astype(self._mx.float16)
        _, probs = self.model.detect_language(mel)
        return probs

    def transcribe(self, audio, language):
        result = self._mlx_whisper.transcribe(
            audio, path_or_hf_repo=self.repo, language=language,
            condition_on_previous_text=False, verbose=None,
        )
        return result["segments"]


class TorchWhisperBackend:
    """Reference openai-whisper implementation (CUDA / CPU; works on any platform)."""
    name = "torch"

    def __init__(self, model):
        import whisper
        self._whisper = whisper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model, device=self.device)

    def load_audio(self, path):
        return self._whisper.load_audio(path)

    def language_probs(self, audio):
        w = self._whisper
        mel = w.log_mel_spectrogram(w.pad_or_trim(audio), n_mels=self.model.dims.n_mels).to(self.device)
        _, probs = self.model.detect_language(mel)
        return probs

    def transcribe(self, audio, language):
        result = self._whisper.transcribe(
            self.model, audio, language=language, fp16=(self.device == "cuda"),
            condition_on_previous_text=False,
        )
        return result["segments"]


def create_backend(backend, model):
    if backend == "auto":
        is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
        backend = "mlx" if is_apple_silicon else "torch"
    return MLXWhisperBackend(model) if backend == "mlx" else TorchWhisperBackend(model)


# --- Pipeline ----------------------------------------------------------------------
def vad_segments(audio, threshold, min_speech_duration_ms, min_silence_duration_ms, speech_pad_ms):
    """Yields (start_sec, end_sec, samples) for each speech region found by Silero VAD."""
    vad_model = load_silero_vad()
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio), vad_model,
        threshold=threshold, sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
    for ts in timestamps:
        yield ts["start"] / SAMPLE_RATE, ts["end"] / SAMPLE_RATE, audio[ts["start"]:ts["end"]]


def choose_language(probs, primary_lang, secondary_lang, threshold):
    """
    Chooses the primary language only when Whisper is confident about it;
    everything else falls back to the secondary language.

    Comparing only the two expected languages does NOT work well: e.g. short Hindi
    utterances are often detected as Urdu (or unrelated languages), so the residual
    probability must count against the primary language rather than be ignored.
    Returns (language, confidence).
    """
    p = probs.get(primary_lang, 0.0)
    if p >= threshold:
        return primary_lang, p
    return secondary_lang, 1.0 - p


def format_time(seconds):
    """Formats time in seconds to a human-readable MM:SS.ms string."""
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def transcribe_audio(input_audio_path, output_text_path, primary_lang, secondary_lang, args):
    """
    Main transcription pipeline:
    1. Loads the Whisper backend (MLX on Apple Silicon, openai-whisper elsewhere).
    2. Segments audio using Silero VAD.
    3. Chooses between the primary/secondary language for each segment.
    4. Transcribes and appends readings (IAST / romaji / kana) where supported.
    """
    print("Loading Whisper model...")
    backend = create_backend(args.backend, args.model)
    print(f"Backend: {backend.name}")

    audio = backend.load_audio(input_audio_path)
    min_samples = SAMPLE_RATE * args.min_segment_duration_ms / 1000

    with open(output_text_path, "w", encoding="utf-8") as f:
        for start_abs, end_abs, segment in vad_segments(
            audio, args.vad_threshold, args.min_speech_duration_ms,
            args.min_silence_duration_ms, args.speech_pad_ms,
        ):
            if len(segment) < min_samples:
                continue

            probs = backend.language_probs(segment)
            target_lang, confidence = choose_language(
                probs, primary_lang, secondary_lang, args.lang_prob_threshold
            )
            low = confidence < LOW_CONFIDENCE

            for seg in backend.transcribe(segment, target_lang):
                text = seg["text"].strip()
                if not text:
                    continue
                # Fix the label when the transcript is clearly in the other language's script
                lang = relabel_by_script(text, target_lang, (primary_lang, secondary_lang))
                relabeled = lang != target_lang
                output = f"{format_time(start_abs + seg['start'])} [{lang}] {text}"

                reading = make_reading(text, lang, args.reading)
                if reading:
                    output += f" ({reading})"

                f.write(output + "\n")
                f.flush()
                # Console logging ('*' = low-confidence language decision, '!' = relabeled by script)
                mark = "!" if relabeled else ("*" if low else "")
                print(f"  {format_time(start_abs + seg['start'])}: {lang}{mark} "
                      f"({int(confidence * 100)}%) - {text[:30]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe multi-language audio using VAD & Whisper.")
    parser.add_argument("-i", "--input", required=True, help="Input audio path.")
    parser.add_argument("-o", "--output", help="Output path (file or directory).")
    parser.add_argument("-l", "--lang_codes", required=True, help="Codes e.g., 'en,hi' or 'ja,ko'.")
    parser.add_argument("--reading", default="romaji", choices=READING_STYLES,
                        help="Reading appended to supported languages (default: romaji).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Whisper model name or HF repo (default: {DEFAULT_MODEL}).")
    parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"])
    parser.add_argument("--min_speech_duration_ms", type=int, default=200)
    parser.add_argument("--min_silence_duration_ms", type=int, default=300)
    parser.add_argument("--speech_pad_ms", type=int, default=100,
                        help="Padding added around each VAD speech region (ms).")
    parser.add_argument("--min_segment_duration_ms", type=int, default=500,
                        help="Skip VAD segments shorter than this (ms).")
    parser.add_argument("--vad_threshold", type=float, default=0.5,
                        help="Silero VAD speech probability threshold (lower = more sensitive).")
    parser.add_argument("--lang_prob_threshold", type=float, default=0.90,
                        help="Primary language is chosen only if its detected probability is at least "
                             "this value; otherwise the secondary language is used.")

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

    transcribe_audio(args.input, out, langs[0], langs[1], args)
