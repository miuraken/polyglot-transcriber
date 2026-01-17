import argparse
import os
from collections import deque

import librosa
import numpy as np
import soundfile as sf
import torch
import webrtcvad
import whisper
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

# --- Constants ---
VAD_AGGRESSIVENESS = 1  # Aggressiveness mode for VAD (0-3)
SAMPLE_RATE = 16000     # Required sample rate for webrtcvad
FRAME_LENGTH_MS = 30    # Frame duration in milliseconds

def read_wave(path):
    """Reads audio file and converts to 16-bit PCM format for VAD processing."""
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio_int16 = (audio * 32767).astype(np.int16)
    return sr, audio_int16.tobytes()

class Frame:
    """Represents a short frame of audio data for VAD analysis."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from raw PCM data."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset, timestamp = 0, 0.0
    duration = (n / 2) / sample_rate
    while offset + n <= len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        offset += n
        timestamp += duration

def vad_segment_generator(wave_file, aggressiveness, min_speech_duration_ms=300, min_silence_duration_ms=600):
    """Identifies speech segments using Voice Activity Detection (VAD)."""
    sample_rate, audio = wave_file
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(frame_generator(FRAME_LENGTH_MS, audio, sample_rate))
    
    is_speech_flags = [vad.is_speech(f.bytes, sample_rate) for f in frames]
    current_speech_frames = 0
    current_silence_frames = 0
    start_index = 0

    for i, is_speech in enumerate(is_speech_flags):
        if is_speech:
            current_speech_frames += 1
            current_silence_frames = 0
        else:
            current_silence_frames += 1
            current_speech_frames = 0

        # Start segment if speech duration exceeds threshold
        if current_speech_frames * FRAME_LENGTH_MS >= min_speech_duration_ms and start_index == 0:
            start_index = i - current_speech_frames + 1

        # End segment if silence duration exceeds threshold
        if current_silence_frames * FRAME_LENGTH_MS >= min_silence_duration_ms and start_index != 0:
            end_index = i - current_silence_frames + 1
            yield _create_segment(frames, start_index, end_index)
            start_index, current_speech_frames, current_silence_frames = 0, 0, 0

    if start_index != 0:
        yield _create_segment(frames, start_index, len(frames))

def _create_segment(frames, start, end):
    """Helper to combine frames into a single audio segment."""
    data = b"".join(f.bytes for f in frames[start:end])
    return (frames[start].timestamp, frames[end-1].timestamp + frames[end-1].duration, data)

def format_time(seconds):
    """Converts seconds to MM:SS.ms string format."""
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"

def transcribe_audio(input_audio_path, output_text_path, primary_lang, secondary_lang, 
                    min_speech_duration_ms, min_silence_duration_ms, lang_prob_threshold, vad_aggressiveness):
    """Main workflow: Segment audio via VAD and transcribe each part using Whisper."""
    print("Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = whisper.load_model("large", device=device)
    
    sr, audio_data = read_wave(input_audio_path)
    pid, h = os.getpid(), hash(input_audio_path) % (10**8)

    with open(output_text_path, "w", encoding="utf-8") as f:
        for i, (start_abs, end_abs, segment_bytes) in enumerate(
            vad_segment_generator((sr, audio_data), vad_aggressiveness, min_speech_duration_ms, min_silence_duration_ms)
        ):
            # Convert to float32 and skip very short segments
            audio_f32 = np.frombuffer(segment_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_f32) / sr < 1.0:
                continue

            # Create temp file for language detection and transcription
            tmp = f"temp_{pid}_{h}_{i}.wav"
            sf.write(tmp, audio_f32, SAMPLE_RATE)
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(whisper.load_audio(tmp)), n_mels=128).to(model.device)
            
            _, probs = model.detect_language(mel)
            most_likely = max(probs, key=probs.get)
            
            # Logic: Force secondary language if primary language confidence is too low
            lang_forced = (most_likely == primary_lang and probs.get(primary_lang, 0) < lang_prob_threshold)
            target_lang = secondary_lang if (lang_forced or most_likely != primary_lang) else primary_lang

            # Execute transcription
            result = whisper.transcribe(model, tmp, fp16=False, language=target_lang)
            
            for seg in result["segments"]:
                text = seg["text"].strip()
                output = f"{format_time(start_abs + seg['start'])} [{target_lang}] {text}"
                
                # Special handling for Hindi transliteration
                if target_lang == "hi" == secondary_lang:
                    output += f" ({transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)})"
                
                f.write(output + "\n")
                f.flush()
                print(f"  {format_time(start_abs)}: {target_lang}{'*' if lang_forced else ''} ({int(probs[most_likely]*100)}%) - {text[:30]}...")

            os.remove(tmp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-language transcription with VAD & Whisper")
    parser.add_argument("-i", "--input", required=True, help="Path to input audio file")
    parser.add_argument("-o", "--output", help="Path to output text file")
    parser.add_argument("-l", "--lang_codes", required=True, help="Comma-separated codes (e.g., en,hi)")
    parser.add_argument("--min_speech_duration_ms", type=int, default=300)
    parser.add_argument("--min_silence_duration_ms", type=int, default=600)
    parser.add_argument("--lang_prob_threshold", type=float, default=0.90)
    parser.add_argument("--vad_aggressiveness", type=int, default=1, choices=[0,1,2,3])

    args = parser.parse_args()
    out = args.output or os.path.splitext(args.input)[0] + ".txt"
    langs = [l.strip() for l in args.lang_codes.split(",")]
    
    if len(langs) < 2:
        print("Error: Two language codes required."); exit(1)

    transcribe_audio(args.input, out, langs[0], langs[1], args.min_speech_duration_ms, 
                     args.min_silence_duration_ms, args.lang_prob_threshold, args.vad_aggressiveness)
