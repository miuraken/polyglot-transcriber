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

# VAD aggressiveness setting (1: aggressive, 2: mid-aggressive, 3: least aggressive)
VAD_AGGRESSIVENESS = 1
SAMPLE_RATE = 16000 # webrtcvad requires 16kHz
FRAME_LENGTH_MS = 30 # webrtcvad requires 10, 20, or 30ms frame lengths

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
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2) # 2 bytes per sample for int16
    offset = 0
    timestamp = 0.0
    duration = (n / 2) / sample_rate
    while offset + n <= len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        offset += n
        timestamp += duration

def vad_segment_generator(wave_file, aggressiveness):
    """Generates segments of audio based on voice activity detection."""
    sample_rate, audio = wave_file
    vad = webrtcvad.Vad(aggressiveness)
    frames = frame_generator(FRAME_LENGTH_MS, audio, sample_rate)
    frames = list(frames)

    ring_buffer = deque(maxlen=10) # 0.3 seconds of frames
    triggered = False
    voiced_frames = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def transcribe_audio(input_audio_path, output_text_path, primary_lang, secondary_lang, add_transliteration):
    print("Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = whisper.load_model("large", device=device)
    print(f"Whisper model loaded. Using device: {device}")

    sample_rate, audio_data = read_wave(input_audio_path)

    all_segments_transcription = []
    current_overall_time = 0.0 # Track overall elapsed time

    print("Detecting speech segments with VAD...")
    for i, segment_bytes in enumerate(vad_segment_generator((sample_rate, audio_data), VAD_AGGRESSIVENESS)):
        segment_audio_float = np.frombuffer(segment_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segment_duration_vad = len(segment_audio_float) / sample_rate

        # Skip too short VAD segments to avoid unstable Whisper language detection
        if segment_duration_vad < 1.0: # Skip segments shorter than 1 second
            current_overall_time += segment_duration_vad
            continue

        # Save VAD segment to a temporary file
        temp_audio_path = f"temp_vad_segment_{i}.wav"
        sf.write(temp_audio_path, segment_audio_float, SAMPLE_RATE)

        # Detect language for the VAD segment using Whisper's low-level API
        segment_audio_whisper_for_lang_detect = whisper.load_audio(temp_audio_path)
        segment_audio_padded_for_lang_detect = whisper.pad_or_trim(segment_audio_whisper_for_lang_detect)
        segment_mel_for_lang_detect = whisper.log_mel_spectrogram(segment_audio_padded_for_lang_detect, n_mels=128).to(model.device)
        _, probs = model.detect_language(segment_mel_for_lang_detect)
        detected_language_for_transcribe = max(probs, key=probs.get)

        # Apply language code replacement logic
        if detected_language_for_transcribe != primary_lang:
            detected_language_for_transcribe = secondary_lang

        print(f"  VAD Segment {current_overall_time:.2f}s - {current_overall_time + segment_duration_vad:.2f}s: Detected language = {detected_language_for_transcribe}")

        # Call whisper.transcribe
        # transcribe handles internal padding/trimming and utterance boundary processing
        transcribe_result = whisper.transcribe(
            model,
            temp_audio_path,
            fp16=False, # Force FP32 for stability, even on GPU
            task="transcribe",
            language=detected_language_for_transcribe # Pass detected language to transcribe
        )
        
        # Use segments from transcribe_result for more detailed output
        for segment_whisper in transcribe_result["segments"]:
            all_segments_transcription.append({
                "start": current_overall_time + segment_whisper["start"],
                "end": current_overall_time + segment_whisper["end"],
                "language": detected_language_for_transcribe, # Language after replacement
                "text": segment_whisper["text"]
            })
        
        current_overall_time += segment_duration_vad # Update overall elapsed time
        os.remove(temp_audio_path) # Delete temporary file

    print("Transcription complete.")

    with open(output_text_path, "w", encoding="utf-8") as f:
        for segment in all_segments_transcription:
            output_line = f"{segment['start']:.2f} [{segment['language']}] {segment['text'].strip()}"
            
            if add_transliteration and segment['language'] == secondary_lang:
                if secondary_lang == "hi":
                    # Transliterate Devanagari to IAST
                    roman_text = transliterate(segment['text'].strip(), sanscript.DEVANAGARI, sanscript.IAST)
                    output_line += f" ({roman_text})"
                else:
                    output_line += " (Transliteration not available for this language)"
            
            f.write(output_line + "\n")

    print(f"--- Process finished ---")
    print(f"Results saved to {output_text_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe multi-language audio using Whisper and VAD.")
    parser.add_argument("-i", "--input", required=True, help="Input audio file path (e.g., input_source.mp3)")
    parser.add_argument("-o", "--output", required=True, help="Output text file path (e.g., output_text.txt)")
    parser.add_argument("-l", "--lang_codes", required=True, 
                        help="Comma-separated language codes (e.g., en,hi). First is primary, second is fallback.")
    parser.add_argument("-t", "--transliterate", action="store_true", 
                        help="If specified, add phonetic transliteration for the secondary language.")
    
    args = parser.parse_args()

    lang_codes = args.lang_codes.split(',')
    if len(lang_codes) < 2:
        print("Error: Please provide at least two language codes (primary, secondary).")
        exit(1)
    
    primary_lang = lang_codes[0].strip()
    secondary_lang = lang_codes[1].strip()

    transcribe_audio(args.input, args.output, primary_lang, secondary_lang, args.transliterate)
