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

# VADの感度設定 (1: 積極的, 2: 中程度, 3: 穏やか)
VAD_AGGRESSIVENESS = 1
SAMPLE_RATE = 16000 # webrtcvadは16kHzを要求
FRAME_LENGTH_MS = 30 # webrtcvadは10, 20, 30msのフレーム長を要求

def read_wave(path):
    """Reads a .wav file.
    Takes the path, returns (sample_rate, audio_data).
    """
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
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

    ring_buffer = deque(maxlen=10)
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
    print("Whisperモデルをロード中...")
    model = whisper.load_model("large")
    print("モデルのロードが完了しました。")

    print(f"音声ファイル {input_audio_path} の処理を開始します...")

    sample_rate, audio_data = read_wave(input_audio_path)

    all_segments_transcription = []
    segment_start_time = 0.0

    print("VADで音声セグメントを検出中...")
    for i, segment_bytes in enumerate(vad_segment_generator((sample_rate, audio_data), VAD_AGGRESSIVENESS)):
        temp_audio_path = f"temp_vad_segment_{i}.wav"
        segment_audio_np = np.frombuffer(segment_bytes, dtype=np.int16)
        write_wave(temp_audio_path, segment_audio_np, sample_rate)

        segment_audio_whisper = whisper.load_audio(temp_audio_path)
        segment_audio_padded = whisper.pad_or_trim(segment_audio_whisper)
        segment_mel = whisper.log_mel_spectrogram(segment_audio_padded, n_mels=128).to(model.device)

        _, probs = model.detect_language(segment_mel)
        detected_language = max(probs, key=probs.get)
        
        # 言語コードの置き換えロジック
        if detected_language != primary_lang:
            detected_language = secondary_lang
        
        segment_duration = len(segment_audio_np) / sample_rate
        segment_end_time = segment_start_time + segment_duration

        print(f"  セグメント {segment_start_time:.2f}s - {segment_end_time:.2f}s: 検出言語 = {detected_language}")

        segment_options = whisper.DecodingOptions(
            fp16=False,
            task="transcribe",
            language=detected_language
        )
        result = whisper.decode(model, segment_mel, segment_options)

        all_segments_transcription.append({
            "start": segment_start_time,
            "end": segment_end_time,
            "language": detected_language,
            "text": result.text
        })
        
        segment_start_time = segment_end_time
        os.remove(temp_audio_path)

    print("書き起こしが完了しました。")

    with open(output_text_path, "w", encoding="utf-8") as f:
        for segment in all_segments_transcription:
            output_line = f"{segment['start']:.2f} [{segment['language']}] {segment['text'].strip()}"
            
            # アルファベット表記の追記ロジック
            if add_transliteration and segment['language'] == secondary_lang:
                # 現在はヒンディー語のみ対応
                if secondary_lang == "hi":
                    roman_text = transliterate(segment['text'].strip(), sanscript.DEVANAGARI, sanscript.IAST)
                    output_line += f" ({roman_text})"
                else:
                    # 他の言語のローマ字表記は未実装
                    output_line += " (Transliteration not available for this language)"
            
            f.write(output_line + "\n")

    print(f"--- 完了 ---")
    print(f"結果は {output_text_path} に保存されました。")

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
