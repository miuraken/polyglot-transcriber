# Polyglot Transcriber

This project provides a command-line tool to transcribe multi-language audio files using OpenAI's Whisper model combined with Voice Activity Detection (VAD) for segmenting audio by silence. It's particularly useful for language learners who need to transcribe mixed-language audio, offering the flexibility to specify primary and secondary languages, and to include phonetic transliteration for the secondary language.

## Features

*   **Multi-language Transcription**: Transcribes audio files containing multiple languages.
*   **Voice Activity Detection (VAD)**: Segments audio based on silence, allowing for more accurate language detection and transcription per segment.
*   **Configurable Language Handling**:
    *   Specify a primary language and a fallback secondary language.
    *   Any segment not detected as the primary language will be transcribed using the secondary language.
*   **Phonetic Transliteration**: Optionally adds IAST (International Alphabet of Sanskrit Transliteration) phonetic notation for the secondary language (currently supported for Hindi).
*   **Command-Line Interface**: Easy to use via command-line arguments.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/miuraken/polyglot-transcriber.git
    cd polyglot-transcriber
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install FFmpeg:**
    `pydub` (used internally by some libraries) and `librosa` rely on FFmpeg for audio processing. If you don't have FFmpeg installed, please install it.
    
    *   **macOS (using Homebrew):**
        ```bash
        brew install ffmpeg
        ```
    *   **Linux (Debian/Ubuntu):**
        ```bash
        sudo apt update
        sudo apt install ffmpeg
        ```
    *   **Windows:**
        Download from the [FFmpeg website](https://ffmpeg.org/download.html) and add it to your system's PATH.

## Usage

Run the `transcriber.py` script with the following command-line arguments:

```bash
python3 transcriber.py -i <input_audio_file> -o <output_text_file> -l <primary_lang,secondary_lang> [-t] [--min_speech_duration_ms <ms>] [--min_silence_duration_ms <ms>]
```

### Arguments

*   `-i, --input <input_audio_file>`: Path to the input audio file (e.g., `audio.mp3`, `audio.m4a`). **Required.**
*   `-o, --output <output_path>`: Path to the output text file (e.g., `transcription.txt`) or a directory. If a directory path is provided, the output file will be automatically named based on the input file (e.g., `<input_audio_file>.txt`) and saved in that directory. **Required.**
*   `-l, --lang_codes <primary_lang,secondary_lang>`: Comma-separated language codes. The first code is the primary language (e.g., `en`), and the second is the fallback secondary language (e.g., `hi`). Any segment not detected as the primary language will be transcribed using the secondary language. **Required.**
*   `-t, --transliterate`: (Optional switch) If specified, phonetic transliteration (IAST) will be added in parentheses for segments transcribed in the secondary language.
*   `--min_speech_duration_ms <ms>`: Minimum duration of speech to consider as a segment (ms). Default: `300ms`.
*   `--min_silence_duration_ms <ms>`: Minimum duration of silence to consider as a segment boundary (ms). Default: `600ms`.

### Example

To transcribe `hindi1_sample.m4a` with English as primary and Hindi as secondary, including IAST transliteration for Hindi:

```bash
python3 transcriber.py -i hindi1_sample.m4a -o transcription.txt -l en,hi -t
```

The console output will show progress like this:

```
...
  VAD Segment 00:00.00 - 00:03.42: Detected language = en - "This is Unit 1 of Pimsleur's Hindi."
  VAD Segment 00:03.42 - 00:05.94: Detected language = en - "Listen to this Hindi conversation."
  VAD Segment 00:05.94 - 00:09.12: Detected language = hi - "सुनिये, क्या आप अंग्रेजी समझती हैं?"
...
```

The output `transcription.txt` will look something like this:

```
00:00.00 [en] This is Unit 1 of Pimsleur's Hindi.
00:03.42 [en] Listen to this Hindi conversation.
00:05.94 [hi] सुनिये, क्या आप अंग्रेजी समझती हैं? (suniye, kyā āpa aṃgrejī samajhatī haiṃ?)
00:09.12 [hi] जी नहीं, मैं अंग्रेजी नहीं समझती हूँ. (jī nahīṃ, maiṃ aṃgrejī nahīṃ samajhatī hū~.)
00:12.60 [hi] मैं थोड़ी थोड़ी हिंदी समझता हूँ. (maiṃ thor̤ī thor̤ī hiṃdī samajhatā hū~.)
00:15.12 [hi] क्या आप अमरीकी हैं? (kyā āpa amarīkī haiṃ?)
00:17.10 [en] Jee haan.
00:18.24 [en] In the next few minutes, you'll learn not only to understand this conversation,
00:23.43 [en] but to take part in it yourself.
00:25.71 [en] Imagine that an American man meets a
```

## Supported Languages for Transliteration

Currently, phonetic transliteration (`-t` option) is only supported for **Hindi (`hi`)**. Support for other languages will be added in future updates.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.