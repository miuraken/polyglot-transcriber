# Polyglot Transcriber

This project provides a command-line tool to transcribe multi-language audio files using OpenAI's Whisper model combined with Voice Activity Detection (VAD) for segmenting audio by silence. It's particularly useful for language learners who need to transcribe mixed-language audio, offering the flexibility to specify primary and secondary languages, and to include IAST phonetic transliteration for Indic-script languages such as Hindi.

## Features

*   **Multi-language Transcription**: Transcribes audio files containing multiple languages.
*   **Voice Activity Detection (VAD)**: Segments audio based on silence, allowing for more accurate language detection and transcription per segment.
*   **Configurable Language Handling**:
    *   Specify a primary language and a fallback secondary language.
    *   Any segment not detected as the primary language will be transcribed using the secondary language.
*   **Phonetic Transliteration**: Automatically adds IAST (International Alphabet of Sanskrit Transliteration) phonetic notation for segments transcribed in an Indic-script language (Hindi, Marathi, Nepali, Sanskrit, Bengali, etc.).
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
python3 transcriber.py -i <input_audio_file> [-o <output_path>] -l <primary_lang,secondary_lang> [options]
```

### Arguments

*   `-i, --input <input_audio_file>`: Path to the input audio file (e.g., `audio.mp3`, `audio.m4a`). **Required.**
*   `-o, --output <output_path>`: Path to the output text file (e.g., `transcription.txt`) or a directory. If a directory path is provided, the output file will be automatically named based on the input file (e.g., `<input_audio_file>.txt`) and saved in that directory. If not specified, the output file will be saved in the same directory as the input audio file, named after the input file with a `.txt` extension.
*   `-l, --lang_codes <primary_lang,secondary_lang>`: Comma-separated language codes. The first code is the primary language (e.g., `en`), and the second is the fallback secondary language (e.g., `hi`). **Required.**
*   `--min_speech_duration_ms <ms>`: Minimum duration of speech to consider as a segment (ms). Default: `200ms`.
*   `--min_silence_duration_ms <ms>`: Minimum duration of silence to consider as a segment boundary (ms). Default: `300ms`.
*   `--min_segment_duration_ms <ms>`: VAD segments shorter than this are skipped (ms). Default: `500ms`. Lower it if very short utterances are missing; raise it if Whisper hallucinates on short noises.
*   `--lang_prob_threshold <float>`: Probability threshold for language detection. If the detected probability of the primary language is below this value, the script will fall back to the secondary language. Default: `0.90`. This is useful for forcing the secondary language when Whisper is not confident about its detection of the primary language.
*   `--vad_aggressiveness <0-3>`: Sets the VAD aggressiveness mode. An integer from 0 to 3, where 0 is the least aggressive (most sensitive to picking up speech) and 3 is the most aggressive (least sensitive). Default: `1`.

### Example

To transcribe `hindi1_sample.m4a` with English as primary and Hindi as secondary, saving the output to `hindi1_sample.txt` in the same directory:

```bash
python3 transcriber.py -i hindi1_sample.m4a -l en,hi
```

You can still specify an output path if needed:

```bash
python3 transcriber.py -i hindi1_sample.m4a -o transcription.txt -l en,hi
```

If you find that short words are being missed entirely, try making the VAD more sensitive by setting `--vad_aggressiveness` to `0`:
```bash
python3 transcriber.py -i your_long_file.mp3 -l en,hi --vad_aggressiveness 0
```

If you find that Hindi segments are being misidentified as English, you can try lowering the language probability threshold:
```bash
python3 transcriber.py -i your_long_file.mp3 -l en,hi --lang_prob_threshold 0.80
```

The console output will show progress like this:

```
...
  00:02.04: en (99%) - This is Unit 1 of Pimsleur's H...
  00:05.55: en (98%) - Listen to this Hindi conversat...
  00:08.79: hi (95%) - सुनिये, क्या आप अंग्रेजी समझती...
...
```

An asterisk after the language code (e.g. `hi*`) means the primary language was detected but with a probability below `--lang_prob_threshold`, so the secondary language was forced.

The output `transcription.txt` will look something like this:

```
00:00.00 [en] This is Unit 1 of Hindi Lesson.
00:05.59 [en] Listen to this Hindi conversation.
00:08.82 [hi] सुन्ये, क्या आप अंग्रेजी समझती हैं? (sunye, kyā āpa aṃgrejī samajhatī haiṃ?)
00:12.14 [hi] जी नहीं, मैं अंग्रेजी नहीं समझती हूँ. (jī nahīṃ, maiṃ aṃgrejī nahīṃ samajhatī hū~.)
00:15.80 [hi] मैं थोड़ी थोड़ी हिंदी समझता हूँ. (maiṃ thor̤ī thor̤ī hiṃdī samajhatā hū~.)
00:18.62 [hi] क्या आप अमरीकी हैं? (kyā āpa amarīkī haiṃ?)
00:20.12 [hi] जी हाँ. (jī hā~.)
00:22.65 [en] In the next few minutes, you'll learn not only to understand this conversation,
00:27.71 [en] but to take part in it yourself.
```

## Supported Languages for Transliteration

IAST transliteration is applied automatically to any segment transcribed in one of the following languages (IAST is designed for Indic/Brahmic scripts, so other languages are not transliterated):

| Code | Language | Script |
|---|---|---|
| `hi` | Hindi | Devanagari |
| `mr` | Marathi | Devanagari |
| `ne` | Nepali | Devanagari |
| `sa` | Sanskrit | Devanagari |
| `bn` | Bengali | Bengali |
| `gu` | Gujarati | Gujarati |
| `pa` | Punjabi | Gurmukhi |
| `ta` | Tamil | Tamil |
| `te` | Telugu | Telugu |
| `kn` | Kannada | Kannada |
| `ml` | Malayalam | Malayalam |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
