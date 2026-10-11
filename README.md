# Polyglot Transcriber

This project provides a command-line tool to transcribe multi-language audio files (e.g. language-course audio that mixes an explanation language with the target language). It segments audio with Silero VAD, decides the language of each segment with Whisper, transcribes it, and optionally appends a pronunciation reading (IAST for Indic scripts, romaji / katakana for Korean).

## Features

*   **Multi-language Transcription**: Transcribes audio files containing multiple languages.
*   **Voice Activity Detection (VAD)**: Segments audio with [Silero VAD](https://github.com/snakers4/silero-vad), allowing language detection and transcription per utterance.
*   **Fast on Apple Silicon**: Uses [mlx-whisper](https://pypi.org/project/mlx-whisper/) with `large-v3-turbo` on M-series Macs; falls back to `openai-whisper` elsewhere.
*   **Configurable Language Handling**:
    *   Specify a primary language and a fallback secondary language.
    *   A segment is assigned to the primary language only if Whisper is confident about it; everything else is transcribed as the secondary language.
*   **Pronunciation Readings** (`--reading`):
    *   Indic scripts (Hindi, Marathi, Nepali, Sanskrit, Bengali, etc.): IAST.
    *   Korean: Revised Romanization with pronunciation rules (via [koroman](https://pypi.org/project/koroman/)) and/or katakana.
    *   Can be turned off with `--reading none` once you no longer need it.
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
    `mlx-whisper` is installed only on Apple Silicon Macs.

4.  **Install FFmpeg** (used to decode audio files):
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

The Whisper model is downloaded automatically on first run (about 1.6GB for `large-v3-turbo`).

## Usage

```bash
python3 transcriber.py -i <input_audio_file> [-o <output_path>] -l <primary_lang,secondary_lang> [options]
```

### Arguments

*   `-i, --input <input_audio_file>`: Path to the input audio file (e.g., `audio.mp3`, `audio.m4a`). **Required.**
*   `-o, --output <output_path>`: Path to the output text file or a directory. If a directory is given, the output is named after the input file (`<input>.txt`). If omitted, the output is saved next to the input file with a `.txt` extension.
*   `-l, --lang_codes <primary_lang,secondary_lang>`: Comma-separated language codes. The first is the primary language (e.g. the explanation language `ja`), the second is the fallback secondary language (e.g. the target language `ko`). **Required.**
*   `--reading <romaji|kana|both|none>`: Reading appended to supported languages. Default: `romaji`. `kana` is available for Korean only; for other languages it falls back to romaji (IAST).
*   `--model <name>`: Whisper model name or Hugging Face repo. Default: `large-v3-turbo`.
*   `--backend <auto|mlx|torch>`: `auto` uses MLX on Apple Silicon and openai-whisper elsewhere. Default: `auto`.
*   `--lang_prob_threshold <float>`: The primary language is chosen only if its detected probability is at least this value; otherwise the secondary language is used. Default: `0.90`.
*   `--vad_threshold <float>`: Silero VAD speech probability threshold. Lower = more sensitive. Default: `0.5`.
*   `--min_speech_duration_ms <ms>`: Minimum duration of speech to consider as a segment. Default: `200`.
*   `--min_silence_duration_ms <ms>`: Minimum duration of silence to consider as a segment boundary. Default: `300`.
*   `--speech_pad_ms <ms>`: Padding added around each speech region. Default: `100`.
*   `--min_segment_duration_ms <ms>`: VAD segments shorter than this are skipped. Default: `500`. Lower it if very short utterances are missing; raise it if Whisper hallucinates on short noises.

### Examples

Korean course explained in Japanese, with romaji and katakana readings:

```bash
python3 transcriber.py -i korean_lesson01.mp3 -l ja,ko --reading both
```

```
00:03.12 [ja] 今日は挨拶の表現を学びましょう。
00:06.48 [ko] 안녕하세요? (annyeonghaseyo? / アンニョンハセヨ?)
00:08.90 [ko] 감사합니다. (gamsahamnida. / カㇺサハㇺニダ.)
```

Once you no longer need the readings:

```bash
python3 transcriber.py -i korean_lesson20.mp3 -l ja,ko --reading none
```

Hindi course explained in English:

```bash
python3 transcriber.py -i hindi1_sample.m4a -l en,hi
```

```
00:01.98 [en] This is Unit 1 of Pimsleur's Hindi.
00:05.47 [en] Listen to this Hindi conversation.
00:09.69 [hi] क्या आप अंग्रेजी समझती हैं? (kyā āpa aṃgrejī samajhatī haiṃ?)
00:12.09 [hi] जी नहीं (jī nahīṃ)
00:13.31 [hi] मैं अंग्रेजी नहीं समझती हूँ (maiṃ aṃgrejī nahīṃ samajhatī hū~)
00:22.56 [en] in the next few minutes.
```

### Tuning

*   Short words are missing → lower `--vad_threshold` (e.g. `0.35`) and/or `--min_segment_duration_ms` (e.g. `300`).
*   Target-language segments are labeled as the primary language → **raise** `--lang_prob_threshold` (e.g. `0.95`).
*   Primary-language segments are labeled as the target language → **lower** `--lang_prob_threshold` (e.g. `0.80`).

The console output shows progress like this:

```
  00:01.98: en (99%) - This is Unit 1 of Pimsleur's H...
  00:09.69: hi (98%) - क्या आप अंग्रेजी समझती हैं?...
  00:20.60: hi* (57%) - जीहा...
```

The percentage is the confidence of the language decision. An asterisk (e.g. `hi*`) marks a low-confidence decision (below 75%) that is worth reviewing.

## Supported Readings

| Code | Language | `romaji` | `kana` |
|---|---|---|---|
| `ko` | Korean | Revised Romanization with pronunciation rules | Katakana (e.g. 감사합니다 → カㇺサハㇺニダ) |
| `hi` | Hindi | IAST | – |
| `mr` | Marathi | IAST | – |
| `ne` | Nepali | IAST | – |
| `sa` | Sanskrit | IAST | – |
| `bn` | Bengali | IAST | – |
| `gu` | Gujarati | IAST | – |
| `pa` | Punjabi | IAST | – |
| `ta` | Tamil | IAST | – |
| `te` | Telugu | IAST | – |
| `kn` | Kannada | IAST | – |
| `ml` | Malayalam | IAST | – |

Readings are implemented in [readings.py](readings.py). Katakana for Korean is an approximation for learners (liaison, nasalization and voicing are applied; final consonants use small kana such as ㇰ ㇽ ㇺ ㇷ゚).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
