# Audiblez + Kokoro-82M Pipeline

EPUB-to-audiobook conversion pipeline. Converts `.epub` files (and optionally PDF/MOBI via Calibre) into `.m4b` audiobooks with chapter markers, using the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech model and [Audiblez](https://github.com/santinic/audiblez) for audio assembly.

Runs locally, open source, GPU-accelerated when CUDA is available.

---

## Features

- **Single file conversion** -- EPUB to M4B with chapter markers and optional cover art
- **Batch processing** -- Convert an entire directory of EPUBs in one command, with skip-existing and dry-run modes
- **Multi-voice narration** -- Assign different Kokoro voices to different chapters via JSON config
- **Text preprocessing** -- Expand abbreviations, numbers, ordinals, Roman numerals, currency, and percentages to spoken English before synthesis
- **Custom pronunciation** -- User-defined word substitutions via JSON dictionary
- **Format conversion** -- PDF, MOBI, AZW3, DOCX, FB2, and RTF input via Calibre (optional)
- **LLM-enhanced narration** -- Optional Ollama integration for OCR error correction and dialogue detection

---

## Prerequisites

### System dependencies (required)

```bash
sudo apt install espeak-ng ffmpeg
```

- **espeak-ng** -- Phoneme backend for Kokoro TTS
- **ffmpeg** (includes ffprobe) -- Audio encoding and M4B assembly

### Python

- Python 3.10, 3.11, or 3.12
- A virtual environment is strongly recommended

### Optional

- **Calibre** (`ebook-convert`) -- Required only for PDF/MOBI/AZW3/DOCX/FB2/RTF input. Install with `sudo apt install calibre`.
- **Ollama** -- Required only for LLM-enhanced preprocessing. Install from [ollama.com](https://ollama.com/download), then `ollama pull llama3.2`.
- **NVIDIA GPU + CUDA** -- Not required but provides roughly 8x speedup over CPU.

---

## Quick Start

```bash
# Clone and enter the project
cd audiblez-kokoro-pipeline

# Run setup (checks system deps, creates venv, installs packages)
bash setup.sh

# Activate the virtual environment
source .venv/bin/activate

# Convert a single EPUB
python -m src.batch_convert -i ./input -o ./output -v af_sky
```

Or place `.epub` files in `input/` and run batch mode:

```bash
python -m src.batch_convert -i input/ -o output/
```

---

## CLI Reference

The main entry point is `src.batch_convert`. All examples assume the venv is activated.

### Single file conversion (via batch mode with one file)

```bash
# Convert one book with default voice (af_sky)
python -m src.batch_convert -i input/ -o output/

# Use a specific voice
python -m src.batch_convert -i input/ -o output/ -v am_adam

# Adjust speed (1.0 = normal, 0.8 = slower, 1.2 = faster)
python -m src.batch_convert -i input/ -o output/ -v af_heart -s 1.1
```

### Batch processing

```bash
# Convert all EPUBs in a directory
python -m src.batch_convert -i ./books -o ./audiobooks

# Skip already-converted files (default behavior)
python -m src.batch_convert -i ./books -o ./audiobooks

# Force re-conversion of everything
python -m src.batch_convert -i ./books -o ./audiobooks --no-skip

# Dry run: parse and show info without synthesizing
python -m src.batch_convert -i ./books -o ./audiobooks --dry-run
```

### Multi-voice narration

```bash
# Use a voice config file for per-chapter voice assignment
python -m src.batch_convert -i input/ -o output/ --voice-config config/voice_config.json
```

### Text preprocessing options

```bash
# Disable text preprocessing
python -m src.batch_convert -i input/ -o output/ --no-preprocess

# Use custom pronunciation substitutions
python -m src.batch_convert -i input/ -o output/ --substitutions config/substitutions.json
```

### Performance options

```bash
# Force CPU-only mode (disable CUDA)
python -m src.batch_convert -i input/ -o output/ --cpu
```

### All flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input-dir` | `-i` | `input/` | Directory containing `.epub` files |
| `--output-dir` | `-o` | `output/` | Directory for `.m4b` output files |
| `--voice` | `-v` | `af_sky` | Kokoro voice identifier |
| `--speed` | `-s` | `1.0` | Playback speed multiplier |
| `--voice-config` | | None | Path to multi-voice JSON config |
| `--no-preprocess` | | False | Disable text preprocessing |
| `--substitutions` | | None | Path to JSON pronunciation substitutions |
| `--cpu` | | False | Force CPU-only synthesis |
| `--no-skip` | | False | Re-convert even if output exists |
| `--dry-run` | | False | Parse and display info only |

---

## Multi-Voice Configuration

Create a `voice_config.json` to assign different voices to different chapters:

```json
{
  "default_voice": "af_sky",
  "default_speed": 1.0,
  "chapters": {
    "default": {
      "voice": "af_sky",
      "speed": 1.0
    },
    "0": {
      "voice": "am_adam",
      "speed": 0.9
    },
    "1-3": {
      "voice": "af_heart",
      "speed": 1.0
    },
    "7": {
      "voice": "bf_emma",
      "speed": 1.1
    }
  }
}
```

### Resolution order

1. **Exact match** -- `"0"` matches chapter index 0
2. **Range match** -- `"1-3"` matches chapters 1, 2, and 3 (first matching range wins)
3. **Default entry** -- `"default"` inside `chapters`
4. **Top-level fallback** -- `default_voice` and `default_speed`

### Voice naming convention

Kokoro voice names follow the pattern `{lang}{gender}_{name}`:

- First character: language (`a` = American English, `b` = British English, etc.)
- Second character: gender (`f` = female, `m` = male)
- After underscore: voice name

Example: `af_sky` = American female, voice "Sky"

---

## Custom Pronunciation

Create a `substitutions.json` for words the TTS engine mispronounces:

```json
{
  "Tolkien": "TOLL-keen",
  "Cthulhu": "kuh-THOO-loo",
  "Hermione": "her-MY-oh-nee",
  "Nguyen": "WIN",
  "GIF": "jif"
}
```

Each key is matched as a whole word (case-sensitive) and replaced before synthesis.

### Built-in expansions

The preprocessing pipeline automatically handles:

- **Abbreviations** -- `Dr.` -> `Doctor`, `Mr.` -> `Mister`, `etc.` -> `et cetera`, month abbreviations, and more
- **Numbers** -- `42` -> `forty-two`, `$3.50` -> `three dollars and fifty cents`, `85%` -> `eighty-five percent`
- **Ordinals** -- `1st` -> `first`, `21st` -> `twenty-first`
- **Years in context** -- `in 1984` -> `in nineteen eighty-four`
- **Roman numerals after keywords** -- `Chapter XIV` -> `Chapter fourteen`
- **Punctuation normalization** -- Curly quotes straightened, double hyphens to em-dashes, triple dots to ellipsis

---

## Batch Processing

### Basic usage

```bash
python -m src.batch_convert -i ./books -o ./audiobooks
```

This will:

1. Scan `./books` for all `.epub` files
2. Skip any that already have a corresponding `.m4b` in `./audiobooks`
3. Convert remaining files sequentially with rich progress bars
4. Print a summary table on completion

### Skip existing

By default, files with existing output are skipped. Use `--no-skip` to force re-conversion.

### Dry run

```bash
python -m src.batch_convert -i ./books -o ./audiobooks --dry-run
```

Parses all EPUBs and displays a table with title, author, chapter count, word count, and estimated duration -- without synthesizing any audio.

---

## LLM-Enhanced Narration

Optional preprocessing using a local Ollama LLM to improve narration quality.

### Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# Start the server
ollama serve
```

### Available modes

| Mode | Purpose |
|------|---------|
| `enhance` | Fix OCR errors, expand unusual abbreviations, improve readability |
| `dialogue_detect` | Identify dialogue passages and speakers for multi-voice assignment |

### Programmatic usage

```python
from src.llm_pipeline import LLMPipeline

pipeline = LLMPipeline(model='llama3.2')
enhanced_text = pipeline.process(raw_text, mode='enhance')
```

The LLM pipeline is entirely optional. The main pipeline works without it.

---

## Architecture

### Module dependency graph

```
src/
  epub_parser.py       EPUB parsing, Chapter/BookMetadata dataclasses
       |
  preprocess.py        Text normalization for TTS (abbreviations, numbers, etc.)
       |
  kokoro_direct.py     Direct Kokoro TTS wrapper (synthesize text/chapter/book)
       |
  audio_utils.py       ffmpeg wrappers: WAV concat, chapter metadata, M4B assembly
       |
  batch_convert.py     Batch orchestration, CLI entry point, progress display
       |
  multi_voice.py       Per-chapter voice assignment from JSON config
       |
  format_wrapper.py    PDF/MOBI/etc. to EPUB conversion via Calibre (optional)
       |
  llm_pipeline.py      Ollama LLM preprocessing (optional)
```

### Data flow

```
.epub / .pdf / .mobi
       |
       v
 [format_wrapper]     (optional: convert non-EPUB to EPUB via Calibre)
       |
       v
 [epub_parser]        Parse EPUB -> BookMetadata + list[Chapter]
       |
       v
 [preprocess]         Expand abbreviations, numbers, punctuation
       |
       v
 [llm_pipeline]       (optional: LLM-enhanced text cleaning via Ollama)
       |
       v
 [kokoro_direct]      Synthesize text -> WAV audio per chapter (24kHz mono)
       |
       v
 [audio_utils]        Concatenate WAVs, create chapter metadata, encode M4B
       |
       v
    .m4b              Final audiobook with chapter markers and optional cover art
```

---

## Performance

Expected synthesis speeds (English text, single voice):

| Device | Speed | Example |
|--------|-------|---------|
| NVIDIA GPU (CUDA) | ~500 chars/sec | 100K-char book in ~3 minutes |
| CPU only | ~60 chars/sec | 100K-char book in ~28 minutes |

Performance varies with text complexity, voice selection, and hardware. GPU acceleration requires CUDA-compatible NVIDIA drivers and PyTorch with CUDA support.

---

## Troubleshooting

### `espeak-ng not found`

```bash
sudo apt install espeak-ng
```

On macOS: `brew install espeak-ng`

### `ffmpeg not found`

```bash
sudo apt install ffmpeg
```

The pipeline requires both `ffmpeg` and `ffprobe` (bundled together in most distributions).

### `kokoro` import error

Ensure you installed kokoro in the correct order. The setup script handles this, but manually:

```bash
pip install "kokoro>=0.9.4"
pip install audiblez --no-deps
```

Audiblez 0.4.9 pins `kokoro<0.8.0` but Kokoro 0.9.4 works correctly. The `--no-deps` flag avoids the version conflict.

### CUDA not detected

Verify PyTorch sees your GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

If `False`, reinstall PyTorch with CUDA support per [pytorch.org](https://pytorch.org/get-started/locally/).

### `No chapters extracted` from EPUB

The parser filters chapters shorter than 200 characters. Some EPUBs with unusual structure (deeply nested navigation, image-heavy content) may produce no extractable chapters. Check the EPUB with Calibre's viewer first.

### PDF conversion quality is poor

PDF-to-EPUB conversion quality depends on the PDF structure. Scanned/image-based PDFs produce poor results. Text-based PDFs with simple layouts convert best. Always prefer the original EPUB if available.

### LLM pipeline errors

Verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

If not, start it with `ollama serve`. Ensure the model is pulled: `ollama pull llama3.2`.

### DRM-protected EPUBs

DRM-protected EPUB files cannot be processed. The parser will fail on encrypted content. Remove DRM before conversion (this is a legal gray area depending on jurisdiction).

---

## Available Voices

Kokoro-82M ships with voices for multiple languages. The most commonly used English voices:

### American English

| Female | Male |
|--------|------|
| `af_alloy` | `am_adam` |
| `af_aoede` | `am_echo` |
| `af_bella` | `am_eric` |
| `af_heart` | `am_fenrir` |
| `af_jessica` | `am_liam` |
| `af_kore` | `am_michael` |
| `af_nicole` | `am_onyx` |
| `af_nova` | `am_puck` |
| `af_river` | `am_santa` |
| `af_sarah` | |
| `af_sky` | |

### British English

| Female | Male |
|--------|------|
| `bf_alice` | `bm_daniel` |
| `bf_emma` | `bm_fable` |
| `bf_isabella` | `bm_george` |
| `bf_lily` | `bm_lewis` |

### Other languages

Kokoro supports Spanish (`e`), French (`f`), Hindi (`h`), Italian (`i`), Japanese (`j`), Portuguese (`p`), and Chinese (`z`). Voice names follow the same `{lang}{gender}_{name}` pattern. See the [Kokoro model card](https://huggingface.co/hexgrad/Kokoro-82M) for the full voice list.

---

## License

This pipeline is provided as-is. Kokoro-82M is licensed under Apache 2.0. Audiblez is licensed under MIT. Check individual component licenses for redistribution requirements.
