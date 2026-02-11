# audiblez-kokoro-pipeline

Local EPUB-to-audiobook converter. Takes `.epub` files (or PDF/MOBI via Calibre) and produces `.m4b` audiobooks with chapter markers and cover art, using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) for text-to-speech.

Runs entirely on your machine. GPU-accelerated when CUDA is available, works on CPU otherwise.

## Quick start

```bash
git clone https://github.com/Wondermonger-daydreaming/audiblez-kokoro-pipeline.git
cd audiblez-kokoro-pipeline

# Install system deps (Ubuntu/Debian)
sudo apt install espeak-ng ffmpeg

# Run setup (creates venv, installs Python packages)
bash setup.sh
source .venv/bin/activate

# Convert a book
python -m src.batch_convert -i input/ -o output/
```

Drop `.epub` files in `input/` and audiobooks appear in `output/`.

## Features

- **Chapter markers** — M4B output with per-chapter navigation and optional cover art
- **Batch processing** — Convert a directory of books in one command, with skip-existing and dry-run modes
- **Multi-voice narration** — Assign different voices to different chapters via JSON config
- **Text preprocessing** — Abbreviations, numbers, ordinals, Roman numerals, currency, and punctuation automatically expanded to natural speech
- **Custom pronunciation** — JSON dictionary for words the TTS mispronounces
- **Format conversion** — PDF, MOBI, AZW3, DOCX, FB2, RTF input via Calibre (optional)
- **LLM preprocessing** — Optional Ollama integration for OCR cleanup and dialogue detection

## Requirements

| Dependency | Required | Install |
|------------|----------|---------|
| Python 3.10+ | Yes | — |
| espeak-ng | Yes | `sudo apt install espeak-ng` |
| ffmpeg | Yes | `sudo apt install ffmpeg` |
| Calibre | No | `sudo apt install calibre` (for PDF/MOBI input) |
| Ollama | No | [ollama.com](https://ollama.com/download) (for LLM preprocessing) |
| NVIDIA GPU | No | ~8x faster with CUDA, works fine on CPU |

## Usage

### Basic conversion

```bash
# Single voice (default: af_sky)
python -m src.batch_convert -i input/ -o output/

# Choose a voice and speed
python -m src.batch_convert -i input/ -o output/ -v am_adam -s 1.1

# Dry run — parse books and show info without synthesizing
python -m src.batch_convert -i input/ -o output/ --dry-run

# Force re-conversion of already-converted books
python -m src.batch_convert -i input/ -o output/ --no-skip

# CPU-only mode
python -m src.batch_convert -i input/ -o output/ --cpu
```

### Multi-voice narration

Create a voice config to assign different voices per chapter:

```json
{
  "default_voice": "af_sky",
  "default_speed": 1.0,
  "chapters": {
    "default": { "voice": "af_sky", "speed": 1.0 },
    "0":       { "voice": "am_adam", "speed": 0.9 },
    "1-3":     { "voice": "af_heart", "speed": 1.0 }
  }
}
```

```bash
python -m src.batch_convert -i input/ -o output/ --voice-config config/voice_config.json
```

Chapters resolve in order: exact match (`"0"`) > range match (`"1-3"`) > `"default"` entry > top-level `default_voice`.

### Custom pronunciation

For words the TTS engine gets wrong, create a substitutions file:

```json
{
  "Tolkien": "TOLL-keen",
  "Cthulhu": "kuh-THOO-loo",
  "Hermione": "her-MY-oh-nee"
}
```

```bash
python -m src.batch_convert -i input/ -o output/ --substitutions config/substitutions.json
```

### Built-in text preprocessing

The pipeline automatically expands before synthesis:

- `Dr.` → `Doctor`, `Mr.` → `Mister`, `etc.` → `et cetera`
- `42` → `forty-two`, `$3.50` → `three dollars and fifty cents`
- `1st` → `first`, `21st` → `twenty-first`
- `in 1984` → `in nineteen eighty-four`
- `Chapter XIV` → `Chapter fourteen`
- Curly quotes straightened, double hyphens to em-dashes

Disable with `--no-preprocess`.

## CLI reference

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input-dir` | `-i` | `input/` | Directory containing `.epub` files |
| `--output-dir` | `-o` | `output/` | Directory for `.m4b` output |
| `--voice` | `-v` | `af_sky` | Kokoro voice identifier |
| `--speed` | `-s` | `1.0` | Speed multiplier (0.5–2.0) |
| `--voice-config` | | — | Multi-voice JSON config path |
| `--substitutions` | | — | Pronunciation substitutions JSON path |
| `--no-preprocess` | | `false` | Skip text preprocessing |
| `--cpu` | | `false` | Force CPU-only synthesis |
| `--no-skip` | | `false` | Re-convert existing outputs |
| `--dry-run` | | `false` | Parse and display info only |

## Voices

Voice names follow `{lang}{gender}_{name}`:

| | American female | American male | British female | British male |
|---|---|---|---|---|
| | `af_alloy` | `am_adam` | `bf_alice` | `bm_daniel` |
| | `af_aoede` | `am_echo` | `bf_emma` | `bm_fable` |
| | `af_bella` | `am_eric` | `bf_isabella` | `bm_george` |
| | `af_heart` | `am_fenrir` | `bf_lily` | `bm_lewis` |
| | `af_jessica` | `am_liam` | | |
| | `af_kore` | `am_michael` | | |
| | `af_nicole` | `am_onyx` | | |
| | `af_nova` | `am_puck` | | |
| | `af_river` | `am_santa` | | |
| | `af_sarah` | | | |
| | `af_sky` | | | |

Also supports Spanish (`e`), French (`f`), Hindi (`h`), Italian (`i`), Japanese (`j`), Portuguese (`p`), and Chinese (`z`). See the [Kokoro model card](https://huggingface.co/hexgrad/Kokoro-82M) for full list.

## Architecture

```
.epub/.pdf/.mobi
      │
      ▼
 format_wrapper ──── (optional: convert to EPUB via Calibre)
      │
      ▼
 epub_parser ─────── Parse EPUB → BookMetadata + list[Chapter]
      │
      ▼
 preprocess ──────── Expand abbreviations, numbers, punctuation
      │
      ▼
 llm_pipeline ────── (optional: LLM text cleaning via Ollama)
      │
      ▼
 kokoro_direct ───── Synthesize text → WAV per chapter (24kHz mono)
      │
      ▼
 audio_utils ─────── Concatenate WAVs, chapter metadata, encode M4B
      │
      ▼
    .m4b ─────────── Audiobook with chapters and cover art
```

## Performance

| Device | Speed | 100K-char book |
|--------|-------|----------------|
| NVIDIA GPU (CUDA) | ~500 chars/sec | ~3 min |
| CPU only | ~60 chars/sec | ~28 min |

## Troubleshooting

**`espeak-ng not found`** — `sudo apt install espeak-ng` (macOS: `brew install espeak-ng`)

**`ffmpeg not found`** — `sudo apt install ffmpeg`

**`kokoro` import error** — Install order matters. The setup script handles this, but manually: `pip install "kokoro>=0.9.4"` first, then `pip install audiblez --no-deps`.

**CUDA not detected** — Check with `python -c "import torch; print(torch.cuda.is_available())"`. If false, reinstall PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).

**No chapters extracted** — The parser filters chapters < 200 chars. Some EPUBs with unusual structure may need Calibre's viewer to verify.

**DRM-protected EPUBs** — Cannot be processed. DRM must be removed first.

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## License

[MIT](LICENSE)
