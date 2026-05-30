# audiblez-kokoro-pipeline

Local EPUB-to-audiobook converter. Takes `.epub` files (or PDF/MOBI/AZW3/DOCX/FB2/RTF via Calibre) and produces `.m4b` audiobooks with chapter markers and cover art, using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) for text-to-speech.

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

# Convert a single book
python -m src.pipeline book.epub -o output/

# ...or drop .epub files in input/ and batch-convert the directory
python -m src.pipeline --batch input/ -o output/
```

Audiobooks appear in `output/` as `<Book Title>.m4b`.

## Features

- **Chapter markers** — M4B output with per-chapter navigation and optional cover art
- **Single-file or batch** — Convert one book, or a whole directory in one command (with skip-existing and dry-run modes)
- **Format conversion** — PDF, MOBI, AZW3, DOCX, FB2, RTF input via Calibre (optional)
- **Multi-voice narration** — Assign different voices to different chapters via JSON config
- **Text preprocessing** — Abbreviations, numbers, ordinals, Roman numerals, currency, and punctuation automatically expanded to natural speech
- **Custom pronunciation** — JSON dictionary for words the TTS mispronounces
- **LLM preprocessing** — Optional Ollama integration for OCR cleanup and dialogue detection

## Requirements

| Dependency | Required | Install |
|------------|----------|---------|
| Python 3.10+ | Yes | — |
| espeak-ng | Yes | `sudo apt install espeak-ng` |
| ffmpeg (+ ffprobe) | Yes | `sudo apt install ffmpeg` |
| Calibre | No | `sudo apt install calibre` (for PDF/MOBI/AZW3/… input) |
| Ollama | No | [ollama.com](https://ollama.com/download) (for `--llm` preprocessing) |
| NVIDIA GPU | No | Much faster with CUDA; works fine on CPU |

## The CLI

`python -m src.pipeline` is the main entry point (`python -m src` runs the same thing). It handles single files, format conversion, batch mode, multi-voice, preprocessing, and optional LLM enhancement.

```bash
python -m src.pipeline --help
```

> Tip: `--help` and `--dry-run` work without the `kokoro`/`torch` stack installed — the TTS engine is only imported when synthesis actually runs.

### Single file

```bash
# Defaults: voice af_sky, speed 1.0, auto device
python -m src.pipeline book.epub -o output/

# Pick a voice and speed
python -m src.pipeline book.epub -v am_adam -s 1.1

# Non-EPUB formats are converted to EPUB first (requires Calibre)
python -m src.pipeline document.pdf -o output/
```

### Batch a directory

```bash
python -m src.pipeline --batch input/ -o output/

# Parse and show info (chapters, words, estimated duration) without synthesizing
python -m src.pipeline --batch input/ --dry-run

# Re-convert books even if the .m4b already exists
python -m src.pipeline --batch input/ -o output/ --no-skip-existing
```

> **Note:** batch mode scans the input directory for `*.epub` only. To convert a PDF/MOBI/etc., run it through single-file mode (which converts to EPUB via Calibre first).

### Device selection

```bash
python -m src.pipeline book.epub --cuda   # force GPU
python -m src.pipeline book.epub --cpu    # force CPU
```

Without a flag the device is chosen automatically (CUDA if available, else CPU).

### Multi-voice narration

Assign different voices to different chapters with a JSON config:

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
python -m src.pipeline book.epub --voice-config config/voice_config.json
python -m src.pipeline --batch input/ --voice-config config/voice_config.json
```

Chapters resolve in order: exact match (`"0"`) → range match (`"1-3"`) → `"default"` entry → top-level `default_voice`. When ranges overlap, the first match wins.

### Custom pronunciation

For words the TTS engine gets wrong, supply a substitutions file:

```json
{
  "Tolkien": "TOLL-keen",
  "Cthulhu": "kuh-THOO-loo",
  "Hermione": "her-MY-oh-nee"
}
```

```bash
python -m src.pipeline book.epub --substitutions config/substitutions.json
```

If `--substitutions` is omitted, `config/substitutions.json` is used automatically when present.

### Built-in text preprocessing

Enabled by default; expands before synthesis:

- `Dr.` → `Doctor`, `Mr.` → `Mister`, `etc.` → `et cetera` (case-insensitive, so sentence-initial `Approx.` is handled too)
- `42` → `forty-two`, `$3.50` → `three dollars and fifty cents`, `85%` → `eighty-five percent`
- `1st` → `first`, `21st` → `twenty-first`
- `in 1984` → `in nineteen eighty-four`
- `Chapter XIV` → `Chapter fourteen`
- Curly quotes straightened, `--` to em-dashes, `...` to an ellipsis

Disable with `--no-preprocess`.

### LLM enhancement (optional)

With a local [Ollama](https://ollama.com/download) server running, run each chapter through an LLM to fix OCR errors and improve readability before synthesis:

```bash
ollama serve            # in another terminal
ollama pull llama3.2
python -m src.pipeline book.epub --llm --llm-model llama3.2
```

If Ollama is unreachable, the pipeline logs a warning and continues with the un-enhanced text.

## CLI reference

`python -m src.pipeline [input] [options]`

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `input` (positional) | | — | Input file (EPUB/PDF/MOBI/…), or a directory with `--batch` |
| `--output` | `-o` | `output/` | Output directory for `.m4b` files |
| `--voice` | `-v` | `af_sky` | Kokoro voice identifier |
| `--speed` | `-s` | `1.0` | Speed multiplier (0.5–2.0) |
| `--cuda` | `-c` | auto | Force CUDA acceleration |
| `--cpu` | | auto | Force CPU-only synthesis |
| `--batch` | `-b` | `false` | Treat `input` as a directory of `.epub` files |
| `--voice-config` | | — | Multi-voice JSON config path |
| `--no-preprocess` | | preprocessing on | Disable text preprocessing |
| `--substitutions` | | `config/substitutions.json` if present | Pronunciation substitutions JSON path |
| `--llm` | | `false` | Enable LLM enhancement via Ollama |
| `--llm-model` | | `llama3.2` | Ollama model for `--llm` |
| `--direct` | `-d` | `false` | Use direct Kokoro synthesis (bypass the audiblez wrapper) |
| `--no-skip-existing` | | skip-existing on | Re-convert even if the `.m4b` already exists |
| `--dry-run` | | `false` | Parse and display info without synthesizing |
| `--bitrate` | | `64k` | AAC bitrate for the M4B |

There is also a standalone, EPUB-only batch entry point with its own flag names (`-i/--input-dir`, `-o/--output-dir`, `--no-skip`):

```bash
python -m src.batch_convert -i input/ -o output/
```

`python -m src.pipeline --batch` is the recommended way to batch-convert.

## Voices

Voice names follow `{lang}{gender}_{name}`:

| American female | American male | British female | British male |
|---|---|---|---|
| `af_alloy` | `am_adam` | `bf_alice` | `bm_daniel` |
| `af_aoede` | `am_echo` | `bf_emma` | `bm_fable` |
| `af_bella` | `am_eric` | `bf_isabella` | `bm_george` |
| `af_heart` | `am_fenrir` | `bf_lily` | `bm_lewis` |
| `af_jessica` | `am_liam` | | |
| `af_kore` | `am_michael` | | |
| `af_nicole` | `am_onyx` | | |
| `af_nova` | `am_puck` | | |
| `af_river` | `am_santa` | | |
| `af_sarah` | | | |
| `af_sky` | | | |

Also supports Spanish (`e`), French (`f`), Hindi (`h`), Italian (`i`), Japanese (`j`), Portuguese (`p`), and Chinese (`z`). See the [Kokoro model card](https://huggingface.co/hexgrad/Kokoro-82M) for the full list. The language is auto-detected from the book text and, for multi-voice configs, derived from each voice's prefix.

## Architecture

```
.epub/.pdf/.mobi
      │
      ▼
 format_wrapper ──── (optional: convert to EPUB via Calibre)
      │
      ▼
 epub_parser ─────── Parse EPUB in spine (reading) order → BookMetadata + list[Chapter]
      │
      ▼
 preprocess ──────── Expand abbreviations, numbers, punctuation
      │
      ▼
 llm_pipeline ────── (optional: LLM text cleaning via Ollama)
      │
      ▼
 kokoro_direct ───── Synthesize text → WAV per chapter (24kHz mono)
 / multi_voice        (multi_voice resolves a per-chapter voice/speed)
      │
      ▼
 audio_utils ─────── Concatenate WAVs, write chapter metadata, encode M4B
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

**`ffmpeg not found`** — `sudo apt install ffmpeg` (provides both `ffmpeg` and `ffprobe`, both required)

**`kokoro` import error** — Install it with `pip install "kokoro>=0.9.4"`, and make sure `espeak-ng` is present (Kokoro uses it for phonemization). `setup.sh` does both.

**CUDA not detected** — Check with `python -c "import torch; print(torch.cuda.is_available())"`. If `False`, reinstall PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/), or pass `--cpu`.

**No chapters extracted** — The parser skips sections under ~200 characters. EPUBs with unusual structure may need Calibre's viewer to verify.

**DRM-protected EPUBs** — Cannot be processed. DRM must be removed first.

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

The unit and regression tests run without `kokoro`/`torch`; the synthesis integration tests are marked `integration` and are skipped automatically when `kokoro` is not installed. Run only the fast tests with `pytest -m "not integration"`.

## License

[MIT](LICENSE)
