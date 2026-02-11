# Audiblez + Kokoro-82M Pipeline

## Project overview
EPUB-to-audiobook pipeline. Converts epub/pdf/mobi to m4b audiobooks with chapter markers.

## Key files
- src/batch_convert.py -- Main CLI entry point (`python -m src.batch_convert`)
- src/epub_parser.py -- EPUB parsing, Chapter/BookMetadata dataclasses
- src/preprocess.py -- Text preprocessing for TTS (abbreviations, numbers, punctuation)
- src/kokoro_direct.py -- Direct Kokoro TTS wrapper (synthesize_text, synthesize_chapter, synthesize_book)
- src/multi_voice.py -- Per-chapter voice assignment from JSON config, VoiceManager class
- src/audio_utils.py -- ffmpeg wrappers: WAV concat, chapter metadata, M4B assembly
- src/format_wrapper.py -- PDF/MOBI conversion via Calibre (optional)
- src/llm_pipeline.py -- Optional Ollama LLM preprocessing (enhance, dialogue_detect)

## Architecture
epub -> epub_parser -> preprocess -> kokoro_direct -> audio_utils -> .m4b

## Development
- Venv: .venv/bin/activate
- Tests: pytest tests/ -v
- System deps: espeak-ng, ffmpeg
- Voice naming: {lang}{gender}_{name} (e.g. af_sky = American female Sky)

## Known limitations
- PDF conversion quality varies (Calibre limitation)
- LLM pipeline requires Ollama running locally
- DRM-protected EPUBs cannot be processed
- No standalone pipeline.py -- batch_convert.py serves as the CLI entry point
