"""
Integration tests for the audiblez-kokoro-pipeline.

Verifies that modules work together as a pipeline. Tests marked with
@pytest.mark.integration require runtime dependencies (kokoro, torch,
espeak-ng, ffmpeg) and are skipped by default in CI.

Run all tests:         pytest tests/test_integration.py -v
Run only unit-level:   pytest tests/test_integration.py -v -m "not integration"
Run only integration:  pytest tests/test_integration.py -v -m integration
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure the src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Project root (audiblez-kokoro-pipeline/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
INPUT_DIR = PROJECT_ROOT / "input"


# ===================================================================
# Module import checks
# ===================================================================


class TestModuleImports:
    """Verify all src modules can be imported without errors.

    Modules that depend only on stdlib + lightweight deps (ebooklib, bs4,
    json, re, etc.) must always import. Modules that depend on heavy
    runtime deps (kokoro, torch, numpy, soundfile) are tested separately
    and allowed to fail with ImportError if those deps are missing.
    """

    def test_preprocess_pipeline_imports(self):
        """Import core modules that should work without kokoro/torch."""
        import epub_parser
        import preprocess
        import multi_voice
        import format_wrapper
        import llm_pipeline

        # epub_parser
        assert hasattr(epub_parser, "Chapter")
        assert hasattr(epub_parser, "BookMetadata")
        assert hasattr(epub_parser, "parse_epub")
        assert hasattr(epub_parser, "extract_text_from_html")
        assert hasattr(epub_parser, "detect_language")

        # preprocess
        assert hasattr(preprocess, "preprocess")
        assert hasattr(preprocess, "expand_abbreviations")
        assert hasattr(preprocess, "expand_numbers")
        assert hasattr(preprocess, "expand_roman_numerals")
        assert hasattr(preprocess, "normalize_punctuation")
        assert hasattr(preprocess, "load_substitutions")
        assert hasattr(preprocess, "apply_custom_substitutions")

        # multi_voice
        assert hasattr(multi_voice, "VoiceManager")
        assert hasattr(multi_voice, "VoiceAssignment")
        assert hasattr(multi_voice, "resolve_voice")
        assert hasattr(multi_voice, "load_voice_config")

        # format_wrapper
        assert hasattr(format_wrapper, "ensure_epub")
        assert hasattr(format_wrapper, "convert_to_epub")
        assert hasattr(format_wrapper, "check_calibre")

        # llm_pipeline
        assert hasattr(llm_pipeline, "LLMPipeline")
        assert hasattr(llm_pipeline, "check_ollama")
        assert hasattr(llm_pipeline, "enhance_for_narration")
        assert hasattr(llm_pipeline, "detect_dialogue")

    def test_runtime_dep_modules_importable(self):
        """Import modules that need numpy/soundfile/kokoro/rich.

        These raise ImportError if the runtime deps are missing, which is
        acceptable -- the test verifies there are no *syntax* errors or
        broken relative imports beyond the expected missing packages.
        """
        import_errors = []

        for module_name in ("audio_utils", "kokoro_direct", "batch_convert"):
            try:
                __import__(module_name)
            except ImportError as exc:
                # Expected if kokoro/soundfile/numpy/rich not installed
                import_errors.append((module_name, str(exc)))
            except Exception as exc:
                # Unexpected error (syntax, attribute, etc.) -- fail hard
                pytest.fail(
                    f"Unexpected error importing {module_name}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if import_errors:
            names = [name for name, _ in import_errors]
            pytest.skip(
                f"Skipped runtime-dep modules (missing packages): {', '.join(names)}"
            )

        # If all imported, verify key symbols
        import audio_utils
        import batch_convert

        assert hasattr(audio_utils, "check_ffmpeg")
        assert hasattr(audio_utils, "create_m4b")
        assert hasattr(audio_utils, "concat_wavs")
        assert hasattr(audio_utils, "probe_duration")

        assert hasattr(batch_convert, "batch_convert")
        assert hasattr(batch_convert, "find_epub_files")


# ===================================================================
# Config file validation
# ===================================================================


class TestConfigFiles:
    """Verify shipped config files are valid and well-formed."""

    def test_substitutions_json_valid(self):
        """Load config/substitutions.json and verify it is a valid dict."""
        subs_path = CONFIG_DIR / "substitutions.json"
        assert subs_path.exists(), f"substitutions.json not found at {subs_path}"

        with open(subs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict), "substitutions.json must be a JSON object"
        assert len(data) > 0, "substitutions.json should not be empty"

        # All keys and values must be strings
        for key, value in data.items():
            assert isinstance(key, str), f"Key {key!r} is not a string"
            assert isinstance(value, str), f"Value for {key!r} is not a string"

    def test_voice_config_json_valid(self):
        """Load config/voice_config.json and verify structure."""
        config_path = CONFIG_DIR / "voice_config.json"
        assert config_path.exists(), f"voice_config.json not found at {config_path}"

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict), "voice_config.json must be a JSON object"
        assert "default_voice" in data, "Missing required key 'default_voice'"
        assert isinstance(data["default_voice"], str)

        if "default_speed" in data:
            assert isinstance(data["default_speed"], (int, float))
            assert data["default_speed"] > 0

        if "chapters" in data:
            assert isinstance(data["chapters"], dict)

        # Also validate through the actual module's loader
        from multi_voice import load_voice_config

        config = load_voice_config(config_path)
        assert config["default_voice"] == data["default_voice"]


# ===================================================================
# Cross-module pipeline tests
# ===================================================================


class TestEpubParserHtmlExtraction:
    """Test the epub_parser -> Chapter pipeline with sample HTML."""

    def test_epub_parser_html_extraction(self):
        """Full chapter pipeline: HTML -> Chapter with text, word count, duration."""
        from epub_parser import Chapter, extract_text_from_html, detect_language

        sample_html = """
        <html>
        <head><title>The Storm</title></head>
        <body>
            <h1>Chapter One: The Storm</h1>
            <p>Dr. Smith stood at the window, watching the rain pour down in
            sheets across the darkened landscape. It was the 3rd night of
            the storm, and supplies were running low.</p>
            <p>"We need to leave," Maria said, her voice barely audible
            above the howling wind. "The river is rising."</p>
            <p>He turned to face her. In 1984, during the last great flood,
            his father had made the same decision. That time, 42 people
            had not survived the crossing.</p>
            <p>The clock read $3.50 worth of nothing -- time was the one
            currency they couldn't spend.</p>
        </body>
        </html>
        """

        # Step 1: Extract text from HTML
        text = extract_text_from_html(sample_html)
        assert len(text) > 0
        assert "Dr. Smith" in text
        assert "Maria said" in text

        # Step 2: Create a Chapter from the HTML
        chapter = Chapter.from_html(index=0, title="Chapter One: The Storm", html=sample_html)
        assert chapter.index == 0
        assert chapter.title == "Chapter One: The Storm"
        assert chapter.word_count > 30
        assert chapter.estimated_duration_sec > 0
        assert "<h1>" in chapter.html  # Original HTML preserved

        # Step 3: Detect language
        lang = detect_language(chapter.text)
        assert lang == "a"  # American English


class TestPreprocessFullPipeline:
    """Test the full preprocessing pipeline on realistic text."""

    def test_preprocess_full_pipeline(self):
        """Run raw chapter text through the entire preprocess pipeline."""
        from preprocess import preprocess

        raw_text = (
            "Dr. Smith arrived on Jan. 5 in 1984. "
            "He spent $3.50 on the 1st edition of Volume XIV. "
            "Approx. 85% of readers, i.e. most people, loved it. "
            'Prof. Oak said "well--you know..." about Chapter III.'
        )

        result = preprocess(raw_text, substitutions_path=CONFIG_DIR / "substitutions.json")

        # Abbreviations expanded
        assert "Doctor Smith" in result
        assert "January" in result
        assert "approximately" in result
        assert "Professor" in result
        assert "that is" in result

        # Numbers expanded
        assert "three dollars and fifty cents" in result
        assert "eighty-five percent" in result
        assert "first edition" in result

        # Roman numerals expanded (after keyword)
        assert "Volume fourteen" in result
        assert "Chapter three" in result

        # Years expanded
        assert "nineteen eighty-four" in result

        # Punctuation normalized
        assert "\u2014" in result  # em-dash from --
        assert "\u2026" in result  # ellipsis from ...

        # Original abbreviations gone
        assert "Dr." not in result
        assert "Jan." not in result
        assert "Prof." not in result
        assert "i.e." not in result

    def test_preprocess_then_chapter_roundtrip(self):
        """Preprocess text, then verify it still forms a valid Chapter."""
        from epub_parser import Chapter
        from preprocess import preprocess

        html = "<h1>Test</h1><p>Dr. Smith scored 85% on the 1st exam in 1984.</p>"
        chapter = Chapter.from_html(index=0, title="Test", html=html)

        processed = preprocess(chapter.text)
        assert "Doctor" in processed
        assert "eighty-five percent" in processed
        assert len(processed) > len(chapter.text)  # Expanded text is longer


# ===================================================================
# Audio utilities
# ===================================================================


class TestAudioUtilsCheckFfmpeg:
    """Test the ffmpeg availability check."""

    def test_audio_utils_check_ffmpeg(self):
        """Verify check_ffmpeg returns a boolean."""
        from audio_utils import check_ffmpeg

        result = check_ffmpeg()
        assert isinstance(result, bool)


# ===================================================================
# Integration tests (require runtime dependencies)
# ===================================================================


@pytest.mark.integration
class TestKokoroHelloWorld:
    """Requires kokoro, torch, and espeak-ng to be installed."""

    def test_kokoro_hello_world(self):
        """Synthesize 'Hello world' and verify the output is a valid audio array."""
        import numpy as np

        from kokoro_direct import init_pipeline, synthesize_text, SAMPLE_RATE

        pipeline = init_pipeline(lang_code="a", device="cpu")
        audio = synthesize_text(pipeline, "Hello world.", voice="af_sky", speed=1.0)

        # Output should be a numpy array
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32

        # Should contain actual audio data (not empty)
        assert len(audio) > 0

        # At 24kHz, "Hello world" should produce at least 0.3 seconds of audio
        duration_sec = len(audio) / SAMPLE_RATE
        assert duration_sec > 0.3, f"Audio too short: {duration_sec:.2f}s"

        # Audio values should be in a reasonable range for normalized float32
        assert audio.max() <= 1.5, f"Audio max {audio.max()} seems too high"
        assert audio.min() >= -1.5, f"Audio min {audio.min()} seems too low"


@pytest.mark.integration
class TestFullPipelineTinyEpub:
    """Requires all runtime dependencies (kokoro, torch, ffmpeg, espeak-ng).

    Runs the full pipeline on a test EPUB if one exists in the input/ directory.
    """

    def test_full_pipeline_tiny_epub(self, tmp_path):
        """If a test epub exists in input/, run the full pipeline end to end."""
        epub_files = sorted(INPUT_DIR.glob("*.epub"))

        if not epub_files:
            pytest.skip("No .epub files found in input/ directory")

        test_epub = epub_files[0]

        from epub_parser import parse_epub
        from preprocess import preprocess as preprocess_text
        from kokoro_direct import init_pipeline, synthesize_chapter, SAMPLE_RATE
        from audio_utils import check_ffmpeg, create_m4b

        # Step 1: Parse
        metadata, chapters = parse_epub(test_epub)
        assert len(chapters) > 0, f"No chapters in {test_epub.name}"

        # Use only the first chapter to keep the test fast
        chapter = chapters[0]

        # Step 2: Preprocess
        processed_text = preprocess_text(chapter.text)
        assert len(processed_text) > 0

        # Truncate to first 500 characters for speed
        truncated = processed_text[:500]

        # Step 3: Synthesize
        pipeline = init_pipeline(lang_code=metadata.language, device="cpu")
        wav_path = tmp_path / "chapter_00.wav"
        synthesize_chapter(
            pipeline=pipeline,
            text=truncated,
            voice="af_sky",
            speed=1.0,
            output_path=wav_path,
        )

        assert wav_path.exists()
        assert wav_path.stat().st_size > 0

        # Step 4: Assemble M4B (only if ffmpeg is available)
        if not check_ffmpeg():
            pytest.skip("ffmpeg not available; skipping M4B assembly")

        m4b_path = tmp_path / "test_output.m4b"
        create_m4b(
            chapter_wav_files=[wav_path],
            chapter_titles=[chapter.title],
            output_path=m4b_path,
        )

        assert m4b_path.exists()
        assert m4b_path.stat().st_size > 0
