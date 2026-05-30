"""Regression tests for three fixed defects.

Each test is written to FAIL against the pre-fix code and PASS after, and to
run without the heavy ``kokoro``/``torch`` stack or ffmpeg installed:

1. Batch mode + ``--voice-config`` raised ``TypeError`` because the multi-voice
   branch called ``convert_with_multi_voice()`` with arguments matching a
   different function's signature.
2. ``--cpu``/``--cuda`` were ignored in the single-voice / ``--direct`` paths:
   ``init_pipeline`` resolved a device but never passed it to ``KPipeline``.
3. Chapters were emitted in manifest order rather than spine (reading) order.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src import audio_utils, batch_convert, kokoro_direct
from src.epub_parser import parse_epub
from src.multi_voice import VoiceManager


def _build_epub(path: Path, chapters, *, manifest_reversed: bool = False) -> None:
    """Write a minimal EPUB. When *manifest_reversed*, items are added to the
    manifest in the reverse of their spine (reading) order."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("regression-book")
    book.set_title("Regression Book")
    book.set_language("en")
    book.add_author("Author")

    items = []
    for i, (title, body) in enumerate(chapters):
        c = epub.EpubHtml(title=title, file_name=f"chap_{i}.xhtml", lang="en")
        c.content = f"<html><body><h1>{title}</h1>{body}</body></html>"
        items.append(c)

    for c in (reversed(items) if manifest_reversed else items):
        book.add_item(c)

    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + items  # spine always declares true reading order
    epub.write_epub(str(path), book)


# ---------------------------------------------------------------------------
# 1. Batch multi-voice path no longer TypeErrors and resolves per-chapter voices
# ---------------------------------------------------------------------------


def test_batch_multi_voice_resolves_per_chapter(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # Padding must live *inside* a <p> so the parser's >200-char filter keeps
    # the chapter (loose text directly in <body> is not extracted).
    _build_epub(input_dir / "book.epub", [
        ("Intro", "<p>First chapter. " + "Pad. " * 80 + "</p>"),
        ("Body", "<p>Second chapter. " + "Pad. " * 80 + "</p>"),
    ])

    cfg = tmp_path / "voices.json"
    cfg.write_text(json.dumps({
        "default_voice": "af_sky",
        "default_speed": 1.0,
        "chapters": {
            "0": {"voice": "am_adam", "speed": 0.9},
            "1-3": {"voice": "af_heart", "speed": 1.1},
        },
    }))

    # Avoid loading a real Kokoro model.
    monkeypatch.setattr(
        VoiceManager, "_get_or_create_pipeline",
        lambda self, lang_code: object(),
    )

    calls: list[tuple[str, float]] = []

    def _stub_synth(pipeline, text, voice, speed, output_path, **kwargs):
        import soundfile as sf
        calls.append((voice, speed))
        sf.write(str(output_path), np.zeros(2400, dtype=np.float32), 24000)
        return Path(output_path)

    monkeypatch.setattr(kokoro_direct, "synthesize_chapter", _stub_synth)

    # No external binaries: pretend ffmpeg is present and stub M4B assembly.
    monkeypatch.setattr(audio_utils, "check_ffmpeg", lambda: True)
    monkeypatch.setattr(
        audio_utils, "create_m4b",
        lambda chapter_wav_files, chapter_titles, output_path, **kw: (
            Path(output_path).write_bytes(b"M4B") or Path(output_path)
        ),
    )

    results = batch_convert.batch_convert(
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        voice_config=cfg,
        use_preprocessing=False,
        use_cuda=False,
        skip_existing=False,
    )

    assert len(results) == 1
    # Pre-fix this was "error" (TypeError from convert_with_multi_voice).
    assert results[0]["status"] == "success"
    # Per-chapter voice/speed came from the config, not the global default.
    assert calls == [("am_adam", 0.9), ("af_heart", 1.1)]


# ---------------------------------------------------------------------------
# 2. init_pipeline forwards the resolved device to KPipeline
# ---------------------------------------------------------------------------


def test_init_pipeline_forwards_device(monkeypatch):
    captured: dict[str, object] = {}

    fake_kokoro = types.ModuleType("kokoro")

    class _FakeKPipeline:
        def __init__(self, lang_code, device=None, **kwargs):
            captured["lang_code"] = lang_code
            captured["device"] = device

    fake_kokoro.KPipeline = _FakeKPipeline
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)

    kokoro_direct.init_pipeline(lang_code="a", device="cpu")

    assert captured["lang_code"] == "a"
    # Pre-fix this was None (device computed but never passed).
    assert captured["device"] == "cpu"


# ---------------------------------------------------------------------------
# 3. parse_epub follows spine order even when the manifest is reversed
# ---------------------------------------------------------------------------


def test_parse_epub_follows_spine_not_manifest(tmp_path):
    epub_path = tmp_path / "divergent.epub"
    _build_epub(
        epub_path,
        [
            ("Alpha", "<p>" + "alpha " * 80 + "</p>"),
            ("Beta", "<p>" + "beta " * 80 + "</p>"),
        ],
        manifest_reversed=True,  # manifest = [Beta, Alpha]; spine = [Alpha, Beta]
    )

    _metadata, chapters = parse_epub(epub_path)

    # Pre-fix (manifest order) this would be ["Beta", "Alpha"].
    assert [c.title for c in chapters] == ["Alpha", "Beta"]
