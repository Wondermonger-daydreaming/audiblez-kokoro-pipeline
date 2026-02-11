"""
Per-chapter voice assignment engine for the audiblez-kokoro pipeline.

Allows different Kokoro TTS voices for different chapters via JSON config.
Supports exact chapter matching, range matching ("1-3"), and cascading
fallback to default entries. Caches KPipeline instances by lang_code to
avoid redundant model loads.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lang-code prefix table: first character of Kokoro voice name -> lang_code
# ---------------------------------------------------------------------------

_VOICE_PREFIX_TO_LANG: dict[str, str] = {
    "a": "a",  # American English
    "b": "b",  # British English
    "e": "e",  # Spanish
    "f": "f",  # French
    "h": "h",  # Hindi
    "i": "i",  # Italian
    "j": "j",  # Japanese
    "p": "p",  # Portuguese
    "z": "z",  # Chinese
}


# ---------------------------------------------------------------------------
# VoiceAssignment dataclass
# ---------------------------------------------------------------------------


@dataclass
class VoiceAssignment:
    """Resolved voice settings for a single chapter."""

    chapter_index: int
    voice: str
    speed: float
    lang_code: str


# ---------------------------------------------------------------------------
# Config loading and validation
# ---------------------------------------------------------------------------


def load_voice_config(config_path: Path) -> dict:
    """Load and validate a voice_config.json file.

    The config must contain at minimum a ``default_voice`` key at the
    top level.  The ``chapters`` mapping is optional; when present it
    should map string keys (exact indices like ``"0"`` or ranges like
    ``"1-3"``) to objects with ``voice`` and optionally ``speed``.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If required keys are missing or the JSON is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Voice config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        try:
            config = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in voice config {config_path}: {exc}"
            ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            f"Voice config must be a JSON object, got {type(config).__name__}"
        )

    if "default_voice" not in config:
        raise ValueError(
            "Voice config is missing required key 'default_voice'"
        )

    # Validate chapters entries when present
    chapters = config.get("chapters", {})
    if not isinstance(chapters, dict):
        raise ValueError(
            f"'chapters' must be a JSON object, got {type(chapters).__name__}"
        )

    for key, entry in chapters.items():
        if key == "default":
            continue
        # Validate that the key is either a non-negative integer or a
        # range like "1-3"
        if not _is_valid_chapter_key(key):
            raise ValueError(
                f"Invalid chapter key '{key}': must be a non-negative "
                f"integer or a range like '1-3'"
            )

    return config


def _is_valid_chapter_key(key: str) -> bool:
    """Return True if *key* is a valid chapter specifier.

    Valid forms:
    - A non-negative integer string: "0", "5", "12"
    - A range of non-negative integers: "1-3", "10-20"
    """
    if key.isdigit():
        return True

    parts = key.split("-", maxsplit=1)
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]) <= int(parts[1])

    return False


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def _derive_lang_code(voice_name: str) -> str:
    """Derive the Kokoro lang_code from the first character of a voice name.

    Kokoro voice names follow the pattern ``{lang}{gender}_{name}`` where
    ``{lang}`` is a single letter: a=American, b=British, etc.

    Falls back to ``'a'`` (American English) for unrecognised prefixes.

    Args:
        voice_name: A Kokoro voice identifier, e.g. ``"af_sky"``.

    Returns:
        Single-character language code.
    """
    if not voice_name:
        return "a"

    prefix = voice_name[0].lower()
    return _VOICE_PREFIX_TO_LANG.get(prefix, "a")


def resolve_voice(chapter_index: int, config: dict) -> VoiceAssignment:
    """Resolve which voice/speed to use for a given chapter.

    Resolution order:
    1. Exact chapter match in ``config["chapters"]`` (e.g. key ``"5"``).
    2. Range match in ``config["chapters"]`` (e.g. key ``"1-3"`` means
       chapters 1, 2, and 3).  When multiple ranges overlap, the first
       match encountered in iteration order wins.
    3. The ``"default"`` entry inside ``config["chapters"]``.
    4. Top-level ``default_voice`` / ``default_speed``.

    The ``lang_code`` is derived from the first character of the resolved
    voice name.

    Args:
        chapter_index: Zero-based chapter index.
        config: Parsed voice config dictionary.

    Returns:
        A VoiceAssignment for the chapter.
    """
    chapters = config.get("chapters", {})
    top_voice = config.get("default_voice", "af_sky")
    top_speed = config.get("default_speed", 1.0)

    # 1. Exact match
    key_exact = str(chapter_index)
    if key_exact in chapters:
        entry = chapters[key_exact]
        voice = entry.get("voice", top_voice)
        speed = entry.get("speed", top_speed)
        return VoiceAssignment(
            chapter_index=chapter_index,
            voice=voice,
            speed=speed,
            lang_code=_derive_lang_code(voice),
        )

    # 2. Range match (first match wins)
    for key, entry in chapters.items():
        if key == "default" or not isinstance(entry, dict):
            continue
        rng = _parse_range(key)
        if rng is not None:
            start, end = rng
            if start <= chapter_index <= end:
                voice = entry.get("voice", top_voice)
                speed = entry.get("speed", top_speed)
                return VoiceAssignment(
                    chapter_index=chapter_index,
                    voice=voice,
                    speed=speed,
                    lang_code=_derive_lang_code(voice),
                )

    # 3. "default" entry in chapters
    if "default" in chapters:
        entry = chapters["default"]
        voice = entry.get("voice", top_voice)
        speed = entry.get("speed", top_speed)
        return VoiceAssignment(
            chapter_index=chapter_index,
            voice=voice,
            speed=speed,
            lang_code=_derive_lang_code(voice),
        )

    # 4. Top-level fallback
    return VoiceAssignment(
        chapter_index=chapter_index,
        voice=top_voice,
        speed=top_speed,
        lang_code=_derive_lang_code(top_voice),
    )


def _parse_range(key: str) -> tuple[int, int] | None:
    """Parse a range key like ``"1-3"`` into (start, end) inclusive.

    Returns None if *key* is not a valid range (exact indices are not
    treated as ranges here).
    """
    parts = key.split("-", maxsplit=1)
    if len(parts) != 2:
        return None

    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError:
        return None

    if start > end:
        return None

    return (start, end)


# ---------------------------------------------------------------------------
# VoiceManager — lazy pipeline caching
# ---------------------------------------------------------------------------


class VoiceManager:
    """Manages Kokoro TTS pipelines and per-chapter voice assignments.

    Pipelines are created lazily on first use and cached by ``lang_code``
    so that chapters sharing the same language prefix reuse a single
    pipeline instance.

    Args:
        config_path: Path to voice_config.json.
        device: Torch device string (``'cpu'``, ``'cuda'``, or ``'auto'``).
            ``'auto'`` selects CUDA when available, otherwise CPU.
    """

    def __init__(self, config_path: Path, device: str = "auto"):
        self._config = load_voice_config(config_path)
        self._device = device
        self._pipelines: dict[str, object] = {}  # lang_code -> KPipeline

    @property
    def config(self) -> dict:
        """Return the loaded voice config dictionary."""
        return self._config

    def get_pipeline_and_voice(
        self, chapter_index: int
    ) -> tuple[object, str, float]:
        """Return the pipeline, voice name, and speed for a chapter.

        The KPipeline is imported and instantiated lazily the first time
        a particular ``lang_code`` is requested, then cached for reuse.

        Args:
            chapter_index: Zero-based chapter index.

        Returns:
            A 3-tuple of ``(pipeline, voice_name, speed)``.
        """
        assignment = resolve_voice(chapter_index, self._config)

        pipeline = self._get_or_create_pipeline(assignment.lang_code)

        return (pipeline, assignment.voice, assignment.speed)

    def _get_or_create_pipeline(self, lang_code: str) -> object:
        """Return a cached KPipeline for *lang_code*, creating if needed."""
        if lang_code in self._pipelines:
            return self._pipelines[lang_code]

        # Lazy import: KPipeline is only loaded when actually needed,
        # keeping the module importable in test environments without
        # the kokoro package installed.
        from kokoro import KPipeline  # type: ignore[import-untyped]

        device = self._resolve_device()

        logger.info(
            "Creating KPipeline for lang_code='%s' on device='%s'",
            lang_code,
            device,
        )
        pipeline = KPipeline(lang_code=lang_code, device=device)
        self._pipelines[lang_code] = pipeline
        return pipeline

    def _resolve_device(self) -> str:
        """Resolve ``'auto'`` to a concrete device string."""
        if self._device != "auto":
            return self._device

        try:
            import torch  # type: ignore[import-untyped]

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# ---------------------------------------------------------------------------
# Full multi-voice conversion pipeline
# ---------------------------------------------------------------------------


def convert_with_multi_voice(
    epub_path: Path,
    config_path: Path,
    output_dir: Path,
    use_preprocessing: bool = True,
    substitutions_path: Optional[Path] = None,
    device: str = "auto",
) -> Path:
    """Convert an EPUB to an M4B audiobook with per-chapter voice assignment.

    Pipeline:
        1. Parse the EPUB into metadata and chapters.
        2. Initialise a VoiceManager from the voice config.
        3. For each chapter: resolve voice, optionally preprocess text,
           synthesise audio via Kokoro, and write a per-chapter WAV.
        4. Assemble all chapter WAVs into a single M4B with chapter
           metadata using ffmpeg.
        5. Clean up intermediate WAV files.

    Args:
        epub_path: Path to the source ``.epub`` file.
        config_path: Path to ``voice_config.json``.
        output_dir: Directory for output files (WAVs and final M4B).
        use_preprocessing: Whether to run the text preprocessor.
        substitutions_path: Optional path to a custom substitutions JSON.
        device: Torch device (``'cpu'``, ``'cuda'``, or ``'auto'``).

    Returns:
        Path to the generated ``.m4b`` file.

    Raises:
        FileNotFoundError: If the EPUB or config file does not exist.
        RuntimeError: If ffmpeg is missing or encoding fails.
    """
    from .epub_parser import parse_epub
    from .preprocess import preprocess
    from .audio_utils import create_m4b, cleanup_chapter_wavs

    epub_path = Path(epub_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse EPUB
    logger.info("Parsing EPUB: %s", epub_path)
    metadata, chapters = parse_epub(epub_path)
    logger.info(
        "Found %d chapters in '%s' by %s",
        len(chapters),
        metadata.title,
        metadata.author,
    )

    if not chapters:
        raise ValueError(f"No chapters found in {epub_path}")

    # Step 2: Initialise voice manager
    voice_manager = VoiceManager(config_path, device=device)

    # Step 3: Synthesise per-chapter WAVs
    chapter_wav_files: list[Path] = []
    chapter_titles: list[str] = []

    for chapter in chapters:
        pipeline, voice_name, speed = voice_manager.get_pipeline_and_voice(
            chapter.index
        )

        logger.info(
            "Chapter %d (%s): voice=%s, speed=%.2f",
            chapter.index,
            chapter.title,
            voice_name,
            speed,
        )

        # Preprocess text if requested
        text = chapter.text
        if use_preprocessing:
            text = preprocess(text, substitutions_path=substitutions_path)

        # Synthesise audio
        wav_path = output_dir / f"chapter_{chapter.index:04d}.wav"
        _synthesize_chapter(pipeline, text, voice_name, speed, wav_path)

        chapter_wav_files.append(wav_path)
        chapter_titles.append(chapter.title)

    # Step 4: Assemble M4B
    safe_title = _sanitize_filename(metadata.title)
    m4b_path = output_dir / f"{safe_title}.m4b"

    # Write cover image to a temp file if available
    cover_path: Optional[Path] = None
    if metadata.cover_image:
        cover_path = output_dir / ".cover_tmp.jpg"
        cover_path.write_bytes(metadata.cover_image)

    logger.info("Assembling M4B: %s", m4b_path)
    create_m4b(
        chapter_wav_files=chapter_wav_files,
        chapter_titles=chapter_titles,
        output_path=m4b_path,
        cover_image=cover_path,
    )

    # Step 5: Cleanup
    cleanup_chapter_wavs(chapter_wav_files)
    if cover_path and cover_path.exists():
        cover_path.unlink()

    logger.info("Done. Output: %s", m4b_path)
    return m4b_path


def _synthesize_chapter(
    pipeline: object,
    text: str,
    voice: str,
    speed: float,
    wav_path: Path,
) -> None:
    """Synthesise a chapter's text to a WAV file using a Kokoro pipeline.

    Generates audio in streaming chunks and writes the concatenated
    result to *wav_path* as a 24kHz mono WAV.

    Args:
        pipeline: A ``KPipeline`` instance.
        text: The preprocessed chapter text.
        voice: Kokoro voice identifier (e.g. ``"af_sky"``).
        speed: Playback speed multiplier.
        wav_path: Destination WAV file.
    """
    import numpy as np  # type: ignore[import-untyped]
    import soundfile as sf  # type: ignore[import-untyped]

    sample_rate = 24000  # Kokoro output sample rate
    audio_chunks: list[np.ndarray] = []

    for _graphemes, _phonemes, audio_chunk in pipeline(
        text, voice=voice, speed=speed
    ):
        if audio_chunk is not None:
            audio_chunks.append(audio_chunk)

    if not audio_chunks:
        logger.warning("No audio generated for %s", wav_path)
        # Write a short silent WAV so the pipeline doesn't break
        silence = np.zeros(sample_rate, dtype=np.float32)
        sf.write(str(wav_path), silence, sample_rate)
        return

    combined = np.concatenate(audio_chunks)
    sf.write(str(wav_path), combined, sample_rate)
    logger.debug("Wrote %s (%.1fs)", wav_path, len(combined) / sample_rate)


def _sanitize_filename(name: str) -> str:
    """Strip or replace characters unsafe for filenames."""
    import re

    # Replace unsafe characters with underscores
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe)
    # Strip leading/trailing whitespace and underscores
    safe = safe.strip(" _")
    return safe if safe else "audiobook"
