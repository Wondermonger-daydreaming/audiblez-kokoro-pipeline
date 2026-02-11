"""Direct Kokoro TTS engine wrapper for text-to-speech synthesis.

Wraps the Kokoro KPipeline for direct TTS synthesis, bypassing audiblez.
This is the central audio generation engine used by all other modules in
the pipeline. Handles single-text synthesis, chapter-level synthesis with
progress tracking, and full-book synthesis across multiple chapters.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

try:
    import soundfile as sf
except ImportError as _sf_err:
    raise ImportError(
        "soundfile is required for audio I/O. "
        "Install it with: pip install soundfile"
    ) from _sf_err

try:
    from kokoro import KPipeline
except ImportError as _kokoro_err:
    raise ImportError(
        "kokoro is required for TTS synthesis. "
        "Install it with: pip install kokoro"
    ) from _kokoro_err

if TYPE_CHECKING:
    from epub_parser import Chapter

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # Kokoro TTS output sample rate in Hz


def init_pipeline(lang_code: str = "a", device: str = "auto") -> KPipeline:
    """Initialize a Kokoro TTS pipeline with automatic device selection.

    Args:
        lang_code: Kokoro language code. 'a' for American English, 'b' for
            British English, 'e' for Spanish, 'f' for French, 'h' for Hindi,
            'i' for Italian, 'j' for Japanese, 'p' for Portuguese, 'z' for
            Chinese.
        device: Compute device. 'auto' selects CUDA when available, falling
            back to CPU. Also accepts 'cuda' or 'cpu' explicitly.

    Returns:
        An initialized KPipeline ready for synthesis.
    """
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    logger.info("Initializing Kokoro pipeline: lang_code=%r, device=%r", lang_code, device)

    pipeline = KPipeline(lang_code=lang_code)

    logger.info("Kokoro pipeline initialized on %s", device)
    return pipeline


def synthesize_text(
    pipeline: KPipeline,
    text: str,
    voice: str = "af_sky",
    speed: float = 1.0,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """Synthesize text to a numpy audio array.

    Iterates over the KPipeline generator and concatenates all audio chunks
    into a single contiguous array. Optionally writes the result to a WAV
    file at the specified path.

    Args:
        pipeline: An initialized KPipeline instance.
        text: The text to synthesize. Empty or whitespace-only text returns
            an empty float32 array.
        voice: Kokoro voice identifier (e.g. 'af_sky', 'am_adam').
        speed: Playback speed multiplier. 1.0 is normal speed.
        output_path: If provided, saves the audio as a WAV file at this path.

    Returns:
        A numpy float32 array of audio samples at 24000 Hz.
    """
    if not text or not text.strip():
        logger.debug("Empty text provided; returning empty audio array.")
        return np.array([], dtype=np.float32)

    logger.debug(
        "Synthesizing %d characters with voice=%r, speed=%.2f",
        len(text),
        voice,
        speed,
    )

    chunks: list[np.ndarray] = []

    for graphemes, phonemes, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None and len(audio) > 0:
            chunks.append(audio)

    if not chunks:
        logger.warning("Pipeline produced no audio chunks for %d characters of text.", len(text))
        return np.array([], dtype=np.float32)

    audio = np.concatenate(chunks)

    logger.debug("Synthesized %.2f seconds of audio (%d samples).", len(audio) / SAMPLE_RATE, len(audio))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, SAMPLE_RATE)
        logger.info("Saved audio to %s", output_path)

    return audio


def synthesize_chapter(
    pipeline: KPipeline,
    text: str,
    voice: str,
    speed: float,
    output_path: Path,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Path:
    """Synthesize a full chapter with progress tracking.

    Processes text through the KPipeline generator chunk by chunk, tracking
    character progress and providing estimated time remaining through an
    optional callback. Saves the result as a WAV file.

    Args:
        pipeline: An initialized KPipeline instance.
        text: The chapter text to synthesize.
        voice: Kokoro voice identifier.
        speed: Playback speed multiplier.
        output_path: Destination path for the WAV file.
        progress_callback: Optional function called after each chunk with
            (chars_processed, total_chars, estimated_remaining_sec).

    Returns:
        The output path where the WAV was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not text or not text.strip():
        logger.warning("Empty chapter text; writing silent WAV to %s", output_path)
        empty_audio = np.array([], dtype=np.float32)
        sf.write(str(output_path), empty_audio, SAMPLE_RATE)
        return output_path

    total_chars = len(text)
    chars_processed = 0
    chunks: list[np.ndarray] = []
    start_time = time.monotonic()

    for graphemes, phonemes, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None and len(audio) > 0:
            chunks.append(audio)

        # Track progress via the graphemes returned by the generator
        chars_processed += len(graphemes) if graphemes else 0

        if progress_callback is not None:
            elapsed = time.monotonic() - start_time

            if chars_processed > 0 and elapsed > 0:
                rate = chars_processed / elapsed
                remaining_chars = total_chars - chars_processed
                estimated_remaining = remaining_chars / rate
            else:
                estimated_remaining = 0.0

            progress_callback(chars_processed, total_chars, estimated_remaining)

    if chunks:
        audio = np.concatenate(chunks)
    else:
        logger.warning("No audio chunks produced for chapter; writing empty WAV.")
        audio = np.array([], dtype=np.float32)

    sf.write(str(output_path), audio, SAMPLE_RATE)

    duration = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0.0
    logger.info(
        "Chapter synthesized: %.2fs of audio, %d chars, saved to %s",
        duration,
        total_chars,
        output_path,
    )

    return output_path


def synthesize_book(
    chapters: list,
    voice: str = "af_sky",
    speed: float = 1.0,
    output_dir: Path = Path("output"),
    device: str = "auto",
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> list[Path]:
    """Synthesize all chapters of a book into individual WAV files.

    Initializes a single pipeline instance and reuses it across all chapters.
    Each chapter is written to its own WAV file named chapter_00.wav,
    chapter_01.wav, etc.

    Args:
        chapters: List of chapter objects with .text and .title attributes
            (typically epub_parser.Chapter dataclass instances).
        voice: Kokoro voice identifier.
        speed: Playback speed multiplier.
        output_dir: Directory for chapter WAV output files.
        device: Compute device for pipeline initialization ('auto', 'cuda',
            or 'cpu').
        progress_callback: Optional function called during synthesis with
            (chars_processed, total_chars, estimated_remaining_sec). Receives
            cumulative progress across all chapters.

    Returns:
        Ordered list of Paths to the generated chapter WAV files.
    """
    if not chapters:
        logger.warning("No chapters provided; returning empty list.")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = init_pipeline(lang_code="a", device=device)

    total_chars = sum(len(ch.text) for ch in chapters)
    cumulative_chars = 0
    start_time = time.monotonic()
    wav_paths: list[Path] = []

    for i, chapter in enumerate(chapters):
        chapter_path = output_dir / f"chapter_{i:02d}.wav"

        logger.info(
            "Synthesizing chapter %d/%d: %r (%d chars)",
            i + 1,
            len(chapters),
            chapter.title,
            len(chapter.text),
        )

        if not chapter.text or not chapter.text.strip():
            logger.warning("Chapter %d (%r) has no text; writing empty WAV.", i, chapter.title)
            sf.write(str(chapter_path), np.array([], dtype=np.float32), SAMPLE_RATE)
            wav_paths.append(chapter_path)
            continue

        def _chapter_progress(chars_done: int, chars_total: int, _est: float) -> None:
            """Relay per-chapter progress as cumulative book progress."""
            if progress_callback is not None:
                book_chars_done = cumulative_chars + chars_done
                elapsed = time.monotonic() - start_time

                if book_chars_done > 0 and elapsed > 0:
                    rate = book_chars_done / elapsed
                    remaining = (total_chars - book_chars_done) / rate
                else:
                    remaining = 0.0

                progress_callback(book_chars_done, total_chars, remaining)

        synthesize_chapter(
            pipeline=pipeline,
            text=chapter.text,
            voice=voice,
            speed=speed,
            output_path=chapter_path,
            progress_callback=_chapter_progress,
        )

        cumulative_chars += len(chapter.text)
        wav_paths.append(chapter_path)

    logger.info(
        "Book synthesis complete: %d chapters, %d total characters.",
        len(wav_paths),
        total_chars,
    )

    return wav_paths
