"""Audio utilities for audiobook assembly via ffmpeg.

Wraps ffmpeg and ffprobe for WAV concatenation, chapter metadata generation,
and M4B audiobook encoding. All operations use subprocess calls with proper
error handling and timeout protection.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # Kokoro TTS output sample rate in Hz


def check_ffmpeg() -> bool:
    """Verify ffmpeg and ffprobe are available on the system PATH.

    Returns:
        True if both ffmpeg and ffprobe are found, False otherwise.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if not ffmpeg_path:
        logger.error("ffmpeg not found on PATH. Install ffmpeg to proceed.")
        return False

    if not ffprobe_path:
        logger.error("ffprobe not found on PATH. Install ffmpeg to proceed.")
        return False

    logger.debug("ffmpeg found at %s", ffmpeg_path)
    logger.debug("ffprobe found at %s", ffprobe_path)
    return True


def probe_duration(wav_path: Path) -> float:
    """Get the duration of a WAV file in seconds via ffprobe.

    Args:
        wav_path: Path to the WAV file.

    Returns:
        Duration in seconds as a float.

    Raises:
        FileNotFoundError: If the WAV file does not exist.
        RuntimeError: If ffprobe fails or returns invalid output.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(wav_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffprobe failed for {wav_path}: {stderr}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffprobe timed out after 600s for {wav_path}"
        ) from exc

    output = result.stdout.decode("utf-8").strip()
    if not output:
        raise RuntimeError(f"ffprobe returned empty duration for {wav_path}")

    try:
        duration = float(output)
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned non-numeric duration '{output}' for {wav_path}"
        ) from exc

    return duration


def concat_wavs(wav_files: list[Path], output_path: Path) -> Path:
    """Concatenate WAV files into a single WAV using ffmpeg concat demuxer.

    Args:
        wav_files: Ordered list of WAV file paths to concatenate.
        output_path: Destination path for the combined WAV.

    Returns:
        The output path.

    Raises:
        ValueError: If wav_files is empty.
        FileNotFoundError: If any input WAV file does not exist.
        RuntimeError: If ffmpeg fails.
    """
    if not wav_files:
        raise ValueError("wav_files list is empty; nothing to concatenate.")

    for wf in wav_files:
        if not Path(wf).exists():
            raise FileNotFoundError(f"WAV file not found: {wf}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the concat demuxer filelist to a temp file in the same directory
    # as the output so relative paths are unnecessary — we use absolute paths.
    filelist_path = output_path.parent / f".filelist_{output_path.stem}.txt"
    try:
        with open(filelist_path, "w", encoding="utf-8") as fh:
            for wf in wav_files:
                # Escape single quotes in paths for ffmpeg concat format
                safe_path = str(Path(wf).resolve()).replace("'", "'\\''")
                fh.write(f"file '{safe_path}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(filelist_path),
            "-c", "copy",
            str(output_path),
        ]

        logger.info("Concatenating %d WAV files into %s", len(wav_files), output_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                timeout=600,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg concat failed: {stderr}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "ffmpeg concat timed out after 600s"
            ) from exc

        logger.debug("ffmpeg stderr: %s", result.stderr.decode("utf-8", errors="replace"))

    finally:
        # Clean up the filelist
        if filelist_path.exists():
            filelist_path.unlink()

    return output_path


def create_chapter_metadata(
    chapters: list[tuple[str, float]],
    output_path: Path,
) -> Path:
    """Write an FFMETADATA1 file with chapter timestamps.

    Args:
        chapters: List of (title, duration_seconds) tuples in order.
        output_path: Where to write the metadata file.

    Returns:
        The output path.

    Raises:
        ValueError: If chapters list is empty.
    """
    if not chapters:
        raise ValueError("chapters list is empty; cannot create metadata.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [";FFMETADATA1"]

    cumulative_ms = 0
    for title, duration_seconds in chapters:
        start_ms = cumulative_ms
        end_ms = cumulative_ms + int(round(duration_seconds * 1000))

        lines.append("")
        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={start_ms}")
        lines.append(f"END={end_ms}")
        # Escape special characters in title per FFMETADATA1 spec
        safe_title = (
            title.replace("\\", "\\\\")
            .replace("=", "\\=")
            .replace(";", "\\;")
            .replace("#", "\\#")
            .replace("\n", "\\\n")
        )
        lines.append(f"title={safe_title}")

        cumulative_ms = end_ms

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info(
        "Wrote chapter metadata with %d chapters to %s",
        len(chapters),
        output_path,
    )
    return output_path


def create_m4b(
    chapter_wav_files: list[Path],
    chapter_titles: list[str],
    output_path: Path,
    cover_image: Optional[Path] = None,
    bitrate: str = "64k",
) -> Path:
    """Assemble chapter WAV files into a single M4B audiobook.

    Pipeline:
        1. Concatenate all chapter WAVs into a single WAV.
        2. Probe each chapter WAV for its duration.
        3. Create FFMETADATA1 file with chapter markers.
        4. Encode to AAC in M4B container with chapter metadata.
        5. Optionally embed cover art.
        6. Clean up temporary files.

    Args:
        chapter_wav_files: Ordered list of per-chapter WAV paths.
        chapter_titles: Titles corresponding to each chapter WAV.
        output_path: Destination for the final .m4b file.
        cover_image: Optional path to cover art (JPEG/PNG).
        bitrate: AAC encoding bitrate (default "64k").

    Returns:
        The output path.

    Raises:
        ValueError: If inputs are inconsistent or empty.
        FileNotFoundError: If any input file is missing.
        RuntimeError: If ffmpeg/ffprobe operations fail.
    """
    if not chapter_wav_files:
        raise ValueError("chapter_wav_files is empty; nothing to encode.")

    if len(chapter_wav_files) != len(chapter_titles):
        raise ValueError(
            f"Mismatch: {len(chapter_wav_files)} WAV files "
            f"but {len(chapter_titles)} chapter titles."
        )

    if not check_ffmpeg():
        raise RuntimeError(
            "ffmpeg and/or ffprobe not found. "
            "Install ffmpeg before creating M4B files."
        )

    for wf in chapter_wav_files:
        if not Path(wf).exists():
            raise FileNotFoundError(f"Chapter WAV not found: {wf}")

    if cover_image is not None:
        cover_image = Path(cover_image)
        if not cover_image.exists():
            raise FileNotFoundError(f"Cover image not found: {cover_image}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a temp directory for intermediate files
    tmp_dir = Path(tempfile.mkdtemp(prefix="m4b_assembly_"))
    combined_wav = tmp_dir / "combined.wav"
    metadata_file = tmp_dir / "metadata.txt"

    try:
        # Step 1: Concatenate all chapter WAVs
        logger.info("Step 1/4: Concatenating %d chapter WAVs...", len(chapter_wav_files))
        concat_wavs(chapter_wav_files, combined_wav)

        # Step 2: Probe each chapter for duration
        logger.info("Step 2/4: Probing chapter durations...")
        chapters: list[tuple[str, float]] = []
        for wav_file, title in zip(chapter_wav_files, chapter_titles):
            duration = probe_duration(Path(wav_file))
            chapters.append((title, duration))
            logger.debug("  %s: %.2fs", title, duration)

        # Step 3: Create FFMETADATA1
        logger.info("Step 3/4: Writing chapter metadata...")
        create_chapter_metadata(chapters, metadata_file)

        # Step 4: Encode to M4B
        logger.info("Step 4/4: Encoding M4B at %s bitrate...", bitrate)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(combined_wav),
            "-i", str(metadata_file),
        ]

        if cover_image is not None:
            cmd.extend(["-i", str(cover_image)])

        cmd.extend(["-map_metadata", "1"])

        if cover_image is not None:
            # Map audio from input 0, video (cover) from input 2
            cmd.extend([
                "-map", "0:a",
                "-map", "2:v",
                "-disposition:v", "attached_pic",
            ])

        cmd.extend([
            "-c:a", "aac",
            "-b:a", bitrate,
            "-movflags", "+faststart",
            str(output_path),
        ])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                timeout=600,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg M4B encoding failed: {stderr}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "ffmpeg M4B encoding timed out after 600s"
            ) from exc

        logger.debug("ffmpeg stderr: %s", result.stderr.decode("utf-8", errors="replace"))
        logger.info("M4B created: %s", output_path)

    finally:
        # Step 6: Clean up temporary files
        for tmp_file in [combined_wav, metadata_file]:
            if tmp_file.exists():
                tmp_file.unlink()
                logger.debug("Cleaned up %s", tmp_file)
        # Remove the temp directory if empty
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    return output_path


def cleanup_chapter_wavs(wav_files: list[Path]) -> None:
    """Delete intermediate chapter WAV files to reclaim disk space.

    Silently skips files that do not exist. Logs each deletion.

    Args:
        wav_files: List of WAV file paths to delete.
    """
    deleted = 0
    for wf in wav_files:
        wf = Path(wf)
        if wf.exists():
            wf.unlink()
            logger.debug("Deleted %s", wf)
            deleted += 1
        else:
            logger.debug("Skipped (not found): %s", wf)

    logger.info("Cleaned up %d of %d chapter WAV files.", deleted, len(wav_files))
