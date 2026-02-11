"""Batch conversion module for the audiblez-kokoro-pipeline.

Processes all .epub files in a directory into .m4b audiobooks, skipping
already-converted files, with rich progress bars and a summary table.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from . import epub_parser, preprocess, kokoro_direct, audio_utils

logger = logging.getLogger(__name__)

console = Console()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_epub_files(input_dir: Path) -> list[Path]:
    """Find all .epub files in a directory (non-recursive).

    Args:
        input_dir: Directory to scan for .epub files.

    Returns:
        Sorted list of Path objects for each .epub file found.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        logger.warning("Input directory does not exist: %s", input_dir)
        return []

    epub_files = sorted(input_dir.glob("*.epub"), key=lambda p: p.name)
    logger.info("Found %d .epub file(s) in %s", len(epub_files), input_dir)
    return epub_files


def check_existing_output(epub_path: Path, output_dir: Path) -> bool:
    """Check if a corresponding .m4b already exists for an epub.

    Args:
        epub_path: Path to the source .epub file.
        output_dir: Directory where .m4b outputs are stored.

    Returns:
        True if ``output_dir / "{epub_stem}.m4b"`` exists, False otherwise.
    """
    m4b_path = Path(output_dir) / f"{epub_path.stem}.m4b"
    return m4b_path.exists()


# ---------------------------------------------------------------------------
# Single-book conversion (internal)
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS or M:SS."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _convert_single(
    epub_path: Path,
    output_dir: Path,
    voice: str,
    speed: float,
    voice_config: Optional[Path],
    use_preprocessing: bool,
    substitutions: Optional[Path],
    use_cuda: bool,
    overall_progress: Progress,
    overall_task_id: int,
    book_progress: Progress,
) -> dict:
    """Convert a single .epub to .m4b. Returns a result dict.

    This is the inner workhorse called by :func:`batch_convert` for each
    book. It handles parsing, optional preprocessing, TTS synthesis (single
    or multi-voice), audiobook assembly, and cleanup of intermediate files.
    """
    m4b_path = output_dir / f"{epub_path.stem}.m4b"
    start_time = time.monotonic()

    try:
        # --- Parse ---
        metadata, chapters = epub_parser.parse_epub(epub_path)
        logger.info(
            "Parsed '%s' by %s: %d chapters, lang=%s",
            metadata.title,
            metadata.author,
            len(chapters),
            metadata.language,
        )

        if not chapters:
            logger.warning("No chapters extracted from %s; skipping.", epub_path)
            return {
                "epub": str(epub_path),
                "status": "error",
                "duration_sec": time.monotonic() - start_time,
                "output": None,
                "error": "No chapters extracted",
            }

        # --- Preprocess ---
        if use_preprocessing:
            for ch in chapters:
                ch.text = preprocess.preprocess(
                    ch.text,
                    substitutions_path=substitutions,
                )

        # --- Synthesize ---
        device = "cuda" if use_cuda else "cpu"
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"batch_{epub_path.stem}_"))

        book_task = book_progress.add_task(
            f"[cyan]{metadata.title[:40]}",
            total=len(chapters),
        )

        wav_paths: list[Path] = []
        chapter_titles: list[str] = []

        if voice_config is not None:
            # Multi-voice path
            from .multi_voice import VoiceManager, convert_with_multi_voice

            voice_mgr = VoiceManager(voice_config)

            for i, ch in enumerate(chapters):
                chapter_wav = tmp_dir / f"chapter_{i:02d}.wav"
                convert_with_multi_voice(
                    text=ch.text,
                    output_path=chapter_wav,
                    voice_manager=voice_mgr,
                    speed=speed,
                    device=device,
                )
                wav_paths.append(chapter_wav)
                chapter_titles.append(ch.title)
                book_progress.update(book_task, advance=1)
        else:
            # Single-voice path: init pipeline once, reuse
            pipeline = kokoro_direct.init_pipeline(
                lang_code=metadata.language,
                device=device,
            )

            for i, ch in enumerate(chapters):
                chapter_wav = tmp_dir / f"chapter_{i:02d}.wav"
                kokoro_direct.synthesize_chapter(
                    pipeline=pipeline,
                    text=ch.text,
                    voice=voice,
                    speed=speed,
                    output_path=chapter_wav,
                )
                wav_paths.append(chapter_wav)
                chapter_titles.append(ch.title)
                book_progress.update(book_task, advance=1)

        book_progress.update(book_task, description=f"[green]{metadata.title[:40]}")

        # --- Write cover image to temp file if present ---
        cover_path: Optional[Path] = None
        if metadata.cover_image:
            cover_path = tmp_dir / "cover.jpg"
            cover_path.write_bytes(metadata.cover_image)

        # --- Assemble M4B ---
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_utils.create_m4b(
            chapter_wav_files=wav_paths,
            chapter_titles=chapter_titles,
            output_path=m4b_path,
            cover_image=cover_path,
        )

        # --- Cleanup temp files ---
        audio_utils.cleanup_chapter_wavs(wav_paths)
        if cover_path and cover_path.exists():
            cover_path.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

        elapsed = time.monotonic() - start_time
        overall_progress.advance(overall_task_id)

        return {
            "epub": str(epub_path),
            "status": "success",
            "duration_sec": elapsed,
            "output": str(m4b_path),
            "error": None,
        }

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.exception("Failed to convert %s: %s", epub_path, exc)
        overall_progress.advance(overall_task_id)
        return {
            "epub": str(epub_path),
            "status": "error",
            "duration_sec": elapsed,
            "output": None,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def batch_convert(
    input_dir: Path = Path("input"),
    output_dir: Path = Path("output"),
    voice: str = "af_sky",
    speed: float = 1.0,
    voice_config: Optional[Path] = None,
    use_preprocessing: bool = True,
    substitutions: Optional[Path] = None,
    use_cuda: bool = True,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> list[dict]:
    """Batch-convert all .epub files in a directory to .m4b audiobooks.

    Discovers epub files, optionally skips those whose output already exists,
    and converts each sequentially (GPU-bound). Displays rich progress bars
    and returns a results manifest.

    Args:
        input_dir: Directory containing .epub files.
        output_dir: Directory for .m4b output files.
        voice: Kokoro voice identifier for single-voice mode.
        speed: Playback speed multiplier (1.0 = normal).
        voice_config: Path to a multi-voice YAML/JSON config. When provided,
            uses multi-voice synthesis instead of the single *voice* parameter.
        use_preprocessing: Whether to run the text preprocessing pipeline.
        substitutions: Optional path to a JSON pronunciation substitutions file.
        use_cuda: Whether to attempt CUDA acceleration.
        skip_existing: If True, skip epubs whose .m4b already exists.
        dry_run: If True, parse and display info without synthesizing.

    Returns:
        List of result dicts, one per epub. Each dict contains:
        ``epub``, ``status`` ('success'|'error'|'skipped'), ``duration_sec``,
        ``output``, ``error``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    epub_files = find_epub_files(input_dir)

    if not epub_files:
        console.print("[yellow]No .epub files found in[/yellow]", str(input_dir))
        return []

    console.print(
        f"\n[bold]Found {len(epub_files)} epub file(s)[/bold] in {input_dir}\n"
    )

    results: list[dict] = []

    # --- Classify: skip existing vs. to-process ---
    to_process: list[Path] = []
    for epub_path in epub_files:
        if skip_existing and check_existing_output(epub_path, output_dir):
            logger.info("Skipping %s (output already exists)", epub_path.name)
            results.append({
                "epub": str(epub_path),
                "status": "skipped",
                "duration_sec": 0.0,
                "output": str(output_dir / f"{epub_path.stem}.m4b"),
                "error": None,
            })
        else:
            to_process.append(epub_path)

    skipped_count = len(results)
    if skipped_count > 0:
        console.print(
            f"[dim]Skipping {skipped_count} already-converted file(s)[/dim]"
        )

    if not to_process:
        console.print("[green]All files already converted. Nothing to do.[/green]")
        return results

    # --- Dry run: parse and display info only ---
    if dry_run:
        console.print("[bold yellow]DRY RUN[/bold yellow] -- no synthesis\n")

        info_table = Table(title="Dry Run Summary")
        info_table.add_column("File", style="cyan")
        info_table.add_column("Title", style="white")
        info_table.add_column("Author", style="dim")
        info_table.add_column("Chapters", justify="right")
        info_table.add_column("Words", justify="right")
        info_table.add_column("Est. Duration", justify="right", style="green")

        for epub_path in to_process:
            try:
                metadata, chapters = epub_parser.parse_epub(epub_path)
                total_words = sum(ch.word_count for ch in chapters)
                total_duration = sum(ch.estimated_duration_sec for ch in chapters)

                info_table.add_row(
                    epub_path.name,
                    metadata.title[:50],
                    metadata.author[:30],
                    str(len(chapters)),
                    f"{total_words:,}",
                    _format_duration(total_duration),
                )

                results.append({
                    "epub": str(epub_path),
                    "status": "dry_run",
                    "duration_sec": 0.0,
                    "output": None,
                    "error": None,
                })

            except Exception as exc:
                info_table.add_row(
                    epub_path.name,
                    "[red]PARSE ERROR[/red]",
                    str(exc)[:40],
                    "-",
                    "-",
                    "-",
                )
                results.append({
                    "epub": str(epub_path),
                    "status": "error",
                    "duration_sec": 0.0,
                    "output": None,
                    "error": str(exc),
                })

        console.print(info_table)
        return results

    # --- Full conversion ---
    if not audio_utils.check_ffmpeg():
        console.print(
            "[bold red]ffmpeg not found.[/bold red] "
            "Install ffmpeg before batch conversion."
        )
        for epub_path in to_process:
            results.append({
                "epub": str(epub_path),
                "status": "error",
                "duration_sec": 0.0,
                "output": None,
                "error": "ffmpeg not found",
            })
        return results

    batch_start = time.monotonic()

    # Overall progress: one tick per book
    overall_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Overall"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    # Per-book chapter progress
    book_progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with overall_progress:
        overall_task = overall_progress.add_task(
            "Converting", total=len(to_process)
        )

        with book_progress:
            for epub_path in to_process:
                console.print(
                    f"\n[bold]Processing:[/bold] {epub_path.name}"
                )

                result = _convert_single(
                    epub_path=epub_path,
                    output_dir=output_dir,
                    voice=voice,
                    speed=speed,
                    voice_config=voice_config,
                    use_preprocessing=use_preprocessing,
                    substitutions=substitutions,
                    use_cuda=use_cuda,
                    overall_progress=overall_progress,
                    overall_task_id=overall_task,
                    book_progress=book_progress,
                )
                results.append(result)

                status_style = (
                    "green" if result["status"] == "success" else "red"
                )
                console.print(
                    f"  [{status_style}]{result['status'].upper()}[/{status_style}]"
                    f"  ({_format_duration(result['duration_sec'])})"
                )

    # --- Summary ---
    batch_elapsed = time.monotonic() - batch_start
    _print_summary(results, batch_elapsed)

    return results


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict], batch_elapsed: float) -> None:
    """Print a rich summary table of all batch results."""
    table = Table(title="\nBatch Conversion Summary")
    table.add_column("File", style="cyan", max_width=40)
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Output", style="dim", max_width=50)

    success_count = 0
    error_count = 0
    skip_count = 0

    for result in results:
        epub_name = Path(result["epub"]).name

        status = result["status"]
        if status == "success":
            status_display = "[green]SUCCESS[/green]"
            success_count += 1
        elif status == "skipped":
            status_display = "[yellow]SKIPPED[/yellow]"
            skip_count += 1
        elif status == "dry_run":
            status_display = "[blue]DRY RUN[/blue]"
        else:
            status_display = "[red]ERROR[/red]"
            error_count += 1

        duration = _format_duration(result["duration_sec"])

        output_display = result.get("output") or ""
        if output_display:
            output_display = Path(output_display).name

        # For errors, show the error message in the output column
        if result["status"] == "error" and result.get("error"):
            error_msg = result["error"]
            # Truncate long error messages
            if len(error_msg) > 48:
                error_msg = error_msg[:45] + "..."
            output_display = f"[red]{error_msg}[/red]"

        table.add_row(epub_name, status_display, duration, output_display)

    console.print(table)

    # Totals line
    parts = []
    if success_count:
        parts.append(f"[green]{success_count} succeeded[/green]")
    if skip_count:
        parts.append(f"[yellow]{skip_count} skipped[/yellow]")
    if error_count:
        parts.append(f"[red]{error_count} failed[/red]")

    totals = ", ".join(parts)
    console.print(
        f"\n[bold]{len(results)} total[/bold]: {totals}"
        f"  [dim]({_format_duration(batch_elapsed)} elapsed)[/dim]\n"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for batch epub-to-audiobook conversion."""
    parser = argparse.ArgumentParser(
        description="Batch convert EPUB files to M4B audiobooks using Kokoro TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s -i ./books -o ./audiobooks\n"
            "  %(prog)s -i ./books --voice am_adam --speed 1.1\n"
            "  %(prog)s -i ./books --voice-config voices.yaml\n"
            "  %(prog)s -i ./books --dry-run\n"
        ),
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=Path("input"),
        help="Directory containing .epub files (default: input/)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("output"),
        help="Directory for .m4b output files (default: output/)",
    )
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default="af_sky",
        help="Kokoro voice identifier (default: af_sky)",
    )
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--voice-config",
        type=Path,
        default=None,
        help="Path to multi-voice config file (YAML/JSON). "
             "Overrides --voice with per-character voice assignments.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable text preprocessing (abbreviations, numbers, etc.)",
    )
    parser.add_argument(
        "--substitutions",
        type=Path,
        default=None,
        help="Path to JSON pronunciation substitutions file",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only synthesis (disable CUDA)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-convert files even if .m4b output already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and display info without synthesizing audio",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    results = batch_convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        voice=args.voice,
        speed=args.speed,
        voice_config=args.voice_config,
        use_preprocessing=not args.no_preprocess,
        substitutions=args.substitutions,
        use_cuda=not args.cpu,
        skip_existing=not args.no_skip,
        dry_run=args.dry_run,
    )

    # Exit with non-zero status if any conversions failed
    if any(r["status"] == "error" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
