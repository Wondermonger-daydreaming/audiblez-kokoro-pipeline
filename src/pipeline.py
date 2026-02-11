"""
Unified CLI entry point for the audiblez-kokoro EPUB-to-audiobook pipeline.

Ties together all modules: EPUB parsing, text preprocessing, LLM enhancement,
single-voice and multi-voice Kokoro TTS synthesis, format conversion, batch
processing, and M4B audiobook assembly.

Usage examples::

    python -m src.pipeline book.epub
    python -m src.pipeline book.epub -v af_heart --cuda
    python -m src.pipeline --batch input/ -o output/
    python -m src.pipeline book.epub --voice-config config/voice_config.json
    python -m src.pipeline book.epub --preprocess --substitutions config/substitutions.json
    python -m src.pipeline book.epub --llm --llm-model llama3.2
    python -m src.pipeline book.epub --direct
    python -m src.pipeline document.pdf -o output/
    python -m src.pipeline --batch input/ --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
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

from .audio_utils import check_ffmpeg, cleanup_chapter_wavs, create_m4b
from .batch_convert import batch_convert
from .epub_parser import BookMetadata, Chapter, parse_epub
from .format_wrapper import ensure_epub
from .kokoro_direct import init_pipeline, synthesize_chapter
from .llm_pipeline import LLMPipeline
from .multi_voice import VoiceManager
from .preprocess import preprocess

logger = logging.getLogger(__name__)

console = Console()

DEFAULT_VOICE = "af_sky"
DEFAULT_SPEED = 1.0
DEFAULT_BITRATE = "64k"
DEFAULT_SUBSTITUTIONS = Path("config/substitutions.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS or M:SS."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _sanitize_filename(name: str) -> str:
    """Strip or replace characters unsafe for filenames."""
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip(" _")
    return safe if safe else "audiobook"


def _resolve_device(cuda: bool, cpu: bool) -> str:
    """Resolve the compute device from CLI flags.

    Args:
        cuda: True if --cuda was passed.
        cpu: True if --cpu was passed.

    Returns:
        One of 'cuda', 'cpu', or 'auto'.
    """
    if cpu:
        return "cpu"
    if cuda:
        return "cuda"
    return "auto"


def _estimate_total_duration(chapters: list[Chapter]) -> float:
    """Sum estimated durations across all chapters in seconds."""
    return sum(ch.estimated_duration_sec for ch in chapters)


def _total_words(chapters: list[Chapter]) -> int:
    """Sum word counts across all chapters."""
    return sum(ch.word_count for ch in chapters)


# ---------------------------------------------------------------------------
# Book info display
# ---------------------------------------------------------------------------


def _print_book_info(metadata: BookMetadata, chapters: list[Chapter]) -> None:
    """Print a rich panel summarising the parsed book."""
    total_words = _total_words(chapters)
    est_duration = _estimate_total_duration(chapters)

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="bold cyan")
    info_table.add_column("Value")

    info_table.add_row("Title", metadata.title)
    info_table.add_row("Author", metadata.author)
    info_table.add_row("Language", metadata.language)
    info_table.add_row("Chapters", str(metadata.chapter_count))
    info_table.add_row("Words", f"{total_words:,}")
    info_table.add_row("Est. Duration", _format_duration(est_duration))
    info_table.add_row("Cover Image", "Yes" if metadata.cover_image else "No")

    console.print(Panel(info_table, title="Book Info", border_style="blue"))


# ---------------------------------------------------------------------------
# Single-file conversion
# ---------------------------------------------------------------------------


def convert_single(
    input_path: Path,
    output_dir: Path,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    device: str = "auto",
    voice_config: Optional[Path] = None,
    use_preprocessing: bool = True,
    substitutions_path: Optional[Path] = None,
    use_llm: bool = False,
    llm_model: str = "llama3.2",
    direct_mode: bool = False,
    skip_existing: bool = True,
    dry_run: bool = False,
    bitrate: str = DEFAULT_BITRATE,
) -> Path:
    """Convert a single EPUB/PDF/MOBI to an M4B audiobook.

    This is the main orchestration function for single-file conversion.
    It coordinates format conversion, parsing, preprocessing, LLM
    enhancement, TTS synthesis (single or multi-voice), M4B assembly,
    and cleanup.

    Args:
        input_path: Path to the input file (EPUB, PDF, MOBI, etc.).
        output_dir: Directory for output files.
        voice: Kokoro voice identifier for single-voice mode.
        speed: Playback speed multiplier (0.5-2.0).
        device: Compute device ('auto', 'cuda', or 'cpu').
        voice_config: Path to multi-voice JSON config. Overrides *voice*.
        use_preprocessing: Whether to run the text preprocessor.
        substitutions_path: Path to a JSON pronunciation substitutions file.
        use_llm: Whether to run LLM-enhanced preprocessing.
        llm_model: Ollama model name for LLM preprocessing.
        direct_mode: If True, use direct Kokoro synthesis (bypass audiblez).
        skip_existing: If True, skip conversion when .m4b already exists.
        dry_run: If True, parse and display info without synthesizing.
        bitrate: AAC encoding bitrate for the M4B.

    Returns:
        Path to the generated .m4b file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If ffmpeg is missing or a required tool is unavailable.
        ValueError: If no chapters are found in the EPUB.
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    start_time = time.monotonic()

    # --- Step 1: Format conversion if needed ---
    console.print(f"\n[bold]Input:[/bold] {input_path.name}")

    epub_path = ensure_epub(input_path, output_dir=output_dir)
    if epub_path != input_path:
        console.print(
            f"[dim]Converted {input_path.suffix} to EPUB via Calibre[/dim]"
        )

    # --- Step 2: Parse EPUB ---
    console.print("[bold]Parsing EPUB...[/bold]")
    metadata, chapters = parse_epub(epub_path)

    if not chapters:
        raise ValueError(f"No chapters found in {epub_path}")

    # --- Step 3: Print book info ---
    _print_book_info(metadata, chapters)

    # --- Check for existing output ---
    safe_title = _sanitize_filename(metadata.title)
    m4b_path = output_dir / f"{safe_title}.m4b"

    if skip_existing and m4b_path.exists():
        console.print(
            f"[yellow]Output already exists:[/yellow] {m4b_path.name}\n"
            "[dim]Use --no-skip-existing to reconvert.[/dim]"
        )
        return m4b_path

    # --- Dry run: stop here ---
    if dry_run:
        console.print(
            "\n[bold yellow]DRY RUN[/bold yellow] -- no synthesis performed.\n"
        )
        return m4b_path

    # --- Verify ffmpeg ---
    if not check_ffmpeg():
        raise RuntimeError(
            "ffmpeg and/or ffprobe not found on PATH. "
            "Install ffmpeg before creating M4B files."
        )

    # --- Step 4: Preprocess text ---
    if use_preprocessing:
        console.print("[bold]Preprocessing text...[/bold]")

        subs_path = substitutions_path
        if subs_path is None and DEFAULT_SUBSTITUTIONS.exists():
            subs_path = DEFAULT_SUBSTITUTIONS

        for chapter in chapters:
            chapter.text = preprocess(chapter.text, substitutions_path=subs_path)

    # --- Step 5: LLM enhancement ---
    if use_llm:
        console.print(f"[bold]LLM enhancement[/bold] (model: {llm_model})...")

        llm = LLMPipeline(model=llm_model)
        if not llm.available:
            console.print(
                "[bold red]Ollama is not running.[/bold red] "
                "Skipping LLM enhancement.\n"
                "[dim]Start Ollama with: ollama serve[/dim]"
            )
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Enhancing chapters"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                llm_task = progress.add_task("LLM", total=len(chapters))
                for chapter in chapters:
                    try:
                        chapter.text = llm.process(chapter.text, mode="enhance")
                    except RuntimeError:
                        logger.warning(
                            "LLM enhancement failed for chapter %d (%s); using original text.",
                            chapter.index,
                            chapter.title,
                        )
                    progress.advance(llm_task)

    # --- Step 6: Voice resolution and synthesis ---
    tmp_dir = Path(tempfile.mkdtemp(prefix="pipeline_"))
    wav_paths: list[Path] = []
    chapter_titles: list[str] = []

    if voice_config is not None:
        # Multi-voice path
        console.print(
            f"[bold]Multi-voice synthesis[/bold] (config: {voice_config.name})"
        )
        voice_mgr = VoiceManager(voice_config, device=device)

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            synth_task = progress.add_task(
                "[bold blue]Synthesizing chapters",
                total=len(chapters),
            )

            for chapter in chapters:
                pipeline, voice_name, ch_speed = voice_mgr.get_pipeline_and_voice(
                    chapter.index
                )

                wav_path = tmp_dir / f"chapter_{chapter.index:04d}.wav"

                progress.update(
                    synth_task,
                    description=f"[cyan]Ch {chapter.index + 1}: {chapter.title[:30]}",
                )

                synthesize_chapter(
                    pipeline=pipeline,
                    text=chapter.text,
                    voice=voice_name,
                    speed=ch_speed,
                    output_path=wav_path,
                )

                wav_paths.append(wav_path)
                chapter_titles.append(chapter.title)
                progress.advance(synth_task)

    elif direct_mode:
        # Direct Kokoro mode (single voice, bypass audiblez)
        console.print(
            f"[bold]Direct Kokoro synthesis[/bold] (voice: {voice}, speed: {speed})"
        )

        pipeline = init_pipeline(lang_code=metadata.language, device=device)

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            synth_task = progress.add_task(
                "[bold blue]Synthesizing chapters",
                total=len(chapters),
            )

            for chapter in chapters:
                wav_path = tmp_dir / f"chapter_{chapter.index:04d}.wav"

                progress.update(
                    synth_task,
                    description=f"[cyan]Ch {chapter.index + 1}: {chapter.title[:30]}",
                )

                synthesize_chapter(
                    pipeline=pipeline,
                    text=chapter.text,
                    voice=voice,
                    speed=speed,
                    output_path=wav_path,
                )

                wav_paths.append(wav_path)
                chapter_titles.append(chapter.title)
                progress.advance(synth_task)

    else:
        # Default single-voice mode via kokoro_direct.synthesize_book
        console.print(
            f"[bold]Synthesizing[/bold] (voice: {voice}, speed: {speed})"
        )

        pipeline = init_pipeline(lang_code=metadata.language, device=device)

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            synth_task = progress.add_task(
                "[bold blue]Synthesizing chapters",
                total=len(chapters),
            )

            for chapter in chapters:
                wav_path = tmp_dir / f"chapter_{chapter.index:04d}.wav"

                progress.update(
                    synth_task,
                    description=f"[cyan]Ch {chapter.index + 1}: {chapter.title[:30]}",
                )

                synthesize_chapter(
                    pipeline=pipeline,
                    text=chapter.text,
                    voice=voice,
                    speed=speed,
                    output_path=wav_path,
                )

                wav_paths.append(wav_path)
                chapter_titles.append(chapter.title)
                progress.advance(synth_task)

    # --- Step 8: M4B assembly ---
    console.print("[bold]Assembling M4B audiobook...[/bold]")

    # Write cover image to temp file if available
    cover_path: Optional[Path] = None
    if metadata.cover_image:
        cover_path = tmp_dir / "cover.jpg"
        cover_path.write_bytes(metadata.cover_image)

    create_m4b(
        chapter_wav_files=wav_paths,
        chapter_titles=chapter_titles,
        output_path=m4b_path,
        cover_image=cover_path,
        bitrate=bitrate,
    )

    # --- Step 9: Cleanup temp WAVs ---
    cleanup_chapter_wavs(wav_paths)
    if cover_path and cover_path.exists():
        cover_path.unlink()
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    # --- Step 10: Print result ---
    elapsed = time.monotonic() - start_time
    file_size_mb = m4b_path.stat().st_size / (1024 * 1024)

    result_table = Table(show_header=False, box=None, padding=(0, 2))
    result_table.add_column("Key", style="bold green")
    result_table.add_column("Value")

    result_table.add_row("Output", str(m4b_path))
    result_table.add_row("Size", f"{file_size_mb:.1f} MB")
    result_table.add_row("Time", _format_duration(elapsed))
    result_table.add_row("Chapters", str(len(wav_paths)))

    console.print(Panel(result_table, title="Complete", border_style="green"))

    return m4b_path


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the pipeline CLI."""
    parser = argparse.ArgumentParser(
        prog="audiblez-kokoro",
        description=(
            "Convert EPUB, PDF, and MOBI files to M4B audiobooks "
            "using Kokoro TTS."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s book.epub\n"
            "  %(prog)s book.epub -v af_heart --cuda\n"
            "  %(prog)s --batch input/ -o output/\n"
            "  %(prog)s book.epub --voice-config config/voice_config.json\n"
            "  %(prog)s book.epub --preprocess --substitutions config/substitutions.json\n"
            "  %(prog)s book.epub --llm --llm-model llama3.2\n"
            "  %(prog)s book.epub --direct\n"
            "  %(prog)s document.pdf -o output/\n"
            "  %(prog)s --batch input/ --dry-run\n"
        ),
    )

    # Positional: input file
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Input file path (EPUB, PDF, MOBI, etc.)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output/)",
    )

    # Voice
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default=DEFAULT_VOICE,
        help=f"Narrator voice (default: {DEFAULT_VOICE})",
    )

    # Speed
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=DEFAULT_SPEED,
        help=f"Playback speed, 0.5-2.0 (default: {DEFAULT_SPEED})",
    )

    # Device selection (mutually exclusive)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--cuda", "-c",
        action="store_true",
        default=False,
        help="Force CUDA acceleration",
    )
    device_group.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force CPU-only synthesis",
    )

    # Batch mode
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        default=False,
        help="Batch mode: treat input as a directory of files",
    )

    # Multi-voice config
    parser.add_argument(
        "--voice-config",
        type=Path,
        default=None,
        help="Multi-voice JSON config file path",
    )

    # Preprocessing (--preprocess is default True, --no-preprocess disables)
    preprocess_group = parser.add_mutually_exclusive_group()
    preprocess_group.add_argument(
        "--preprocess", "-p",
        action="store_true",
        default=True,
        help="Enable text preprocessing (default: enabled)",
    )
    preprocess_group.add_argument(
        "--no-preprocess",
        action="store_true",
        default=False,
        help="Disable text preprocessing",
    )

    # Substitutions
    parser.add_argument(
        "--substitutions",
        type=Path,
        default=None,
        help="Path to pronunciation substitutions JSON (default: config/substitutions.json)",
    )

    # LLM enhancement
    parser.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help="Enable LLM-enhanced preprocessing via Ollama",
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3.2",
        help="Ollama model for LLM preprocessing (default: llama3.2)",
    )

    # Direct Kokoro mode
    parser.add_argument(
        "--direct", "-d",
        action="store_true",
        default=False,
        help="Direct Kokoro synthesis (bypass audiblez)",
    )

    # Skip existing
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip conversion if .m4b already exists (default: enabled)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        default=False,
        help="Reconvert even if .m4b output already exists",
    )

    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Parse and display info without synthesizing audio",
    )

    # Bitrate
    parser.add_argument(
        "--bitrate",
        type=str,
        default=DEFAULT_BITRATE,
        help=f"AAC bitrate for M4B encoding (default: {DEFAULT_BITRATE})",
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Main CLI entry point for the audiblez-kokoro pipeline.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate speed range
    if not 0.5 <= args.speed <= 2.0:
        console.print(
            f"[bold red]Error:[/bold red] Speed must be between 0.5 and 2.0, "
            f"got {args.speed}"
        )
        raise SystemExit(1)

    # Resolve device
    device = _resolve_device(args.cuda, args.cpu)

    # Resolve preprocessing flag
    use_preprocessing = not args.no_preprocess

    # Resolve skip-existing flag
    skip_existing = not args.no_skip_existing

    # --- Batch mode ---
    if args.batch:
        if args.input is None:
            console.print(
                "[bold red]Error:[/bold red] --batch requires an input directory.\n"
                "Usage: audiblez-kokoro --batch input_dir/ -o output/"
            )
            raise SystemExit(1)

        input_dir = args.input
        if not input_dir.is_dir():
            console.print(
                f"[bold red]Error:[/bold red] Not a directory: {input_dir}"
            )
            raise SystemExit(1)

        use_cuda = device in ("cuda", "auto")

        results = batch_convert(
            input_dir=input_dir,
            output_dir=args.output,
            voice=args.voice,
            speed=args.speed,
            voice_config=args.voice_config,
            use_preprocessing=use_preprocessing,
            substitutions=args.substitutions,
            use_cuda=use_cuda,
            skip_existing=skip_existing,
            dry_run=args.dry_run,
        )

        if any(r["status"] == "error" for r in results):
            raise SystemExit(1)
        return

    # --- Single-file mode ---
    if args.input is None:
        parser.print_help()
        console.print(
            "\n[bold red]Error:[/bold red] An input file is required.\n"
            "Usage: audiblez-kokoro book.epub [-o output/]"
        )
        raise SystemExit(1)

    input_path = args.input
    if not input_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] File not found: {input_path}"
        )
        raise SystemExit(1)

    try:
        convert_single(
            input_path=input_path,
            output_dir=args.output,
            voice=args.voice,
            speed=args.speed,
            device=device,
            voice_config=args.voice_config,
            use_preprocessing=use_preprocessing,
            substitutions_path=args.substitutions,
            use_llm=args.llm,
            llm_model=args.llm_model,
            direct_mode=args.direct,
            skip_existing=skip_existing,
            dry_run=args.dry_run,
            bitrate=args.bitrate,
        )
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
