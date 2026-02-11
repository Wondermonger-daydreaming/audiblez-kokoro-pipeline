"""
Format conversion wrapper for non-EPUB ebook formats.

Converts PDF, MOBI, AZW3, DOCX, FB2, RTF, and other formats to EPUB
using Calibre's ebook-convert CLI tool. This module is OPTIONAL and
requires Calibre to be installed on the system.

Install Calibre:
    - Linux:   sudo apt install calibre
    - macOS:   brew install calibre
    - Windows: https://calibre-ebook.com/download

Usage:
    from format_wrapper import ensure_epub
    epub_path = ensure_epub(Path("book.pdf"))

Or standalone:
    python format_wrapper.py input.pdf -o /tmp/output/
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {'.pdf', '.mobi', '.azw3', '.azw', '.docx', '.fb2', '.rtf'}


def check_calibre() -> bool:
    """Check if Calibre's ebook-convert tool is available on the system PATH.

    Returns:
        True if ebook-convert is found, False otherwise.
    """
    return shutil.which('ebook-convert') is not None


def convert_to_epub(input_path: Path, output_dir: Optional[Path] = None) -> Path:
    """Convert a supported ebook format to EPUB using Calibre's ebook-convert.

    Args:
        input_path: Path to the input file. Must be a supported format.
        output_dir: Directory for the output EPUB. Defaults to the same
                    directory as the input file.

    Returns:
        Path to the generated EPUB file.

    Raises:
        RuntimeError: If Calibre is not installed or conversion fails.
        FileNotFoundError: If the input file does not exist.
        ValueError: If the input format is not supported.
    """
    input_path = Path(input_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    if not check_calibre():
        raise RuntimeError(
            "Calibre's ebook-convert tool is not installed or not on PATH.\n"
            "Install Calibre to enable format conversion:\n"
            "  Linux:   sudo apt install calibre\n"
            "  macOS:   brew install calibre\n"
            "  Windows: https://calibre-ebook.com/download"
        )

    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    output_epub = output_dir / f"{input_path.stem}.epub"

    if suffix == '.pdf':
        logger.warning(
            "PDF conversion quality varies significantly depending on the PDF structure. "
            "Scanned PDFs (image-based) will produce poor results. Text-based PDFs with "
            "simple layouts convert best. Consider using the original EPUB if available."
        )

    logger.info("Converting %s -> %s", input_path.name, output_epub.name)

    try:
        result = subprocess.run(
            ['ebook-convert', str(input_path), str(output_epub)],
            capture_output=True,
            check=True,
            timeout=300,
        )
        logger.debug("ebook-convert stdout: %s", result.stdout.decode(errors='replace'))
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Conversion timed out after 300 seconds for: {input_path.name}"
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors='replace') if e.stderr else 'No error output'
        raise RuntimeError(
            f"Calibre conversion failed for {input_path.name}:\n{stderr}"
        )

    if not output_epub.exists():
        raise RuntimeError(
            f"Conversion appeared to succeed but output file not found: {output_epub}"
        )

    logger.info("Conversion complete: %s (%.1f KB)", output_epub.name, output_epub.stat().st_size / 1024)
    return output_epub


def ensure_epub(input_path: Path, output_dir: Optional[Path] = None) -> Path:
    """Ensure the input is an EPUB, converting if necessary.

    If the input is already an EPUB file, returns it as-is. If it is a
    supported non-EPUB format, converts it via Calibre. Raises an error
    for unsupported formats.

    Args:
        input_path: Path to any supported ebook file.
        output_dir: Directory for converted output. Only used if conversion
                    is needed. Defaults to the input file's directory.

    Returns:
        Path to an EPUB file (either the original or newly converted).

    Raises:
        ValueError: If the format is not supported and not EPUB.
        RuntimeError: If Calibre is not installed (when conversion is needed).
        FileNotFoundError: If the input file does not exist.
    """
    input_path = Path(input_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == '.epub':
        logger.info("Input is already EPUB: %s", input_path.name)
        return input_path

    if suffix in SUPPORTED_FORMATS:
        logger.info("Converting %s format to EPUB via Calibre", suffix)
        return convert_to_epub(input_path, output_dir)

    all_formats = sorted(SUPPORTED_FORMATS | {'.epub'})
    raise ValueError(
        f"Unsupported format '{suffix}' for file: {input_path.name}\n"
        f"Supported formats: {', '.join(all_formats)}"
    )


def main():
    """CLI entry point for standalone format conversion."""
    parser = argparse.ArgumentParser(
        description='Convert ebook formats to EPUB using Calibre.',
        epilog=(
            f'Supported input formats: {", ".join(sorted(SUPPORTED_FORMATS))}\n'
            'Requires Calibre (ebook-convert) to be installed.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'input',
        type=Path,
        help='Path to the input ebook file',
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=None,
        help='Output directory for the EPUB (default: same as input)',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging output',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    try:
        result = ensure_epub(args.input, args.output_dir)
        print(f"Output: {result}")
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
