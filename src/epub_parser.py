"""
EPUB parsing module for the audiblez-kokoro-pipeline.

Extracts chapters and metadata from .epub files for text-to-speech processing
using the Kokoro TTS engine. Handles HTML cleaning, language detection,
and chapter segmentation with duration estimation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub


@dataclass
class Chapter:
    """A single chapter extracted from an EPUB file."""

    index: int
    title: str
    text: str
    html: str
    word_count: int
    estimated_duration_sec: float

    @classmethod
    def from_html(cls, index: int, title: str, html: str) -> Chapter:
        """Create a Chapter from raw HTML content."""
        text = extract_text_from_html(html)
        word_count = len(text.split()) if text.strip() else 0
        estimated_duration_sec = word_count / 150.0 * 60.0
        return cls(
            index=index,
            title=title,
            text=text,
            html=html,
            word_count=word_count,
            estimated_duration_sec=estimated_duration_sec,
        )


@dataclass
class BookMetadata:
    """Metadata extracted from an EPUB file."""

    title: str
    author: str
    language: str
    cover_image: Optional[bytes]
    chapter_count: int


def extract_text_from_html(html: str) -> str:
    """
    Extract clean text from HTML content.

    Pulls text from p, h1-h4, li, and title tags. Appends periods to lines
    that lack terminal punctuation to ensure proper TTS phrasing.

    Args:
        html: Raw HTML string.

    Returns:
        Cleaned plain text with one paragraph per line.
    """
    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "html.parser")

    target_tags = ["title", "h1", "h2", "h3", "h4", "p", "li"]
    elements = soup.find_all(target_tags)

    lines = []
    for element in elements:
        text = element.get_text(separator=" ", strip=True)
        if not text:
            continue

        # Normalize internal whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Append period if missing terminal punctuation
        text = _ensure_terminal_punctuation(text)

        lines.append(text)

    return "\n".join(lines)


def _ensure_terminal_punctuation(text: str) -> str:
    """
    Append a period to text that lacks terminal punctuation.

    Terminal punctuation characters: . ! ? : ; ... (ellipsis)

    Args:
        text: A single line of text.

    Returns:
        Text guaranteed to end with punctuation.
    """
    if not text:
        return text

    terminal_chars = ".!?;:"
    stripped = text.rstrip()

    if stripped and stripped[-1] not in terminal_chars:
        return stripped + "."

    return stripped


def detect_language(text: str) -> str:
    """
    Detect the language of a text sample using simple heuristics.

    Returns Kokoro TTS language codes:
        'a' - American English (default)
        'b' - British English
        'e' - Spanish
        'f' - French
        'h' - Hindi
        'i' - Italian
        'j' - Japanese
        'p' - Portuguese
        'z' - Chinese

    The detection uses Unicode script ranges and common word frequency
    analysis. Falls back to American English when uncertain.

    Args:
        text: Sample text for language detection.

    Returns:
        Single-character Kokoro language code.
    """
    if not text or not text.strip():
        return "a"

    # Check for CJK characters (Japanese and Chinese)
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    hiragana_katakana_count = len(
        re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", text)
    )

    total_chars = len(text.strip())
    if total_chars == 0:
        return "a"

    # Japanese: has hiragana/katakana
    if hiragana_katakana_count > total_chars * 0.1:
        return "j"

    # Chinese: has CJK but no hiragana/katakana
    if cjk_count > total_chars * 0.1:
        return "z"

    # Hindi: Devanagari script
    devanagari_count = len(re.findall(r"[\u0900-\u097f]", text))
    if devanagari_count > total_chars * 0.1:
        return "h"

    # For Latin-script languages, use common word detection
    text_lower = text.lower()
    words = re.findall(r"\b[a-z]+\b", text_lower)

    if not words:
        return "a"

    word_set = set(words)

    # Spanish markers
    spanish_markers = {
        "el", "la", "los", "las", "de", "del", "en", "que", "por",
        "con", "una", "para", "como", "pero", "esto", "esta", "ese",
        "son", "tiene", "hacer", "puede", "desde", "entre", "sobre",
    }

    # French markers
    french_markers = {
        "le", "la", "les", "des", "une", "dans", "que", "qui", "est",
        "pour", "pas", "sur", "avec", "tout", "mais", "cette", "sont",
        "nous", "vous", "leur", "fait", "peut", "aussi", "comme", "avoir",
    }

    # Italian markers
    italian_markers = {
        "il", "la", "di", "che", "non", "una", "per", "sono", "della",
        "con", "gli", "dal", "dei", "delle", "nella", "alla", "questo",
        "quella", "anche", "come", "essere", "hanno", "fatto", "molto",
    }

    # Portuguese markers
    portuguese_markers = {
        "de", "que", "uma", "para", "com", "por", "mas", "como",
        "mais", "quando", "muito", "tambem", "nos", "ela", "isso",
        "mesmo", "pode", "tem", "foi", "ser", "isto", "aqui", "voce",
    }

    # Score each language by marker overlap
    scores = {
        "e": len(word_set & spanish_markers),
        "f": len(word_set & french_markers),
        "i": len(word_set & italian_markers),
        "p": len(word_set & portuguese_markers),
    }

    # Require a minimum threshold of marker hits
    best_lang = max(scores, key=scores.get)
    if scores[best_lang] >= 3:
        return best_lang

    # Default to American English
    return "a"


def _extract_cover_image(book: epub.EpubBook) -> Optional[bytes]:
    """
    Attempt to extract the cover image from an EPUB book.

    Tries multiple strategies:
    1. Look for an item with 'cover' in the ID
    2. Look for an item with 'cover-image' properties
    3. Look for the first image item

    Args:
        book: Parsed EpubBook object.

    Returns:
        Raw image bytes if found, None otherwise.
    """
    # Strategy 1: item with 'cover' in ID
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            item_id = (item.get_id() or "").lower()
            if "cover" in item_id:
                return item.get_content()

    # Strategy 2: item with cover-image property
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            name = (item.get_name() or "").lower()
            if "cover" in name:
                return item.get_content()

    # Strategy 3: first image
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            return item.get_content()

    return None


def _extract_title_from_html(html: str) -> str:
    """
    Extract a chapter title from HTML content.

    Looks for the first heading tag (h1-h4) or title tag.

    Args:
        html: Raw HTML string.

    Returns:
        Title string, or empty string if none found.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    for tag_name in ["h1", "h2", "h3", "h4", "title"]:
        tag = soup.find(tag_name)
        if tag:
            title = tag.get_text(strip=True)
            if title:
                return title

    return ""


def parse_epub(epub_path: Path) -> tuple[BookMetadata, list[Chapter]]:
    """
    Parse an EPUB file and extract metadata and chapters.

    Reads the EPUB, extracts book-level metadata (title, author, language,
    cover image), then iterates through document items to build a list of
    chapters. Only includes sections with substantial text content (>200
    characters after HTML stripping).

    Args:
        epub_path: Path to the .epub file.

    Returns:
        A tuple of (BookMetadata, list[Chapter]).

    Raises:
        FileNotFoundError: If the epub_path does not exist.
        ebooklib.epub.EpubException: If the file is not a valid EPUB.
    """
    epub_path = Path(epub_path)
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB file not found: {epub_path}")

    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})

    # Extract metadata with safe fallbacks
    title = book.get_metadata("DC", "title")
    title = title[0][0] if title else "Unknown Title"

    author = book.get_metadata("DC", "creator")
    author = author[0][0] if author else "Unknown Author"

    language_meta = book.get_metadata("DC", "language")
    language_raw = language_meta[0][0] if language_meta else "en"

    cover_image = _extract_cover_image(book)

    # Extract chapters from document items
    chapters: list[Chapter] = []
    chapter_index = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_content = item.get_content().decode("utf-8", errors="replace")

        # Extract text and check minimum length
        text = extract_text_from_html(html_content)
        if len(text) <= 200:
            continue

        # Determine chapter title
        title_from_html = _extract_title_from_html(html_content)
        chapter_title = title_from_html if title_from_html else f"Chapter {chapter_index + 1}"

        chapter = Chapter.from_html(
            index=chapter_index,
            title=chapter_title,
            html=html_content,
        )
        chapters.append(chapter)
        chapter_index += 1

    # Detect language from the combined text of all chapters
    combined_text = " ".join(ch.text[:500] for ch in chapters[:5])
    language = detect_language(combined_text) if combined_text.strip() else _map_language_code(language_raw)

    metadata = BookMetadata(
        title=title,
        author=author,
        language=language,
        cover_image=cover_image,
        chapter_count=len(chapters),
    )

    return metadata, chapters


def _map_language_code(iso_code: str) -> str:
    """
    Map an ISO 639-1 language code to a Kokoro TTS language code.

    Args:
        iso_code: ISO language code (e.g., 'en', 'es', 'fr').

    Returns:
        Kokoro language code character.
    """
    mapping = {
        "en": "a",
        "en-us": "a",
        "en-gb": "b",
        "es": "e",
        "fr": "f",
        "hi": "h",
        "it": "i",
        "ja": "j",
        "pt": "p",
        "zh": "z",
        "zh-cn": "z",
        "zh-tw": "z",
    }

    normalized = iso_code.lower().strip()
    if normalized in mapping:
        return mapping[normalized]

    # Try just the first two characters
    prefix = normalized[:2]
    return mapping.get(prefix, "a")
