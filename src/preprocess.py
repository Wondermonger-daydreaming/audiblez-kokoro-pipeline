"""
Text preprocessing for the audiblez-kokoro TTS pipeline.

Cleans and normalizes text extracted from EPUBs before synthesis.
Each function is idempotent: applying it twice yields the same result
as applying it once.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Abbreviation table
# ---------------------------------------------------------------------------

ABBREVIATIONS: dict[str, str] = {
    "Dr.": "Doctor",
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miz",
    "Prof.": "Professor",
    "St.": "Saint",
    "vs.": "versus",
    "etc.": "et cetera",
    "e.g.": "for example",
    "i.e.": "that is",
    "Jr.": "Junior",
    "Sr.": "Senior",
    "Ave.": "Avenue",
    "Blvd.": "Boulevard",
    "approx.": "approximately",
    "dept.": "department",
    "govt.": "government",
    "Jan.": "January",
    "Feb.": "February",
    "Mar.": "March",
    "Apr.": "April",
    "Jun.": "June",
    "Jul.": "July",
    "Aug.": "August",
    "Sep.": "September",
    "Sept.": "September",
    "Oct.": "October",
    "Nov.": "November",
    "Dec.": "December",
}


# ---------------------------------------------------------------------------
# Number-to-words engine
# ---------------------------------------------------------------------------

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]

_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]

_ORDINAL_ONES = [
    "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth",
]

_ORDINAL_TENS = [
    "", "", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth",
    "seventieth", "eightieth", "ninetieth",
]


def _int_to_words(n: int) -> str:
    """Convert an integer 0..999_999_999 to English words (recursive)."""
    if n < 0:
        return "negative " + _int_to_words(-n)
    if n == 0:
        return "zero"
    return _positive_int_to_words(n).strip()


def _positive_int_to_words(n: int) -> str:
    """Recursive core: returns words for 1..999_999_999 (no leading 'zero')."""
    if n == 0:
        return ""

    if n < 20:
        return _ONES[n]

    if n < 100:
        rest = _positive_int_to_words(n % 10)
        return _TENS[n // 10] + ("-" + rest if rest else "")

    if n < 1_000:
        rest = _positive_int_to_words(n % 100)
        return _ONES[n // 100] + " hundred" + (" " + rest if rest else "")

    if n < 1_000_000:
        thousands = _positive_int_to_words(n // 1_000)
        rest = _positive_int_to_words(n % 1_000)
        return thousands + " thousand" + (" " + rest if rest else "")

    if n < 1_000_000_000:
        millions = _positive_int_to_words(n // 1_000_000)
        rest = _positive_int_to_words(n % 1_000_000)
        return millions + " million" + (" " + rest if rest else "")

    return str(n)  # beyond scope — pass through


def _int_to_ordinal(n: int) -> str:
    """Convert a positive integer to an ordinal word (first, second, ...)."""
    if n <= 0:
        return str(n)

    if n < 20:
        return _ORDINAL_ONES[n]

    if n < 100:
        ones = n % 10
        tens = n // 10
        if ones == 0:
            return _ORDINAL_TENS[tens]
        return _TENS[tens] + "-" + _ORDINAL_ONES[ones]

    # For larger ordinals, build cardinal then append suffix
    words = _int_to_words(n)
    # Replace trailing word with its ordinal form
    if words.endswith("one"):
        return words[:-3] + "first"
    if words.endswith("two"):
        return words[:-3] + "second"
    if words.endswith("three"):
        return words[:-5] + "third"
    if words.endswith("four"):
        return words[:-4] + "fourth"
    if words.endswith("five"):
        return words[:-4] + "fifth"
    if words.endswith("six"):
        return words[:-3] + "sixth"
    if words.endswith("seven"):
        return words[:-5] + "seventh"
    if words.endswith("eight"):
        return words[:-5] + "eighth"
    if words.endswith("nine"):
        return words[:-4] + "ninth"
    if words.endswith("twelve"):
        return words[:-6] + "twelfth"
    if words.endswith("y"):
        return words[:-1] + "ieth"
    return words + "th"


def _year_to_words(n: int) -> str:
    """Convert a four-digit year to spoken form.

    1984 -> nineteen eighty-four
    2000 -> two thousand
    2001 -> two thousand one
    2024 -> twenty twenty-four
    """
    if n < 100 or n >= 10000:
        return _int_to_words(n)

    hi = n // 100
    lo = n % 100

    # Years like 2000, 1900
    if lo == 0:
        if hi % 10 == 0:
            return _int_to_words(hi // 10) + " thousand"
        return _int_to_words(hi) + " hundred"

    # Years like 2001-2009
    if lo < 10 and hi % 10 == 0 and hi >= 20:
        return _int_to_words(hi // 10) + " thousand " + _int_to_words(lo)

    # Standard split: 1984 -> nineteen eighty-four
    return _int_to_words(hi) + " " + _int_to_words(lo)


# ---------------------------------------------------------------------------
# Roman numerals
# ---------------------------------------------------------------------------

_ROMAN_VALUES = [
    ("M", 1000), ("CM", 900), ("D", 500), ("CD", 400),
    ("C", 100), ("XC", 90), ("L", 50), ("XL", 40),
    ("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1),
]


def _roman_to_int(s: str) -> int | None:
    """Parse a Roman numeral string, return int or None if invalid."""
    if not s or not re.fullmatch(r"[MDCLXVI]+", s):
        return None

    total = 0
    idx = 0
    for symbol, value in _ROMAN_VALUES:
        while idx < len(s) and s[idx:idx + len(symbol)] == symbol:
            total += value
            idx += len(symbol)

    if idx != len(s):
        return None
    return total if total > 0 else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_substitutions(path: Path) -> dict[str, str]:
    """Load a JSON pronunciation dictionary from *path*.

    Returns an empty dict if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return {str(k): str(v) for k, v in data.items()}


def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations to their spoken forms.

    Operates on whole-word boundaries so 'Dr.' inside a larger token
    is left alone.
    """
    for abbr, expansion in ABBREVIATIONS.items():
        # Build a pattern that matches the abbreviation at word boundaries.
        # The abbreviation ends with '.', so we anchor on the left with \b
        # and on the right we require a space, end-of-string, or punctuation
        # (but NOT another letter, which would indicate it's mid-word).
        pattern = r"(?<!\w)" + re.escape(abbr) + r"(?=\s|$|[,;:!?\"\')])"
        text = re.sub(pattern, expansion, text)
    return text


def expand_numbers(text: str) -> str:
    """Convert numeric expressions to spoken English words.

    Handles: integers, ordinals (1st, 2nd, ...), years in context,
    currency ($N.NN), percentages (N%), and decimals (N.NN).
    """
    # --- Currency: $1,234.56 ---
    def _currency_repl(m: re.Match) -> str:
        dollars_str = m.group(1).replace(",", "")
        cents_str = m.group(2)  # may be None
        dollars = int(dollars_str)
        d_word = _int_to_words(dollars)
        d_unit = "dollar" if dollars == 1 else "dollars"
        if cents_str is not None:
            cents = int(cents_str)
            if cents == 0:
                return f"{d_word} {d_unit}"
            c_word = _int_to_words(cents)
            c_unit = "cent" if cents == 1 else "cents"
            return f"{d_word} {d_unit} and {c_word} {c_unit}"
        return f"{d_word} {d_unit}"

    text = re.sub(
        r"\$([0-9,]+)(?:\.([0-9]{1,2}))?",
        _currency_repl,
        text,
    )

    # --- Percentages: 85% ---
    def _pct_repl(m: re.Match) -> str:
        n = int(m.group(1).replace(",", ""))
        return _int_to_words(n) + " percent"

    text = re.sub(r"(\d[\d,]*)%", _pct_repl, text)

    # --- Ordinals: 1st, 2nd, 3rd, 11th, 21st, ... ---
    def _ordinal_repl(m: re.Match) -> str:
        n = int(m.group(1).replace(",", ""))
        return _int_to_ordinal(n)

    text = re.sub(r"(\d[\d,]*)(?:st|nd|rd|th)\b", _ordinal_repl, text)

    # --- Years in context ---
    # Match 4-digit numbers that look like years (1000-2999)
    # preceded by "in", "of", "year", "circa", or start-of-string
    def _year_repl(m: re.Match) -> str:
        prefix = m.group(1)
        n = int(m.group(2))
        return prefix + _year_to_words(n)

    text = re.sub(
        r"((?:in|of|year|circa|from|since|until|by|around)\s+)"
        r"(\b[12]\d{3}\b)",
        _year_repl,
        text,
        flags=re.IGNORECASE,
    )

    # --- Decimals: 3.14 (but not already-processed currency/dates) ---
    def _decimal_repl(m: re.Match) -> str:
        whole = m.group(1)
        frac = m.group(2)
        whole_word = _int_to_words(int(whole))
        frac_words = " ".join(_ONES[int(d)] if int(d) > 0 else "zero" for d in frac)
        return whole_word + " point " + frac_words

    text = re.sub(r"\b(\d+)\.(\d+)\b", _decimal_repl, text)

    # --- Plain integers (must come last to avoid clobbering) ---
    def _int_repl(m: re.Match) -> str:
        raw = m.group(0).replace(",", "")
        n = int(raw)
        if n > 999_999_999:
            return m.group(0)  # out of range, pass through
        return _int_to_words(n)

    text = re.sub(r"\b\d[\d,]*\b", _int_repl, text)

    return text


def expand_roman_numerals(text: str) -> str:
    """Expand Roman numerals that appear after contextual keywords.

    Only expands when preceded by Chapter, Part, Book, Volume, Act, or Scene.
    Standalone Roman-numeral-like tokens (I, V, X, etc.) are left alone.
    """
    keywords = r"(?:Chapter|Part|Book|Volume|Act|Scene)"

    def _repl(m: re.Match) -> str:
        prefix = m.group(1)  # keyword + space
        roman = m.group(2)
        value = _roman_to_int(roman)
        if value is None:
            return m.group(0)
        return prefix + _int_to_words(value)

    pattern = rf"({keywords}\s+)([MDCLXVI]+)\b"
    text = re.sub(pattern, _repl, text)
    return text


def normalize_punctuation(text: str) -> str:
    """Normalize dashes, ellipses, quotes, and whitespace for TTS."""
    # Double-hyphen to em-dash
    text = text.replace("--", "\u2014")

    # Three dots to ellipsis character
    text = re.sub(r"\.{3}", "\u2026", text)

    # Curly quotes to straight
    text = text.replace("\u2018", "'")   # left single
    text = text.replace("\u2019", "'")   # right single
    text = text.replace("\u201C", '"')   # left double
    text = text.replace("\u201D", '"')   # right double

    # Strip excessive whitespace (multiple spaces/tabs to single space)
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Collapse multiple blank lines to at most two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def apply_custom_substitutions(text: str, subs: dict[str, str]) -> str:
    """Apply user-defined whole-word replacements.

    Each key in *subs* is matched as a whole word (case-sensitive) and
    replaced with its value.
    """
    for original, replacement in subs.items():
        pattern = r"\b" + re.escape(original) + r"\b"
        text = re.sub(pattern, replacement, text)
    return text


def preprocess(text: str, substitutions_path: Optional[Path] = None) -> str:
    """Run the full preprocessing pipeline.

    Order: abbreviations -> numbers -> roman numerals -> custom
    substitutions -> punctuation normalization.
    """
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    text = expand_roman_numerals(text)

    if substitutions_path is not None:
        subs = load_substitutions(substitutions_path)
        text = apply_custom_substitutions(text, subs)

    text = normalize_punctuation(text)
    return text
