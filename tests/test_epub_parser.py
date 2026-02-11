"""
Unit tests for the epub_parser module.

Tests cover HTML text extraction, language detection, Chapter dataclass
construction, edge cases (empty content, malformed HTML), and the
punctuation-appending logic. No actual .epub files required.
"""

import pytest

from epub_parser import (
    Chapter,
    BookMetadata,
    extract_text_from_html,
    detect_language,
    _ensure_terminal_punctuation,
    _extract_title_from_html,
    _map_language_code,
)


# ---------------------------------------------------------------------------
# extract_text_from_html
# ---------------------------------------------------------------------------


class TestExtractTextFromHtml:
    """Tests for the extract_text_from_html function."""

    def test_simple_paragraphs(self):
        html = "<html><body><p>First paragraph.</p><p>Second paragraph.</p></body></html>"
        result = extract_text_from_html(html)
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_heading_extraction(self):
        html = "<h1>Chapter Title</h1><p>Body text here.</p>"
        result = extract_text_from_html(html)
        lines = result.split("\n")
        assert any("Chapter Title" in line for line in lines)
        assert any("Body text here." in line for line in lines)

    def test_multiple_heading_levels(self):
        html = "<h1>H1.</h1><h2>H2.</h2><h3>H3.</h3><h4>H4.</h4>"
        result = extract_text_from_html(html)
        assert "H1." in result
        assert "H2." in result
        assert "H3." in result
        assert "H4." in result

    def test_list_items(self):
        html = "<ul><li>Item one.</li><li>Item two.</li></ul>"
        result = extract_text_from_html(html)
        assert "Item one." in result
        assert "Item two." in result

    def test_title_tag(self):
        html = "<html><head><title>Book Title</title></head><body><p>Content.</p></body></html>"
        result = extract_text_from_html(html)
        assert "Book Title" in result

    def test_empty_html_returns_empty_string(self):
        assert extract_text_from_html("") == ""
        assert extract_text_from_html("   ") == ""
        assert extract_text_from_html(None) == ""

    def test_html_with_no_target_tags(self):
        html = "<html><body><div>This is in a div.</div><span>And a span.</span></body></html>"
        result = extract_text_from_html(html)
        assert result == ""

    def test_whitespace_normalization(self):
        html = "<p>  Multiple   spaces   and\nnewlines\there.  </p>"
        result = extract_text_from_html(html)
        assert "Multiple spaces and newlines here." in result

    def test_nested_tags_in_paragraphs(self):
        html = "<p>Text with <strong>bold</strong> and <em>italic</em> words.</p>"
        result = extract_text_from_html(html)
        assert "Text with bold and italic words." in result

    def test_period_appended_to_lines_without_punctuation(self):
        html = "<p>This line has no ending punctuation</p>"
        result = extract_text_from_html(html)
        assert result.strip().endswith(".")

    def test_period_not_doubled(self):
        html = "<p>This line already ends with a period.</p>"
        result = extract_text_from_html(html)
        assert not result.strip().endswith("..")

    def test_exclamation_preserved(self):
        html = "<p>What an exciting line!</p>"
        result = extract_text_from_html(html)
        assert result.strip().endswith("!")
        assert not result.strip().endswith("!.")

    def test_question_mark_preserved(self):
        html = "<p>Is this a question?</p>"
        result = extract_text_from_html(html)
        assert result.strip().endswith("?")

    def test_malformed_html(self):
        html = "<p>Unclosed paragraph<p>Another one</p>"
        result = extract_text_from_html(html)
        # BeautifulSoup should handle this gracefully
        assert "Unclosed paragraph" in result or "Another one" in result

    def test_html_entities_decoded(self):
        html = "<p>Smart &amp; simple &mdash; that&#39;s the goal.</p>"
        result = extract_text_from_html(html)
        assert "&amp;" not in result
        assert "Smart" in result

    def test_empty_tags_skipped(self):
        html = "<p></p><p>Real content.</p><p>   </p>"
        result = extract_text_from_html(html)
        lines = [line for line in result.split("\n") if line.strip()]
        assert len(lines) == 1
        assert "Real content." in lines[0]


# ---------------------------------------------------------------------------
# _ensure_terminal_punctuation
# ---------------------------------------------------------------------------


class TestEnsureTerminalPunctuation:
    """Tests for the punctuation-appending helper."""

    def test_appends_period_when_missing(self):
        assert _ensure_terminal_punctuation("Hello world") == "Hello world."

    def test_preserves_period(self):
        assert _ensure_terminal_punctuation("Hello world.") == "Hello world."

    def test_preserves_exclamation(self):
        assert _ensure_terminal_punctuation("Hello world!") == "Hello world!"

    def test_preserves_question_mark(self):
        assert _ensure_terminal_punctuation("Hello world?") == "Hello world?"

    def test_preserves_colon(self):
        assert _ensure_terminal_punctuation("Chapter One:") == "Chapter One:"

    def test_preserves_semicolon(self):
        assert _ensure_terminal_punctuation("First clause;") == "First clause;"

    def test_empty_string(self):
        assert _ensure_terminal_punctuation("") == ""

    def test_strips_trailing_whitespace(self):
        assert _ensure_terminal_punctuation("Hello world   ") == "Hello world."

    def test_trailing_whitespace_with_punctuation(self):
        assert _ensure_terminal_punctuation("Hello world.   ") == "Hello world."


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    """Tests for the language detection heuristic."""

    def test_english_default(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert detect_language(text) == "a"

    def test_spanish_detection(self):
        text = (
            "El gato esta en la casa. Los ninos juegan en el parque "
            "con una pelota para divertirse entre ellos."
        )
        assert detect_language(text) == "e"

    def test_french_detection(self):
        text = (
            "Le chat est dans la maison. Les enfants jouent dans "
            "le parc avec une balle pour nous amuser."
        )
        assert detect_language(text) == "f"

    def test_italian_detection(self):
        text = (
            "Il gatto e nella casa. I bambini giocano nel parco "
            "con una palla per questo anche molto bello."
        )
        assert detect_language(text) == "i"

    def test_portuguese_detection(self):
        text = (
            "Uma casa muito bonita que tem mais coisas para voce "
            "quando isso foi tambem mesmo aqui."
        )
        assert detect_language(text) == "p"

    def test_japanese_detection(self):
        text = "これは日本語のテストです。猫は家の中にいます。"
        assert detect_language(text) == "j"

    def test_chinese_detection(self):
        text = "这是一个中文测试。猫在房子里面。今天天气很好。"
        assert detect_language(text) == "z"

    def test_hindi_detection(self):
        text = "यह एक हिंदी परीक्षण है। बिल्ली घर में है।"
        assert detect_language(text) == "h"

    def test_empty_text_returns_default(self):
        assert detect_language("") == "a"
        assert detect_language("   ") == "a"
        assert detect_language(None) == "a"

    def test_mixed_cjk_with_hiragana_is_japanese(self):
        text = "東京は日本の首都です。とても大きな都市です。"
        assert detect_language(text) == "j"

    def test_pure_cjk_without_kana_is_chinese(self):
        text = "北京是中国的首都。这是一个很大的城市。"
        assert detect_language(text) == "z"


# ---------------------------------------------------------------------------
# _map_language_code
# ---------------------------------------------------------------------------


class TestMapLanguageCode:
    """Tests for ISO to Kokoro language code mapping."""

    def test_english_us(self):
        assert _map_language_code("en") == "a"
        assert _map_language_code("en-US") == "a"

    def test_english_gb(self):
        assert _map_language_code("en-GB") == "b"

    def test_spanish(self):
        assert _map_language_code("es") == "e"

    def test_french(self):
        assert _map_language_code("fr") == "f"

    def test_italian(self):
        assert _map_language_code("it") == "i"

    def test_japanese(self):
        assert _map_language_code("ja") == "j"

    def test_portuguese(self):
        assert _map_language_code("pt") == "p"

    def test_chinese_variants(self):
        assert _map_language_code("zh") == "z"
        assert _map_language_code("zh-CN") == "z"
        assert _map_language_code("zh-TW") == "z"

    def test_hindi(self):
        assert _map_language_code("hi") == "h"

    def test_unknown_defaults_to_american_english(self):
        assert _map_language_code("de") == "a"
        assert _map_language_code("ru") == "a"
        assert _map_language_code("ko") == "a"


# ---------------------------------------------------------------------------
# _extract_title_from_html
# ---------------------------------------------------------------------------


class TestExtractTitleFromHtml:
    """Tests for the chapter title extraction helper."""

    def test_h1_title(self):
        html = "<h1>My Chapter</h1><p>Some content.</p>"
        assert _extract_title_from_html(html) == "My Chapter"

    def test_h2_title_when_no_h1(self):
        html = "<h2>Section Title</h2><p>Content.</p>"
        assert _extract_title_from_html(html) == "Section Title"

    def test_prefers_h1_over_h2(self):
        html = "<h2>Second</h2><h1>First</h1><p>Content.</p>"
        assert _extract_title_from_html(html) == "First"

    def test_empty_html(self):
        assert _extract_title_from_html("") == ""
        assert _extract_title_from_html(None) == ""

    def test_no_headings(self):
        html = "<p>Just a paragraph.</p>"
        assert _extract_title_from_html(html) == ""

    def test_empty_heading_skipped(self):
        html = "<h1></h1><h2>Actual Title</h2>"
        assert _extract_title_from_html(html) == "Actual Title"


# ---------------------------------------------------------------------------
# Chapter dataclass
# ---------------------------------------------------------------------------


class TestChapter:
    """Tests for the Chapter dataclass and its factory method."""

    def test_from_html_basic(self):
        html = "<h1>Test Chapter</h1><p>This is a test paragraph with enough words.</p>"
        chapter = Chapter.from_html(index=0, title="Test Chapter", html=html)
        assert chapter.index == 0
        assert chapter.title == "Test Chapter"
        assert "This is a test paragraph" in chapter.text
        assert chapter.word_count > 0
        assert chapter.estimated_duration_sec > 0

    def test_word_count_accuracy(self):
        # "One two three four five" = 5 words per paragraph, 2 paragraphs
        html = "<p>One two three four five.</p><p>Six seven eight nine ten.</p>"
        chapter = Chapter.from_html(index=0, title="Count Test", html=html)
        assert chapter.word_count == 10

    def test_duration_calculation(self):
        # 150 words at 150 WPM = 1 minute = 60 seconds
        words = " ".join(["word"] * 150)
        html = f"<p>{words}.</p>"
        chapter = Chapter.from_html(index=0, title="Duration Test", html=html)
        assert chapter.word_count == 150
        assert abs(chapter.estimated_duration_sec - 60.0) < 0.01

    def test_empty_html_chapter(self):
        html = "<p></p>"
        chapter = Chapter.from_html(index=0, title="Empty", html=html)
        assert chapter.word_count == 0
        assert chapter.estimated_duration_sec == 0.0
        assert chapter.text == ""

    def test_chapter_preserves_original_html(self):
        html = "<h1>Title</h1><p>Content with <strong>bold</strong>.</p>"
        chapter = Chapter.from_html(index=0, title="HTML Test", html=html)
        assert "<strong>bold</strong>" in chapter.html


# ---------------------------------------------------------------------------
# BookMetadata dataclass
# ---------------------------------------------------------------------------


class TestBookMetadata:
    """Tests for the BookMetadata dataclass."""

    def test_creation(self):
        meta = BookMetadata(
            title="Test Book",
            author="Test Author",
            language="a",
            cover_image=None,
            chapter_count=5,
        )
        assert meta.title == "Test Book"
        assert meta.author == "Test Author"
        assert meta.language == "a"
        assert meta.cover_image is None
        assert meta.chapter_count == 5

    def test_with_cover_image(self):
        fake_image = b"\x89PNG\r\n\x1a\n"
        meta = BookMetadata(
            title="Book",
            author="Author",
            language="f",
            cover_image=fake_image,
            chapter_count=10,
        )
        assert meta.cover_image == fake_image
        assert len(meta.cover_image) == 8


# ---------------------------------------------------------------------------
# Integration-style tests (without real EPUB files)
# ---------------------------------------------------------------------------


class TestIntegration:
    """Higher-level tests combining multiple functions."""

    def test_full_chapter_extraction_pipeline(self):
        """Simulate processing a chapter's worth of HTML."""
        html = """
        <html>
        <head><title>A Novel</title></head>
        <body>
            <h1>Chapter One</h1>
            <p>It was a dark and stormy night. The wind howled through
            the empty streets of the old town.</p>
            <p>Nobody dared to venture outside, for the storm had been
            raging for three days without pause</p>
            <p>But Maria knew she had no choice. Her brother was out
            there somewhere, lost in the tempest.</p>
        </body>
        </html>
        """
        chapter = Chapter.from_html(index=0, title="", html=html)

        # Title should be extractable
        title = _extract_title_from_html(html)
        assert title == "Chapter One"

        # Text should include all paragraphs
        assert "dark and stormy night" in chapter.text
        assert "Maria knew" in chapter.text

        # The line "without pause" should get a period appended
        lines = chapter.text.split("\n")
        pause_line = [l for l in lines if "without pause" in l]
        assert len(pause_line) == 1
        assert pause_line[0].strip().endswith(".")

        # Word count should be reasonable
        assert chapter.word_count > 30

        # Duration should be positive
        assert chapter.estimated_duration_sec > 0

    def test_mixed_language_chapter(self):
        """Test that language detection works on extracted text."""
        spanish_html = """
        <html><body>
            <h1>Capitulo Uno</h1>
            <p>El hombre estaba en la casa con los ninos.</p>
            <p>La mujer fue al mercado para comprar una fruta.</p>
            <p>Entre las calles del pueblo, el sol brillaba con fuerza.</p>
        </body></html>
        """
        text = extract_text_from_html(spanish_html)
        lang = detect_language(text)
        assert lang == "e"

    def test_japanese_chapter(self):
        html = """
        <html><body>
            <h1>第一章</h1>
            <p>むかしむかし、あるところにおじいさんとおばあさんがいました。</p>
            <p>おじいさんは山へしばかりに、おばあさんは川へせんたくに行きました。</p>
        </body></html>
        """
        text = extract_text_from_html(html)
        lang = detect_language(text)
        assert lang == "j"
