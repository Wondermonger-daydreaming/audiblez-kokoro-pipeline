"""
Comprehensive tests for the audiblez-kokoro preprocessing pipeline.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import (
    apply_custom_substitutions,
    expand_abbreviations,
    expand_numbers,
    expand_roman_numerals,
    load_substitutions,
    normalize_punctuation,
    preprocess,
)


# ===================================================================
# Abbreviation expansion
# ===================================================================


class TestExpandAbbreviations:
    def test_dr(self):
        assert expand_abbreviations("Dr. Smith") == "Doctor Smith"

    def test_mr(self):
        assert expand_abbreviations("Mr. Jones") == "Mister Jones"

    def test_mrs(self):
        assert expand_abbreviations("Mrs. Jones") == "Missus Jones"

    def test_ms(self):
        assert expand_abbreviations("Ms. Davis") == "Miz Davis"

    def test_prof(self):
        assert expand_abbreviations("Prof. Oak") == "Professor Oak"

    def test_st(self):
        assert expand_abbreviations("St. Louis") == "Saint Louis"

    def test_vs(self):
        assert expand_abbreviations("cats vs. dogs") == "cats versus dogs"

    def test_etc(self):
        assert expand_abbreviations("apples, oranges, etc.") == "apples, oranges, et cetera"

    def test_eg(self):
        assert expand_abbreviations("e.g. apples") == "for example apples"

    def test_ie(self):
        assert expand_abbreviations("i.e. the best") == "that is the best"

    def test_jr(self):
        assert expand_abbreviations("Martin Luther King, Jr.") == "Martin Luther King, Junior"

    def test_sr(self):
        assert expand_abbreviations("John Smith, Sr.") == "John Smith, Senior"

    def test_ave(self):
        assert expand_abbreviations("5th Ave.") == "5th Avenue"

    def test_blvd(self):
        assert expand_abbreviations("Sunset Blvd.") == "Sunset Boulevard"

    def test_approx(self):
        assert expand_abbreviations("approx. 10") == "approximately 10"

    def test_dept(self):
        assert expand_abbreviations("dept. of defense") == "department of defense"

    def test_govt(self):
        assert expand_abbreviations("govt. officials") == "government officials"

    def test_months(self):
        assert expand_abbreviations("Jan. 1") == "January 1"
        assert expand_abbreviations("Feb. 14") == "February 14"
        assert expand_abbreviations("Mar. 15") == "March 15"
        assert expand_abbreviations("Apr. 1") == "April 1"
        assert expand_abbreviations("Jun. 21") == "June 21"
        assert expand_abbreviations("Jul. 4") == "July 4"
        assert expand_abbreviations("Aug. 15") == "August 15"
        assert expand_abbreviations("Sep. 1") == "September 1"
        assert expand_abbreviations("Sept. 1") == "September 1"
        assert expand_abbreviations("Oct. 31") == "October 31"
        assert expand_abbreviations("Nov. 11") == "November 11"
        assert expand_abbreviations("Dec. 25") == "December 25"

    def test_multiple_abbreviations(self):
        result = expand_abbreviations("Dr. Smith and Prof. Oak met on Jan. 5")
        assert result == "Doctor Smith and Professor Oak met on January 5"

    def test_no_false_positives(self):
        # "Drive" shouldn't be touched
        assert expand_abbreviations("Drive carefully") == "Drive carefully"

    def test_end_of_string(self):
        assert expand_abbreviations("etc.") == "et cetera"

    def test_idempotent(self):
        text = "Dr. Smith met Prof. Oak on Jan. 5"
        once = expand_abbreviations(text)
        twice = expand_abbreviations(once)
        assert once == twice


# ===================================================================
# Number expansion
# ===================================================================


class TestExpandNumbers:
    # --- Plain integers ---
    def test_zero(self):
        assert expand_numbers("0") == "zero"

    def test_one(self):
        assert expand_numbers("1") == "one"

    def test_forty_two(self):
        assert expand_numbers("42") == "forty-two"

    def test_one_hundred(self):
        assert expand_numbers("100") == "one hundred"

    def test_one_thousand(self):
        assert expand_numbers("1000") == "one thousand"

    def test_large_number(self):
        assert expand_numbers("999999") == "nine hundred ninety-nine thousand nine hundred ninety-nine"

    def test_thirteen(self):
        assert expand_numbers("13") == "thirteen"

    def test_five_hundred_twelve(self):
        assert expand_numbers("512") == "five hundred twelve"

    # --- Ordinals ---
    def test_ordinal_1st(self):
        assert expand_numbers("1st") == "first"

    def test_ordinal_2nd(self):
        assert expand_numbers("2nd") == "second"

    def test_ordinal_3rd(self):
        assert expand_numbers("3rd") == "third"

    def test_ordinal_11th(self):
        assert expand_numbers("11th") == "eleventh"

    def test_ordinal_21st(self):
        assert expand_numbers("21st") == "twenty-first"

    def test_ordinal_42nd(self):
        assert expand_numbers("42nd") == "forty-second"

    def test_ordinal_100th(self):
        assert expand_numbers("100th") == "one hundredth"

    # --- Years ---
    def test_year_1984(self):
        assert expand_numbers("in 1984") == "in nineteen eighty-four"

    def test_year_2024(self):
        assert expand_numbers("in 2024") == "in twenty twenty-four"

    def test_year_2000(self):
        assert expand_numbers("in 2000") == "in two thousand"

    def test_year_2001(self):
        assert expand_numbers("in 2001") == "in two thousand one"

    def test_year_1776(self):
        assert expand_numbers("of 1776") == "of seventeen seventy-six"

    def test_year_circa(self):
        assert expand_numbers("circa 1450") == "circa fourteen fifty"

    # --- Currency ---
    def test_currency_simple(self):
        assert expand_numbers("$3.50") == "three dollars and fifty cents"

    def test_currency_whole(self):
        assert expand_numbers("$100") == "one hundred dollars"

    def test_currency_one_dollar(self):
        assert expand_numbers("$1") == "one dollar"

    def test_currency_one_cent(self):
        assert expand_numbers("$0.01") == "zero dollars and one cent"

    def test_currency_zero_cents(self):
        assert expand_numbers("$5.00") == "five dollars"

    # --- Percentages ---
    def test_percentage(self):
        assert expand_numbers("85%") == "eighty-five percent"

    def test_percentage_100(self):
        assert expand_numbers("100%") == "one hundred percent"

    # --- Decimals ---
    def test_decimal_pi(self):
        result = expand_numbers("3.14")
        assert result == "three point one four"

    def test_decimal_with_zero(self):
        result = expand_numbers("2.05")
        assert result == "two point zero five"

    # --- Mixed content ---
    def test_numbers_in_sentence(self):
        result = expand_numbers("She bought 3 apples for $1.50")
        assert "three" in result
        assert "one dollar and fifty cents" in result

    def test_idempotent(self):
        text = "He has 42 cats and $3.50"
        once = expand_numbers(text)
        twice = expand_numbers(once)
        assert once == twice


# ===================================================================
# Roman numeral expansion
# ===================================================================


class TestExpandRomanNumerals:
    def test_chapter(self):
        assert expand_roman_numerals("Chapter XIV") == "Chapter fourteen"

    def test_part(self):
        assert expand_roman_numerals("Part III") == "Part three"

    def test_volume(self):
        assert expand_roman_numerals("Volume II") == "Volume two"

    def test_act(self):
        assert expand_roman_numerals("Act V") == "Act five"

    def test_scene(self):
        assert expand_roman_numerals("Scene IX") == "Scene nine"

    def test_book(self):
        assert expand_roman_numerals("Book XII") == "Book twelve"

    def test_large_roman(self):
        assert expand_roman_numerals("Chapter XXIV") == "Chapter twenty-four"

    def test_standalone_I_not_expanded(self):
        result = expand_roman_numerals("I went to the store")
        assert result == "I went to the store"

    def test_standalone_V_not_expanded(self):
        result = expand_roman_numerals("V is a letter")
        assert result == "V is a letter"

    def test_standalone_X_not_expanded(self):
        result = expand_roman_numerals("X marks the spot")
        assert result == "X marks the spot"

    def test_no_context_no_expansion(self):
        # "XIV" alone is not expanded
        result = expand_roman_numerals("The symbol XIV appeared")
        assert result == "The symbol XIV appeared"

    def test_multiple_contexts(self):
        text = "Chapter I and Part II"
        result = expand_roman_numerals(text)
        assert result == "Chapter one and Part two"

    def test_idempotent(self):
        text = "Chapter XIV"
        once = expand_roman_numerals(text)
        twice = expand_roman_numerals(once)
        assert once == twice


# ===================================================================
# Punctuation normalization
# ===================================================================


class TestNormalizePunctuation:
    def test_double_hyphen_to_em_dash(self):
        result = normalize_punctuation("well--you know")
        assert result == "well\u2014you know"

    def test_triple_dots_to_ellipsis(self):
        result = normalize_punctuation("wait...")
        assert result == "wait\u2026"

    def test_curly_single_quotes(self):
        result = normalize_punctuation("\u2018hello\u2019")
        assert result == "'hello'"

    def test_curly_double_quotes(self):
        result = normalize_punctuation("\u201Chello\u201D")
        assert result == '"hello"'

    def test_excessive_spaces(self):
        result = normalize_punctuation("too   many    spaces")
        assert result == "too many spaces"

    def test_tabs_normalized(self):
        result = normalize_punctuation("tab\there")
        assert result == "tab here"

    def test_multiple_blank_lines(self):
        result = normalize_punctuation("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_leading_trailing_whitespace(self):
        result = normalize_punctuation("  hello  \n  world  ")
        assert result == "hello\nworld"

    def test_idempotent(self):
        text = "well--you know...  it's   complicated"
        once = normalize_punctuation(text)
        twice = normalize_punctuation(once)
        assert once == twice


# ===================================================================
# Custom substitutions
# ===================================================================


class TestLoadSubstitutions:
    def test_load_valid_json(self, tmp_path):
        f = tmp_path / "subs.json"
        f.write_text('{"foo": "bar"}', encoding="utf-8")
        result = load_substitutions(f)
        assert result == {"foo": "bar"}

    def test_load_missing_file(self, tmp_path):
        f = tmp_path / "nonexistent.json"
        result = load_substitutions(f)
        assert result == {}

    def test_load_invalid_json_type(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected JSON object"):
            load_substitutions(f)

    def test_load_real_substitutions_file(self):
        config_path = Path(__file__).resolve().parent.parent / "config" / "substitutions.json"
        if config_path.exists():
            result = load_substitutions(config_path)
            assert "Tolkien" in result
            assert result["Tolkien"] == "TOLL-keen"


class TestApplyCustomSubstitutions:
    def test_basic_substitution(self):
        subs = {"GIF": "jif"}
        result = apply_custom_substitutions("A GIF is fun", subs)
        assert result == "A jif is fun"

    def test_case_sensitive(self):
        subs = {"GIF": "jif"}
        result = apply_custom_substitutions("A gif is fun", subs)
        assert result == "A gif is fun"  # lowercase 'gif' not matched

    def test_whole_word_only(self):
        subs = {"cat": "dog"}
        result = apply_custom_substitutions("a cat and a caterpillar", subs)
        assert result == "a dog and a caterpillar"

    def test_multiple_substitutions(self):
        subs = {"Tolkien": "TOLL-keen", "Cthulhu": "kuh-THOO-loo"}
        result = apply_custom_substitutions("Tolkien wrote about Cthulhu", subs)
        assert result == "TOLL-keen wrote about kuh-THOO-loo"

    def test_empty_subs(self):
        result = apply_custom_substitutions("hello world", {})
        assert result == "hello world"

    def test_idempotent(self):
        subs = {"GIF": "jif"}
        text = "A GIF is fun"
        once = apply_custom_substitutions(text, subs)
        twice = apply_custom_substitutions(once, subs)
        assert once == twice


# ===================================================================
# Full pipeline
# ===================================================================


class TestPreprocess:
    def test_full_pipeline(self, tmp_path):
        subs_file = tmp_path / "subs.json"
        subs_file.write_text('{"Tolkien": "TOLL-keen"}', encoding="utf-8")

        text = "Dr. Smith read Chapter XIV of Tolkien's 1st book in 1954..."
        result = preprocess(text, substitutions_path=subs_file)

        assert "Doctor" in result
        assert "fourteen" in result
        assert "TOLL-keen" in result
        assert "first" in result
        assert "\u2026" in result  # ellipsis
        assert "Dr." not in result
        assert "XIV" not in result
        assert "1st" not in result

    def test_pipeline_without_substitutions(self):
        text = "Mr. Jones scored 85% on the 3rd test"
        result = preprocess(text)

        assert "Mister" in result
        assert "eighty-five percent" in result
        assert "third" in result
        assert "Mr." not in result
        assert "85%" not in result

    def test_pipeline_preserves_regular_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        result = preprocess(text)
        assert result == text

    def test_pipeline_complex(self):
        text = "In 1984, Prof. Oak spent $3.50 on Volume II... approx. 100% worth it"
        result = preprocess(text)

        assert "nineteen eighty-four" in result
        assert "Professor" in result
        assert "three dollars and fifty cents" in result
        assert "Volume two" in result
        assert "\u2026" in result
        assert "approximately" in result
        assert "one hundred percent" in result

    def test_idempotency_full_pipeline(self, tmp_path):
        subs_file = tmp_path / "subs.json"
        subs_file.write_text('{"Tolkien": "TOLL-keen"}', encoding="utf-8")

        text = "Dr. Smith read Chapter XIV of Tolkien's 1st book in 1954..."
        once = preprocess(text, substitutions_path=subs_file)
        twice = preprocess(once, substitutions_path=subs_file)
        assert once == twice

    def test_idempotency_numbers_only(self):
        text = "42 cats, 1st place, $3.50, 85%, 3.14"
        once = expand_numbers(text)
        twice = expand_numbers(once)
        assert once == twice

    def test_idempotency_abbreviations_only(self):
        text = "Dr. Smith, Mr. Jones, Prof. Oak, etc."
        once = expand_abbreviations(text)
        twice = expand_abbreviations(once)
        assert once == twice

    def test_idempotency_roman_numerals_only(self):
        text = "Chapter XIV, Part III, Volume II"
        once = expand_roman_numerals(text)
        twice = expand_roman_numerals(once)
        assert once == twice

    def test_idempotency_pipeline_no_subs(self):
        text = "In 1984, Dr. Smith's 1st visit to Chapter XIV cost $3.50..."
        once = preprocess(text)
        twice = preprocess(once)
        assert once == twice

    def test_empty_string(self):
        assert preprocess("") == ""

    def test_nonexistent_substitutions_path(self, tmp_path):
        text = "Hello world"
        result = preprocess(text, substitutions_path=tmp_path / "nope.json")
        assert result == "Hello world"
