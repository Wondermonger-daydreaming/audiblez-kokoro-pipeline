"""
Unit tests for the multi_voice module.

Tests cover config loading, validation, voice resolution (exact match,
range match, default fallback), lang_code derivation, and the
VoiceAssignment dataclass.  No Kokoro installation required.
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure the src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multi_voice import (
    VoiceAssignment,
    VoiceManager,
    load_voice_config,
    resolve_voice,
    _derive_lang_code,
    _is_valid_chapter_key,
    _parse_range,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def valid_config_path(tmp_path: Path) -> Path:
    """Write a valid voice_config.json and return its path."""
    config = {
        "default_voice": "af_sky",
        "default_speed": 1.0,
        "chapters": {
            "default": {"voice": "af_sky", "speed": 1.0},
            "0": {"voice": "am_adam", "speed": 0.9},
            "1-3": {"voice": "af_heart", "speed": 1.0},
            "7": {"voice": "bf_emma", "speed": 1.1},
        },
    }
    p = tmp_path / "voice_config.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    return p


@pytest.fixture()
def valid_config(valid_config_path: Path) -> dict:
    """Return a loaded valid config dict."""
    return load_voice_config(valid_config_path)


@pytest.fixture()
def overlapping_config_path(tmp_path: Path) -> Path:
    """Config with overlapping ranges to test first-match behaviour."""
    config = {
        "default_voice": "af_sky",
        "default_speed": 1.0,
        "chapters": {
            "default": {"voice": "af_sky", "speed": 1.0},
            "2-5": {"voice": "am_echo", "speed": 0.8},
            "3-7": {"voice": "bf_alice", "speed": 1.2},
        },
    }
    p = tmp_path / "overlap.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    return p


# ===================================================================
# load_voice_config
# ===================================================================


class TestLoadVoiceConfig:
    """Tests for config loading and validation."""

    def test_load_valid_config(self, valid_config_path: Path):
        config = load_voice_config(valid_config_path)
        assert config["default_voice"] == "af_sky"
        assert config["default_speed"] == 1.0
        assert "chapters" in config
        assert "0" in config["chapters"]
        assert "1-3" in config["chapters"]

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Voice config not found"):
            load_voice_config(tmp_path / "nope.json")

    def test_missing_default_voice_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text('{"chapters": {}}', encoding="utf-8")
        with pytest.raises(ValueError, match="missing required key 'default_voice'"):
            load_voice_config(p)

    def test_invalid_json_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_voice_config(p)

    def test_non_object_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_voice_config(p)

    def test_invalid_chapter_key_raises(self, tmp_path: Path):
        config = {
            "default_voice": "af_sky",
            "chapters": {"abc": {"voice": "af_sky"}},
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid chapter key 'abc'"):
            load_voice_config(p)

    def test_reversed_range_raises(self, tmp_path: Path):
        config = {
            "default_voice": "af_sky",
            "chapters": {"5-2": {"voice": "af_sky"}},
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid chapter key '5-2'"):
            load_voice_config(p)

    def test_chapters_not_object_raises(self, tmp_path: Path):
        config = {
            "default_voice": "af_sky",
            "chapters": [1, 2, 3],
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(ValueError, match="'chapters' must be a JSON object"):
            load_voice_config(p)

    def test_config_without_chapters_key(self, tmp_path: Path):
        config = {"default_voice": "af_sky", "default_speed": 0.9}
        p = tmp_path / "minimal.json"
        p.write_text(json.dumps(config), encoding="utf-8")
        result = load_voice_config(p)
        assert result["default_voice"] == "af_sky"

    def test_real_config_file(self):
        """Validate the actual shipped config file."""
        config_path = (
            Path(__file__).resolve().parent.parent / "config" / "voice_config.json"
        )
        if config_path.exists():
            config = load_voice_config(config_path)
            assert config["default_voice"] == "af_sky"
            assert "chapters" in config


# ===================================================================
# resolve_voice — exact match
# ===================================================================


class TestResolveVoiceExact:
    """Test exact chapter index matching."""

    def test_exact_match_chapter_0(self, valid_config: dict):
        result = resolve_voice(0, valid_config)
        assert result.voice == "am_adam"
        assert result.speed == 0.9
        assert result.chapter_index == 0

    def test_exact_match_chapter_7(self, valid_config: dict):
        result = resolve_voice(7, valid_config)
        assert result.voice == "bf_emma"
        assert result.speed == 1.1


# ===================================================================
# resolve_voice — range match
# ===================================================================


class TestResolveVoiceRange:
    """Test range-based chapter matching."""

    def test_range_start(self, valid_config: dict):
        result = resolve_voice(1, valid_config)
        assert result.voice == "af_heart"
        assert result.speed == 1.0

    def test_range_middle(self, valid_config: dict):
        result = resolve_voice(2, valid_config)
        assert result.voice == "af_heart"

    def test_range_end(self, valid_config: dict):
        result = resolve_voice(3, valid_config)
        assert result.voice == "af_heart"

    def test_outside_range_falls_to_default(self, valid_config: dict):
        # Chapter 4 is not covered by exact match or range 1-3
        result = resolve_voice(4, valid_config)
        assert result.voice == "af_sky"  # from "default" entry


# ===================================================================
# resolve_voice — default fallback
# ===================================================================


class TestResolveVoiceDefault:
    """Test fallback to default entries."""

    def test_falls_to_chapters_default(self, valid_config: dict):
        result = resolve_voice(99, valid_config)
        assert result.voice == "af_sky"
        assert result.speed == 1.0

    def test_falls_to_top_level_when_no_chapters_default(self, tmp_path: Path):
        config = {
            "default_voice": "am_puck",
            "default_speed": 0.85,
            "chapters": {
                "0": {"voice": "bf_alice", "speed": 1.0},
            },
        }
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(config), encoding="utf-8")
        loaded = load_voice_config(p)

        result = resolve_voice(5, loaded)
        assert result.voice == "am_puck"
        assert result.speed == 0.85

    def test_falls_to_top_level_when_no_chapters_section(self, tmp_path: Path):
        config = {
            "default_voice": "am_echo",
            "default_speed": 1.2,
        }
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(config), encoding="utf-8")
        loaded = load_voice_config(p)

        result = resolve_voice(0, loaded)
        assert result.voice == "am_echo"
        assert result.speed == 1.2


# ===================================================================
# Lang-code derivation
# ===================================================================


class TestDeriveLangCode:
    """Test lang_code derivation from voice name prefix."""

    def test_american_female(self):
        assert _derive_lang_code("af_sky") == "a"

    def test_american_male(self):
        assert _derive_lang_code("am_adam") == "a"

    def test_british_female(self):
        assert _derive_lang_code("bf_emma") == "b"

    def test_british_male(self):
        assert _derive_lang_code("bm_george") == "b"

    def test_empty_voice_defaults_to_american(self):
        assert _derive_lang_code("") == "a"

    def test_unknown_prefix_defaults_to_american(self):
        assert _derive_lang_code("xf_unknown") == "a"

    def test_resolve_voice_sets_lang_code(self, valid_config: dict):
        # Chapter 7 uses bf_emma -> lang_code 'b'
        result = resolve_voice(7, valid_config)
        assert result.lang_code == "b"

    def test_resolve_voice_american_lang_code(self, valid_config: dict):
        # Chapter 0 uses am_adam -> lang_code 'a'
        result = resolve_voice(0, valid_config)
        assert result.lang_code == "a"


# ===================================================================
# VoiceAssignment dataclass
# ===================================================================


class TestVoiceAssignment:
    """Test the VoiceAssignment dataclass."""

    def test_creation(self):
        va = VoiceAssignment(
            chapter_index=3,
            voice="af_heart",
            speed=1.0,
            lang_code="a",
        )
        assert va.chapter_index == 3
        assert va.voice == "af_heart"
        assert va.speed == 1.0
        assert va.lang_code == "a"

    def test_equality(self):
        va1 = VoiceAssignment(0, "af_sky", 1.0, "a")
        va2 = VoiceAssignment(0, "af_sky", 1.0, "a")
        assert va1 == va2

    def test_inequality(self):
        va1 = VoiceAssignment(0, "af_sky", 1.0, "a")
        va2 = VoiceAssignment(0, "am_adam", 0.9, "a")
        assert va1 != va2


# ===================================================================
# Overlapping ranges
# ===================================================================


class TestOverlappingRanges:
    """Test that overlapping ranges resolve to the first match."""

    def test_chapter_in_first_range_only(self, overlapping_config_path: Path):
        config = load_voice_config(overlapping_config_path)
        result = resolve_voice(2, config)
        assert result.voice == "am_echo"  # "2-5" is first

    def test_chapter_in_both_ranges(self, overlapping_config_path: Path):
        config = load_voice_config(overlapping_config_path)
        # Chapter 4 matches both "2-5" and "3-7"; first match wins
        result = resolve_voice(4, config)
        assert result.voice == "am_echo"

    def test_chapter_in_second_range_only(self, overlapping_config_path: Path):
        config = load_voice_config(overlapping_config_path)
        # Chapter 6 only matches "3-7"
        result = resolve_voice(6, config)
        assert result.voice == "bf_alice"


# ===================================================================
# _is_valid_chapter_key helper
# ===================================================================


class TestIsValidChapterKey:
    """Test the chapter key validator."""

    def test_digit(self):
        assert _is_valid_chapter_key("0") is True
        assert _is_valid_chapter_key("12") is True

    def test_valid_range(self):
        assert _is_valid_chapter_key("1-3") is True
        assert _is_valid_chapter_key("10-20") is True

    def test_single_chapter_range(self):
        assert _is_valid_chapter_key("5-5") is True

    def test_reversed_range(self):
        assert _is_valid_chapter_key("5-2") is False

    def test_alpha(self):
        assert _is_valid_chapter_key("abc") is False

    def test_negative(self):
        # "-1" is not digit-only, will fail
        assert _is_valid_chapter_key("-1") is False

    def test_empty(self):
        assert _is_valid_chapter_key("") is False


# ===================================================================
# _parse_range helper
# ===================================================================


class TestParseRange:
    """Test range parsing."""

    def test_valid_range(self):
        assert _parse_range("1-3") == (1, 3)

    def test_single_element_range(self):
        assert _parse_range("5-5") == (5, 5)

    def test_not_a_range_single_digit(self):
        assert _parse_range("5") is None

    def test_reversed_range(self):
        assert _parse_range("5-2") is None

    def test_non_numeric(self):
        assert _parse_range("a-b") is None

    def test_empty(self):
        assert _parse_range("") is None

    def test_multiple_dashes(self):
        # "1-2-3" splits as ("1", "2-3"); int("2-3") fails
        assert _parse_range("1-2-3") is None


# ===================================================================
# VoiceManager (without Kokoro)
# ===================================================================


class TestVoiceManager:
    """Test VoiceManager config loading and resolution (no Kokoro needed)."""

    def test_init_loads_config(self, valid_config_path: Path):
        vm = VoiceManager(valid_config_path, device="cpu")
        assert vm.config["default_voice"] == "af_sky"

    def test_init_missing_config_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            VoiceManager(tmp_path / "nope.json")

    def test_pipeline_creation_raises_without_kokoro(self, valid_config_path: Path):
        """Calling get_pipeline_and_voice should fail gracefully when
        kokoro is not installed — it attempts to import KPipeline."""
        vm = VoiceManager(valid_config_path, device="cpu")
        with pytest.raises((ImportError, ModuleNotFoundError)):
            vm.get_pipeline_and_voice(0)
