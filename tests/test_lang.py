import pytest
from palabra_ai.lang import Language, LanguageRegistry, ES, EN, FR, DE, JA, ZH
from palabra_ai.exc import ConfigurationError


def test_language_creation():
    """Test Language creation and basic properties"""
    # Test existing predefined language
    lang = Language.get_or_create("es")
    assert lang.code == "es"
    assert lang.bcp47 == "es"
    assert lang.flag == "🇪🇸"
    assert lang == ES


def test_language_get_or_create_existing():
    """Test get_or_create with existing language"""
    lang1 = Language.get_or_create("en")
    lang2 = Language.get_or_create("en")
    assert lang1 == lang2  # Equal by code


def test_language_get_or_create_new():
    """Test get_or_create with new language"""
    lang = Language.get_or_create("xyz")
    assert lang.code == "xyz"
    assert lang.bcp47 == "xyz"
    assert lang.flag == "🌐❓"  # Default flag for unknown languages


def test_language_get_by_bcp47():
    """Test get_by_bcp47 method"""
    lang = Language.get_by_bcp47("es")
    assert lang.code == "es"
    assert lang.bcp47 == "es"


def test_language_get_by_bcp47_not_found():
    """Test get_by_bcp47 with unknown language"""
    with pytest.raises(ConfigurationError) as exc_info:
        Language.get_by_bcp47("xyz-XY")
    assert "Language with BCP47 code" in str(exc_info.value)
    assert "not found" in str(exc_info.value)


def test_language_equality():
    """Test Language equality"""
    lang1 = Language.get_or_create("es")
    lang2 = Language.get_or_create("es")
    lang3 = Language.get_or_create("en")

    assert lang1 == lang2
    assert lang1 != lang3
    assert lang1 == "es"  # Can compare with string if language exists

    # Test error when comparing with unknown string
    with pytest.raises(TypeError) as exc_info:
        lang1 == "unknown_lang"
    assert "Cannot compare Language with unknown language code" in str(exc_info.value)

    # Test error when comparing with non-string
    with pytest.raises(TypeError) as exc_info:
        lang1 == 123
    assert "Cannot compare Language with int" in str(exc_info.value)


def test_language_repr():
    """Test Language repr"""
    lang = Language.get_or_create("es")
    assert repr(lang) == "🇪🇸es"


def test_language_str():
    """Test Language str"""
    lang = Language.get_or_create("es")
    assert str(lang) == "es"


def test_language_hash():
    """Test Language hash"""
    lang1 = Language.get_or_create("es")
    lang2 = Language.get_or_create("es")
    assert hash(lang1) == hash(lang2)


def test_predefined_languages():
    """Test predefined languages"""
    # Test a few predefined languages
    languages = {
        "en": "🇬🇧",
        "es": "🇪🇸",
        "fr": "🇫🇷",
        "de": "🇩🇪",
        "ja": "🇯🇵",
        "zh": "🇨🇳",
    }

    for code, flag in languages.items():
        lang = Language.get_or_create(code)
        assert lang.bcp47 == code
        assert lang.flag == flag


def test_language_code_normalization():
    """Test language code normalization"""
    # Should handle uppercase
    lang = Language.get_or_create("ES")
    assert lang.code == "es"

    # Should handle mixed case
    lang = Language.get_or_create("Es")
    assert lang.code == "es"


def test_language_registry():
    """Test LanguageRegistry functionality"""
    registry = LanguageRegistry()

    # Create and register a language
    lang = Language("test", registry=registry, flag="🏴")
    assert lang.code == "test"
    assert registry.by_code["test"] == lang
    assert lang in registry.all_languages

    # Get by BCP47
    found = registry.get_by_bcp47("test")
    assert found == lang

    # Get or create existing
    existing = registry.get_or_create("test")
    assert existing == lang

    # Get or create new
    new_lang = registry.get_or_create("new")
    assert new_lang.code == "new"
    assert new_lang.flag == "🌐❓"


def test_valid_source_language():
    """Test source language validation"""
    from palabra_ai.lang import (
        is_valid_source_language,
        AR, BA, AZ, FIL, TH
    )

    # Valid source languages
    assert is_valid_source_language(AR) is True  # Arabic can be source
    assert is_valid_source_language(EN) is True  # English can be source
    assert is_valid_source_language(BA) is True  # Bashkir can be source
    assert is_valid_source_language(TH) is True  # Thai can be source

    # Invalid source languages
    assert is_valid_source_language(AZ) is False  # Azerbaijani cannot be source
    assert is_valid_source_language(FIL) is False  # Filipino cannot be source


def test_valid_target_language():
    """Test target language validation"""
    from palabra_ai.lang import (
        is_valid_target_language,
        ES, EN_US, ZH_HANS, BA, TH, AZ
    )

    # Valid target languages
    assert is_valid_target_language(ES) is True  # Spanish can be target
    assert is_valid_target_language(EN_US) is True  # English US can be target
    assert is_valid_target_language(ZH_HANS) is True  # Chinese Simplified can be target
    assert is_valid_target_language(AZ) is True  # Azerbaijani can be target

    # Invalid target languages
    assert is_valid_target_language(BA) is False  # Bashkir cannot be target
    assert is_valid_target_language(TH) is False  # Thai cannot be target


def test_auto_detectable_language():
    """Test auto-detectable language validation"""
    from palabra_ai.lang import (
        is_auto_detectable_language,
        EN, ES, AR, BA, AZ
    )

    # Auto-detectable languages
    assert is_auto_detectable_language(EN) is True
    assert is_auto_detectable_language(ES) is True
    assert is_auto_detectable_language(AR) is True

    # Non auto-detectable languages
    assert is_auto_detectable_language(BA) is False  # Bashkir not in auto-detect
    assert is_auto_detectable_language(AZ) is False  # Azerbaijani not in auto-detect
