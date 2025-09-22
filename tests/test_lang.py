import pytest
from palabra_ai.lang import (
    Language, LanguageRegistry,
    ES, EN, FR, DE, JA, ZH, EN_US, EN_GB, ES_MX, ZH_HANS, ZH_HANT,
    AR, BA, AZ, FIL, TH, EO, FA, GA, IA, MN, MR, MT, UG,
    BS, IS, KK, MK, SR
)
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


def test_language_source_code_target_code():
    """Test Language source_code and target_code fields"""
    # Regular language - both codes equal to language code
    assert ES.source_code == "es"
    assert ES.target_code == "es"

    # Source-only language - target_code is None
    assert BA.source_code == "ba"
    assert BA.target_code is None
    assert TH.source_code == "th"
    assert TH.target_code is None

    # Target-only language - source_code is None
    assert AZ.source_code is None
    assert AZ.target_code == "az"
    assert FIL.source_code is None
    assert FIL.target_code == "fil"

    # Smart mapping for EN (base to variant)
    assert EN.source_code == "en"
    assert EN.target_code == "en-us"

    # Smart mapping for ZH (base to variant)
    assert ZH.source_code == "zh"
    assert ZH.target_code == "zh-hans"


def test_language_variants_source_target_mapping():
    """Test language variants have correct source/target mapping"""
    # English variants map to base "en" for source
    assert EN_US.source_code == "en"
    assert EN_US.target_code == "en-us"
    assert EN_GB.source_code == "en"
    assert EN_GB.target_code == "en-gb"

    # Spanish variant maps to base "es" for source
    assert ES_MX.source_code == "es"
    assert ES_MX.target_code == "es-mx"

    # Chinese variants map to base "zh" for source
    assert ZH_HANS.source_code == "zh"
    assert ZH_HANS.target_code == "zh-hans"
    assert ZH_HANT.source_code == "zh"
    assert ZH_HANT.target_code == "zh-hant"


def test_dynamic_language_creation_with_codes():
    """Test creating dynamic language with explicit source/target codes"""
    # Create dynamic language with explicit codes
    lang = Language("custom", source_code="cst", target_code="custom-v2")
    assert lang.code == "custom"
    assert lang.source_code == "cst"
    assert lang.target_code == "custom-v2"

    # Create dynamic language without explicit codes (defaults to code)
    lang2 = Language("another")
    assert lang2.code == "another"
    assert lang2.source_code == "another"
    assert lang2.target_code == "another"


def test_all_source_only_languages():
    """Test all source-only languages have correct configuration"""
    source_only = [BA, EO, FA, GA, IA, MN, MR, MT, TH, UG]
    for lang in source_only:
        assert lang.source_code is not None, f"{lang.code} should have source_code"
        assert lang.target_code is None, f"{lang.code} should have target_code=None"


def test_all_target_only_languages():
    """Test all target-only languages have correct configuration"""
    target_only = [AZ, BS, FIL, IS, KK, MK, SR]
    for lang in target_only:
        assert lang.source_code is None, f"{lang.code} should have source_code=None"
        assert lang.target_code is not None, f"{lang.code} should have target_code"


def test_backwards_compatibility_string_lang():
    """Test backwards compatibility - using strings for lang parameter"""
    from palabra_ai.config import SourceLang, TargetLang

    # Should work with string
    source = SourceLang(lang="es")
    assert source.lang == ES
    assert source.lang.code == "es"

    target = TargetLang(lang="en")
    assert target.lang == EN
    assert target.lang.code == "en"

    # Should validate source/target restrictions
    with pytest.raises(ConfigurationError) as exc_info:
        SourceLang(lang=AZ)  # Azerbaijani is target-only
    assert "not supported as a source language" in str(exc_info.value)

    with pytest.raises(ConfigurationError) as exc_info:
        TargetLang(lang=BA)  # Bashkir is source-only
    assert "not supported as a target language" in str(exc_info.value)


def test_api_compliance_source_languages():
    """Test that all source language codes match API documentation"""
    # From API documentation - all supported source languages
    api_source_codes = {
        "ar", "ba", "eu", "be", "bn", "bg", "ca", "zh", "hr", "cs",
        "da", "nl", "en", "eo", "et", "fi", "fr", "gl", "de", "el",
        "he", "hi", "hu", "id", "ia", "ga", "it", "ja", "ko", "lv",
        "lt", "ms", "mt", "mr", "mn", "no", "fa", "pl", "pt", "ro",
        "ru", "sk", "sl", "es", "sw", "sv", "ta", "th", "tr", "uk",
        "ur", "ug", "vi", "cy"
    }

    # Get all source codes from our language definitions
    from palabra_ai.lang import VALID_SOURCE_LANGUAGES
    our_source_codes = set()
    for lang in VALID_SOURCE_LANGUAGES:
        if lang.source_code is not None:
            our_source_codes.add(lang.source_code)

    # Check that our source codes match API exactly
    assert our_source_codes == api_source_codes, f"Mismatch: ours={our_source_codes}, api={api_source_codes}"


def test_api_compliance_target_languages():
    """Test that all target language codes match API documentation"""
    # From API documentation - all supported target languages
    api_target_codes = {
        "ar", "az", "be", "bs", "bg", "ca", "zh-hans", "zh-hant",
        "hr", "cs", "da", "nl", "en-au", "en-ca", "en-gb", "en-us",
        "et", "fil", "fi", "fr", "fr-ca", "gl", "de", "el", "he",
        "hi", "hu", "is", "id", "it", "ja", "kk", "ko", "lv", "lt",
        "mk", "ms", "no", "pl", "pt", "pt-br", "ro", "ru", "sr",
        "sk", "sl", "es", "es-mx", "sw", "sv", "ta", "tr", "uk",
        "ur", "vi", "cy"
    }

    # Get all target codes from our language definitions
    from palabra_ai.lang import VALID_TARGET_LANGUAGES
    our_target_codes = set()
    for lang in VALID_TARGET_LANGUAGES:
        if lang.target_code is not None:
            our_target_codes.add(lang.target_code)

    # Check that our target codes match API exactly
    assert our_target_codes == api_target_codes, f"Mismatch: ours={our_target_codes}, api={api_target_codes}"


def test_language_variant_mappings():
    """Test that language variants map correctly for API"""
    # Test English variants all map to "en" for source
    assert EN.source_code == "en"
    assert EN_US.source_code == "en"
    assert EN_GB.source_code == "en"

    # Test Chinese variants all map to "zh" for source
    assert ZH.source_code == "zh"
    assert ZH_HANS.source_code == "zh"
    assert ZH_HANT.source_code == "zh"

    # Test that variants have specific codes for target
    assert EN_US.target_code == "en-us"
    assert EN_GB.target_code == "en-gb"
    assert ZH_HANS.target_code == "zh-hans"
    assert ZH_HANT.target_code == "zh-hant"


def test_no_undefined_languages_in_valid_sets():
    """Test that VALID_SOURCE/TARGET_LANGUAGES don't contain undefined languages"""
    from palabra_ai.lang import VALID_SOURCE_LANGUAGES, VALID_TARGET_LANGUAGES

    # Every language in VALID sets should have proper source/target codes
    for lang in VALID_SOURCE_LANGUAGES:
        assert lang.source_code is not None, f"{lang.code} in VALID_SOURCE but has no source_code"

    for lang in VALID_TARGET_LANGUAGES:
        assert lang.target_code is not None, f"{lang.code} in VALID_TARGET but has no target_code"