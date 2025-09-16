"""Test the main __init__.py exports"""
import palabra_ai


def test_version():
    """Test version is accessible"""
    assert hasattr(palabra_ai, "__version__")
    assert isinstance(palabra_ai.__version__, str)
    assert palabra_ai.__version__


def test_main_exports():
    """Test main classes are exported"""
    # Client
    assert hasattr(palabra_ai, "PalabraAI")

    # Config
    assert hasattr(palabra_ai, "Config")
    assert hasattr(palabra_ai, "SourceLang")
    assert hasattr(palabra_ai, "TargetLang")

    # Messages
    assert hasattr(palabra_ai, "Message")
    assert hasattr(palabra_ai, "TranscriptionMessage")
    assert hasattr(palabra_ai, "MessageType")

    # Language
    assert hasattr(palabra_ai, "Language")


def test_language_exports():
    """Test all language constants are exported"""
    languages = [
        "AR", "AR_AE", "AR_SA", "AZ", "BG", "CS", "DA", "DE", "EL",
        "EN", "EN_AU", "EN_CA", "EN_GB", "EN_US", "ES", "ES_MX",
        "FI", "FIL", "FR", "FR_CA", "HE", "HI", "HR", "HU", "ID",
        "IT", "JA", "KO", "MS", "NL", "NO", "PL", "PT", "PT_BR",
        "RO", "RU", "SK", "SV", "TA", "TR", "UK", "VI", "ZH"
    ]

    for lang in languages:
        assert hasattr(palabra_ai, lang)


def test_adapter_exports():
    """Test adapter classes are exported"""
    # File adapters
    assert hasattr(palabra_ai, "FileReader")
    assert hasattr(palabra_ai, "FileWriter")

    # Device adapters
    assert hasattr(palabra_ai, "DeviceReader")
    assert hasattr(palabra_ai, "DeviceWriter")
    assert hasattr(palabra_ai, "DeviceManager")

    # Buffer adapters
    assert hasattr(palabra_ai, "BufferReader")
    assert hasattr(palabra_ai, "BufferWriter")
    assert hasattr(palabra_ai, "RunAsPipe")

    # Dummy adapters
    assert hasattr(palabra_ai, "DummyReader")
    assert hasattr(palabra_ai, "DummyWriter")


def test_all_exports():
    """Test __all__ list is complete"""
    assert "__all__" in dir(palabra_ai)

    # Check some key items are in __all__
    all_items = palabra_ai.__all__
    assert "PalabraAI" in all_items
    assert "Config" in all_items
    assert "Language" in all_items
    assert "__version__" in all_items
