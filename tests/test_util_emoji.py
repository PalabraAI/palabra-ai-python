from palabra_ai.util.emoji import Emoji


def test_emoji_bool_true():
    """Test Emoji.bool with True value"""
    assert Emoji.bool(True) == "✅"


def test_emoji_bool_false():
    """Test Emoji.bool with False value"""
    assert Emoji.bool(False) == "❌"


def test_emoji_bool_truthy():
    """Test Emoji.bool with truthy values"""
    assert Emoji.bool(1) == "✅"
    assert Emoji.bool("text") == "✅"
    assert Emoji.bool([1, 2]) == "✅"


def test_emoji_bool_falsy():
    """Test Emoji.bool with falsy values"""
    assert Emoji.bool(0) == "❌"
    assert Emoji.bool("") == "❌"
    assert Emoji.bool([]) == "❌"
    assert Emoji.bool(None) == "❌"
