from palabra_ai.util.emoji import Emoji


class TestEmoji:
    def test_bool_values(self):
        assert Emoji.bool(True) == "✅"
        assert Emoji.bool(False) == "❌"
