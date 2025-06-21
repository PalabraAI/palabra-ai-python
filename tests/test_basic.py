"""Basic tests to ensure the package works."""


def test_import():
    """Test that the package can be imported."""
    import palabra_ai
    assert palabra_ai is not None


def test_version():
    """Test that version is defined."""
    from palabra_ai import __version__
    assert __version__ is not None
    assert isinstance(__version__, str)
