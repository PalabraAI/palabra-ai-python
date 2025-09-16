#!/usr/bin/env python3
"""
Pre-commit hook to check for non-Latin characters in source files.
Allows emoji but prevents Cyrillic and other non-Latin alphabets.
"""

import argparse
import re
import sys
from pathlib import Path


def has_non_latin_text(
    content: str, allow_emoji: bool = True
) -> tuple[bool, list[tuple[int, str]]]:
    """
    Check if content contains non-Latin alphabetic characters.

    Returns:
        (has_violations, violations_list) where violations_list contains (line_num, line_content)
    """
    violations = []

    for line_num, line in enumerate(content.splitlines(), 1):
        # Skip empty lines and lines with only whitespace
        if not line.strip():
            continue

        # Allow emoji by removing them from check
        if allow_emoji:
            # Remove emoji and other symbols, keep only alphabetic characters
            line_to_check = re.sub(r"[^\w\s]", "", line)
        else:
            line_to_check = line

        # Look for non-ASCII alphabetic characters (Cyrillic, etc.)
        # This pattern matches any non-ASCII letter
        if re.search(r"[^\x00-\x7F]", line_to_check):
            # Further check: is it actually alphabetic (not just special chars)?
            non_ascii_chars = re.findall(r"[^\x00-\x7F]", line_to_check)
            if any(char.isalpha() for char in non_ascii_chars):
                violations.append((line_num, line.strip()))

    return len(violations) > 0, violations


def check_file(file_path: Path, allow_emoji: bool = True) -> bool:
    """Check a single file for non-Latin characters."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Skip binary files
        return True
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    has_violations, violations = has_non_latin_text(content, allow_emoji)

    if has_violations:
        print(f"\n❌ {file_path}: Found non-Latin characters:")
        for line_num, line_content in violations:
            print(f"  Line {line_num}: {line_content}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check for non-Latin characters in source files"
    )
    parser.add_argument("files", nargs="*", help="Files to check")
    parser.add_argument(
        "--no-emoji", action="store_true", help="Disallow emoji as well"
    )
    args = parser.parse_args()

    if not args.files:
        print("No files to check")
        return 0

    allow_emoji = not args.no_emoji
    success = True

    for file_path in args.files:
        path = Path(file_path)

        # Only check text files (Python, JSON, YAML, etc.)
        if path.suffix not in [
            ".py",
            ".json",
            ".yml",
            ".yaml",
            ".toml",
            ".md",
            ".txt",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
        ]:
            continue

        # Skip special files that are allowed to have non-Latin content
        if "emoji.py" in str(path) or "test_emoji" in str(path):
            continue

        if not check_file(path, allow_emoji):
            success = False

    if not success:
        print("\n⚠️  Files contain non-Latin characters. Please translate to English.")
        print(
            "Info: Emoji are allowed, but Cyrillic and other non-Latin alphabets are not."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
