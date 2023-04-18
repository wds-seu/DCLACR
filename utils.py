import unicodedata


def is_control(ch: str):
    """Checks whether `ch` is a control character."""
    # These are technically control characters,
    # but we count them as whitespace characters.
    if ch in "\t\n\r":
        return False
    return unicodedata.category(ch).startswith("C")


def is_whitespace(ch: str):
    """Checks whether `ch` is a whitespace character."""
    # \t, \n, and \r are technically control characters,
    # but we treat them as whitespace
    # since they are generally considered as such.
    if ch in " \t\n\r":
        return True
    return unicodedata.category(ch) == "Zs"


def is_punctuation(char: str):
    """Checks whether `char` is a punctuation character."""
    code = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class, but for consistency, we treat them as
    # punctuation too.
    if (
        (33 <= code <= 47)
        or (58 <= code <= 64)
        or (91 <= code <= 96)
        or (123 <= code <= 126)
    ):
        return True
    return unicodedata.category(char).startswith("P")


def is_chinese_char(char: str):
    """Checks whether `char` is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    code = ord(char)
    return (
        (0x4E00 <= code <= 0x9FFF)
        or (0x3400 <= code <= 0x4DBF)
        or (0x20000 <= code <= 0x2A6DF)
        or (0x2A700 <= code <= 0x2B73F)
        or (0x2B740 <= code <= 0x2B81F)
        or (0x2B820 <= code <= 0x2CEAF)
        or (0xF900 <= code <= 0xFAFF)
        or (0x2F800 <= code <= 0x2FA1F)
    )


def stripe_accents(text: str) -> str:
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = ""
    for char in text:
        if unicodedata.category(char) == "Mn":
            continue
        output += char
    return output
