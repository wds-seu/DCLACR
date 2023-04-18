from typing import List

from . import utils


class BasicTokenizer:
    """Run basic tokenization (puncation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer

        Args:
            do_lower_case: whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes a piece of text."""
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._preprocess_chinese_chars(text)

        tokens = text.strip().split()

        split_tokens = []
        for token in tokens:
            if self.do_lower_case:
                token = utils.stripe_accents(token.lower())
            split_tokens.extend(self._split_on_punc(token))

        return " ".join(split_tokens).strip().split()

    @staticmethod
    def _preprocess_chinese_chars(text: str) -> str:
        """Adds whitespace around any CJK character."""
        output = ""
        for char in text:
            output += f" {char} " if utils.is_chinese_char(char) else char
        return output

    @staticmethod
    def _clean_text(text: str) -> str:
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            ch = ord(char)
            if ch == 0 or ch == 0xfffd or utils.is_control(char):
                continue
            output.append(" " if utils.is_whitespace(char) else char)
        return "".join(output)

    @staticmethod
    def _split_on_punc(text: str) -> List[str]:
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        start_entity = False
        while i < len(chars):
            char = chars[i]
            if char == '@':
                j = i + 1
                while j < len(chars):
                    if chars[j] == ' ':
                        break
                    elif chars[j] == '$':
                        start_entity = True
                        break
                    j += 1
                output.append([char])
                start_new_word = False
            elif char == '$':
                if not start_new_word:
                    start_new_word = True
                    output[-1].append(char)
                    start_entity = False
                else:
                    output.append([])
                    output[-1].append(char)
            elif start_entity:
                output[-1].append(char)
            elif utils.is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(_) for _ in output]
