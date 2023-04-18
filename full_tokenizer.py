from collections import OrderedDict

from .basic_tokenizer import BasicTokenizer
from .wordpiece_tokenizer import WordpieceTokenizer


class FullTokenizer:
    """Runs end-to-end tokenization."""

    def __init__(self, vocab_file, entity_type_list=None, do_lower_case=True):
        if entity_type_list is not None and do_lower_case:
            entity_type_list = [_.lower() for _ in entity_type_list]
        self.vocab = self._load_vocab(vocab_file, entity_type_list)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        split_tokens = []
        basic_tokens = self.basic_tokenizer.tokenize(text)
        for i, token in enumerate(basic_tokens):
            output_tokens = self.wordpiece_tokenizer.tokenize(token)
            for sub_token in output_tokens:
                split_tokens.append(sub_token)

        untokenized_indices_of_split_tokens = []
        orig_tokens = []
        for token in text.split(' '):
            orig_tokens.append(''.join(self.basic_tokenizer.tokenize(token)))
        current_token = ''
        idx = 0
        for i, split_token in enumerate(split_tokens):
            if split_token.startswith('##'):
                current_token += split_token[2:]
            else:
                current_token += split_token
            untokenized_indices_of_split_tokens.append(idx)
            if current_token == orig_tokens[idx]:
                idx += 1
                current_token = ''
        return split_tokens, untokenized_indices_of_split_tokens

    def __call__(self, text):
        split_tokens, _ = self.tokenize(text)
        split_tokens = ["[CLS]"] + split_tokens + ["[SEP]"]

        input_ids = self.convert_tokens_to_ids(split_tokens)
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        }

    def tokenize_texts(self, texts, max_seq_length=None):
        current_max_seq_len = 0
        tokens = []
        for text in texts:
            split_tokens, _ = self.tokenize(text)
            if max_seq_length is not None:
                split_tokens = split_tokens[:max_seq_length-2]
            split_tokens = ["[CLS]"] + split_tokens + ["[SEP]"]
            current_max_seq_len = max(current_max_seq_len, len(split_tokens))
            tokens.append(split_tokens)

        max_seq_length = max_seq_length or 0
        max_seq_length = max(max_seq_length, current_max_seq_len)
        for split_tokens in tokens:
            split_tokens += ["[PAD]"] * (max_seq_length - len(split_tokens))

        input_ids = []
        token_type_ids = []
        attention_masks = []
        for split_tokens in tokens:
            input_ids_ = self.convert_tokens_to_ids(split_tokens)
            token_type_ids_ = [0] * len(input_ids_)
            attention_masks_ = [1] * len(input_ids_)

            input_ids.append(input_ids_)
            token_type_ids.append(token_type_ids_)
            attention_masks.append(attention_masks_)
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks,
        }

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[_] for _ in tokens]

    @staticmethod
    def _load_vocab(vocab_file, entity_type_list=None):
        non_selected_entity_type_list = entity_type_list or []
        vocab = OrderedDict()
        with open(vocab_file) as fh:
            for index, line in enumerate(fh):
                if not line:
                    break
                token = line.strip()
                if len(non_selected_entity_type_list) > 0 and token.startswith("[unused"):
                    ne_type = non_selected_entity_type_list.pop()
                    vocab[ne_type] = index
                else:
                    vocab[token] = index
        return vocab
