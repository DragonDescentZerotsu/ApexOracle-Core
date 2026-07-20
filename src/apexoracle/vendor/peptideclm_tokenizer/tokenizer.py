"""PeptideCLM's SMILES Pair Encoding tokenizer used by ApexOracle.

This is the inference-relevant subset of PeptideCLM's MIT-licensed
``tokenizer/my_tokenizers.py``. Token IDs and special-token construction are
kept unchanged; unrelated atomwise tokenizers and tutorial code are omitted.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Optional

from SmilesPE.tokenizer import SPE_Tokenizer
from transformers import PreTrainedTokenizer


def load_vocab(vocab_file: str | Path) -> collections.OrderedDict[str, int]:
    with Path(vocab_file).open(encoding="utf-8") as reader:
        return collections.OrderedDict(
            (token.rstrip("\n"), index) for index, token in enumerate(reader)
        )


class SMILES_SPE_Tokenizer(PreTrainedTokenizer):
    """SMILES Pair Encoding tokenizer used by the PeptideCLM checkpoint."""

    def __init__(
        self,
        vocab_file: str | Path,
        spe_file: str | Path,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        vocab_path = Path(vocab_file)
        spe_path = Path(spe_file)
        if not vocab_path.is_file():
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_path}'.")
        if not spe_path.is_file():
            raise ValueError(f"Can't find a SPE vocabulary file at path '{spe_path}'.")

        self.vocab = load_vocab(vocab_path)
        self.spe_vocab = spe_path.open(encoding="utf-8")
        self.ids_to_tokens = collections.OrderedDict(
            (index, token) for token, index in self.vocab.items()
        )
        self.spe_tokenizer = SPE_Tokenizer(self.spe_vocab)
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text: str) -> list[str]:
        return self.spe_tokenizer.tokenize(text).split(" ")

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs,
    ) -> str:
        del clean_up_tokenization_spaces, kwargs
        tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        return self.convert_tokens_to_string(tokens)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens).replace(" ##", "").strip()

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: Optional[list[int]] = None,
    ) -> list[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return (
            [self.cls_token_id]
            + token_ids_0
            + [self.sep_token_id]
            + token_ids_1
            + [self.sep_token_id]
        )

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: Optional[list[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "Do not supply a second sequence when special tokens are already present."
                )
            return [
                int(token in {self.sep_token_id, self.cls_token_id})
                for token in token_ids_0
            ]
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: list[int],
        token_ids_1: Optional[list[int]] = None,
    ) -> list[int]:
        first = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        if token_ids_1 is None:
            return [0] * len(first)
        return [0] * len(first) + [1] * (len(token_ids_1) + 1)

    def save_vocabulary(self, save_directory: str, filename_prefix=None):
        destination = Path(save_directory)
        if destination.is_dir():
            prefix = f"{filename_prefix}-" if filename_prefix else ""
            destination = destination / f"{prefix}vocab.txt"
        with destination.open("w", encoding="utf-8") as writer:
            for token, _ in sorted(self.vocab.items(), key=lambda item: item[1]):
                writer.write(f"{token}\n")
        return (str(destination),)
