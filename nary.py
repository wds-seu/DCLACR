from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import pytorch_lightning
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from transformers.utils import PaddingStrategy

Instance = namedtuple("Instance", ("text", "label"))

DRUG_TOKEN = "@DRUG$"
_UNUSED1 = "[unused1]"
GENE_TOKEN = "@GENE$"
_UNUSED2 = "[unused2]"
VAR_TOKEN = "@VARIANT$"
_UNUSED3 = "[unused3]"
DRV_TOKEN_REPLACED = {
    DRUG_TOKEN: _UNUSED1,
    GENE_TOKEN: _UNUSED2,
    VAR_TOKEN : _UNUSED3,
}
DV_TOKEN_REPLACED = {
    DRUG_TOKEN: _UNUSED1,
    VAR_TOKEN : _UNUSED3,
}
LABEL_DICT = {
    "none": 0,
    "resistance": 1,
    "response": 2,
    "resistance or non-response": 3,
    "sensitivity": 4,
}


class _Dataset(Dataset):
    def __init__(
        self, filename, tokenizer: BertTokenizer, mode: str, max_seq_length: int = 512, header=None,
    ):
        super().__init__()
        self.filename = filename
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_seq_length = max_seq_length

        if mode == "drug_gene_var":
            self._special_tokens = DRV_TOKEN_REPLACED
        elif mode == "drug_var":
            self._special_tokens = DV_TOKEN_REPLACED
        else:
            raise NotImplementedError("For now, we just support drug_gene_var or drug_var mode")

        df = pd.read_csv(
            filename, sep="\t",
            names=(
                "index", "drug", "gene", "variant", "tagged_text", "neighbors", "label", "original_text"
            ),
            index_col="index",
            header=header
        )
        data = []
        for text, label in zip(df.tagged_text, df.label):
            replaced_text = self._replace_special_tokens(text).lower()
            item = tokenizer(replaced_text, return_length=True, return_tensors='pt')
            if item.length.item() > self.max_seq_length:
                continue
            item = tokenizer(
                replaced_text, return_length=False, return_tensors="pt",
                padding=PaddingStrategy.MAX_LENGTH, truncation=True, max_length=self.max_seq_length
            )
            data.append((item, LABEL_DICT[label.lower()]))

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Instance:
        return Instance(text=self.data[index][0], label=self.data[index][1])

    def _replace_special_tokens(self, text):
        for from_, to_ in self._special_tokens.items():
            text = text.replace(from_, to_)
        return text


@dataclass
class _Datasets:
    train: _Dataset
    val: _Dataset
    test: _Dataset


class NAryDataset(LightningDataModule):
    def __init__(
        self,
        data_dirpath: str,
        max_seq_length: int = 512,
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        *,
        force_not_shuffle: bool = False,
        mode: str = "drug_gene_var"
    ):
        super().__init__()
        self.data_dirpath = Path(data_dirpath)
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.force_not_shuffle = force_not_shuffle
        assert mode in ("drug_gene_var", "drug_var"), f"mode must in ('drug_gene_var', 'drug_var'), got {mode}"
        self.mode = mode

        # self.tokenizer = BertTokenizer.from_pretrained("/Users/wangran/Workspace/bert_relation_classification/biobert")
        self.tokenizer = BertTokenizer.from_pretrained("./biobert")
        self.datasets: Optional[_Datasets] = None
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if self.mode == "drug_gene_var":
            self.tokenizer.add_special_tokens(dict(
                additional_special_tokens=list(DRV_TOKEN_REPLACED.values())
            ))
        elif self.mode == "drug_var":
            self.tokenizer.add_special_tokens(dict(
                additional_special_tokens=list(DV_TOKEN_REPLACED.values())
            ))
        else:
            raise NotImplementedError("For now, we just support drug_gene_var or drug_var mode.")
        self.datasets = _Datasets(
            train=self._to_dataset(self.data_dirpath / 'train.tsv'),
            val=self._to_dataset(self.data_dirpath / 'dev.tsv'),
            test=self._to_dataset(self.data_dirpath / 'test.tsv', header=1),
        )

    def _to_dataset(self, filename, header: Optional[int] = None):
        return _Dataset(filename, self.tokenizer, self.mode, self.max_seq_length, header=header)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.datasets.train,
            batch_size=self.train_batch_size,
            num_workers=0,
            shuffle=not self.force_not_shuffle,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.datasets.val,
            batch_size=self.eval_batch_size,
            num_workers=0,
            shuffle=False
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.datasets.test,
            batch_size=self.eval_batch_size,
            num_workers=0,
            shuffle=False
        )
