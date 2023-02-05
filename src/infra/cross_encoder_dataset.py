from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pandas import DataFrame as D
from pandas import Series as S
from torch.utils.data import DataLoader, Dataset


class CrossEncoderDataset(Dataset):
    def __init__(
        self, tokenizer: Any, max_seq_length: int, x: D, y: Optional[S] = None
    ):
        super(CrossEncoderDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):

        query = self.x.iloc[index]["query"]
        product_title = self.x.iloc[index]["product_title"]
        product_description = self.x.iloc[index]["product_description"]
        product_bullet_point = self.x.iloc[index]["product_bullet_point"]
        product_brand = self.x.iloc[index]["product_brand"]
        label = self.y.iloc[index] if self.y is not None else None

        input_text = query + " [SEP] " + product_title + " [SEP] " + product_description

        input_encoded = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = input_encoded["input_ids"]
        token_type_ids = input_encoded["token_type_ids"]
        attention_mask = input_encoded["attention_mask"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
            if label is not None
            else torch.zeros(len(self.x)),
        }
