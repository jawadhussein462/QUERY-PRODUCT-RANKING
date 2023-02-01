"""Provide functionnality to apply train the model dataset."""

from typing import List, Any, Dict
from datetime import datetime

import pandas as pd
from pandas import DataFrame as D
from pandas import Series as S
import numpy as np
import torch


class BertModel(torch.nn.Module):
    def __init__(self, model: Any, num_labels: int):

        super(BertModel, self).__init__()
        self.model = model
        self.linear = torch.nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):

        last_hidden_state, pooler_output = self.model(
            input_ids=input_ids.squeeze(1),
            attention_mask=attention_mask,
            return_dict=False,
        )

        last_hidden_state_cls = last_hidden_state[:, 0, :]

        logits = self.linear(last_hidden_state_cls)

        return logits
