"""Provide functionnality to apply train the model dataset."""

from typing import Any

import torch


class BertModel(torch.nn.Module):
    """
    A PyTorch implementation of a BERT model for text classification.

    This class extends torch.nn.Module and implements a forward method to perform
    the forward pass through the model. It takes as input `input_ids` and `attention_mask`
    and outputs the logits for each class. The BERT model is passed as an argument during
    initialization and the number of labels in the target dataset is also provided.
    """
    def __init__(self, model: Any, num_labels: int):

        """
        Initialize the BertModel.

        Parameters
        ----------

        - model: any model instance, the BERT model to be used.
        - num_labels: int, the number of labels or classes in the target dataset.

        """

        super(BertModel, self).__init__()
        self.model = model
        self.linear = torch.nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):

        """
        Perform the forward pass through the model.

        Parameters
        ----------

        - input_ids: torch tensor of shape [batch_size, seq_length], the input ids for the BERT model.
        - attention_mask: torch tensor of shape [batch_size, seq_length], the attention mask for the BERT model.

        Returns
        ----------

        - logits: torch tensor of shape [batch_size, num_labels], the logits for each class.

        """

        last_hidden_state, pooler_output = self.model(
            input_ids=input_ids.squeeze(1),
            attention_mask=attention_mask,
            return_dict=False,
        )

        last_hidden_state_cls = last_hidden_state[:, 0, :]

        logits = self.linear(last_hidden_state_cls)

        return logits
