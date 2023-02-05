"""Prepare data in order to train the model."""

from typing import Any

from pandas import DataFrame as D
from pandas import Series as S

from src.models.cross_encoder_model import CrossEncoderModel


def run(
    x_train: D,
    y_train: S,
    x_val: D,
    y_val: S,
    base_model_name: str,
    base_model_path: str,
    cross_encoder_tokenizer_path: str,
    num_labels: int,
    max_seq_length: int,
    batch_size: int,
    epochs: int,
    lr: float,
    device: Any,
) -> CrossEncoderModel:

    # Initialize the model
    cross_encoder = CrossEncoderModel(
        model_name=base_model_name,
        model_path=base_model_path,
        tokenizer_path=cross_encoder_tokenizer_path,
        num_labels=num_labels,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        device=device,
    )

    # fit the model
    cross_encoder.fit(x_train, y_train, x_val, y_val)

    return cross_encoder
