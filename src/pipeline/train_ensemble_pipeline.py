"""Prepare data in order to train the model."""

from typing import Tuple, Any, Dict
import time
import os

from pandas import DataFrame as D
from pandas import Series as S
import numpy as np

from lightgbm import LGBMRanker

from src.infra import cross_encoder_dataset
from src.models.cross_encoder_model import CrossEncoderModel


def run(
    x_train: D,
    y_train: S,
    x_val: D,
    y_val: S,
    params: Dict,
):

    # train
    group_train = x_train.query_id.value_counts().sort_index().values
    modified_x_train = x_train.sort_values(by=["query_id"], ignore_index=True)
    modified_x_train = modified_x_train.drop(["query_id", "product_id"], axis=1)

    # val
    group_val = x_val.query_id.value_counts().sort_index().values
    modified_x_val = x_val.sort_values(by=["query_id"], ignore_index=True)
    modified_x_val = modified_x_val.drop(["query_id", "product_id"], axis=1)

    # initialize model
    model = LGBMRanker(**params)

    # fit the model
    model.fit(
        X=modified_x_train,
        y=y_train,
        group=group_train,
        eval_set=[(modified_x_val, y_val)],
        eval_group=[group_val],
        eval_at=10,
        verbose=10,
    )

    return model
