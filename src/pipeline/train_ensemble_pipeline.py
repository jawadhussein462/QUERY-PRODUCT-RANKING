"""Prepare data in order to train the model."""

from typing import Tuple, Any, Dict, Optional
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
    x_val: Optional[D],
    y_val: Optional[S],
    params: Dict,
):

    # train
    group_train = x_train.query_id.value_counts().sort_index().values
    modified_x_train = x_train.sort_values(by=["query_id"], ignore_index=True)
    modified_x_train = modified_x_train.drop(["query_id", "product_id"], axis=1)
    
    # define eval_set
    eval_set = [(modified_x_train, y_train)]
    eval_group = [group_val]

    # add validation set
    if x_val is not None and y_val is not None: 
    
        group_val = x_val.query_id.value_counts().sort_index().values
        modified_x_val = x_val.sort_values(by=["query_id"], ignore_index=True)
        modified_x_val = modified_x_val.drop(["query_id", "product_id"], axis=1)
    
        eval_set.append((modified_x_val, y_val))
        eval_group.append(group_val)

    # initialize model
    model = LGBMRanker(**params)

    # fit the model
    model.fit(
        X=modified_x_train,
        y=y_train,
        group=group_train,
        eval_set=eval_set,
        eval_group=eval_group,
        eval_at=10,
        verbose=10,
    )

    return model
