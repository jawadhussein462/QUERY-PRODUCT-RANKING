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
    params : Dict
) -> CrossEncoderModel:

        

    return None
