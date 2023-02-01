from typing import Tuple, Any

from pandas import DataFrame as D
from pandas import Series as S
import pandas as pd

from src.models.cross_encoder_model import CrossEncoderModel


def run(
    x_train: D,
    y_train: S,
    x_val: D,
    y_val: S,
    cross_encoder_model: Any,
    bm25_model_us: Any,
    bm25_model_es: Any,
    bm25_model_jp: Any,
) -> Tuple[D, S]:

    modified_x_train = pd.DataFrame()
    predictions = cross_encoder_model.predict(x_train)
    return modified_x_train, x_val, y_train, y_val
