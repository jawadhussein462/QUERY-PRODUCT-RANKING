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
    num_labels: int,
    bm25_model_us: Any,
    bm25_model_es: Any,
    bm25_model_jp: Any,
) -> Tuple[D, S]:
    
    cross_encoder_predictions = cross_encoder_model.predict_proba(x_train)
    cross_encoder_columns = [f"cross_encoder_{i}" for i in range(num_labels)]
    cross_encoder_predictions = pd.DataFrame(cross_encoder_predictions, columns=cross_encoder_columns)

    modified_x_train = pd.concat([cross_encoder_predictions], axis=1)

    return modified_x_train, x_val, y_train, y_val
