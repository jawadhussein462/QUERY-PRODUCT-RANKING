"""Predict data in order to train the model."""

from pathlib import Path

from lightgbm import LGBMRanker
import pandas as pd
from pandas import DataFrame as D


def run(
    x: D,
    ranking_model: LGBMRanker,
    prediction_file: str,
    query_id_column: str,
    product_id_column: str,
):

    # Create path
    prediction_file = Path(prediction_file)
    prediction_file.parent.mkdir(exist_ok=True, parents=True)

    # Pre-process X
    modified_x = x.drop([query_id_column, product_id_column], axis=1)

    # Predictions
    prediction = pd.DataFrame()
    prediction[query_id_column] = x[query_id_column]
    prediction[product_id_column] = x[product_id_column]
    prediction["score"] = ranking_model.predict(modified_x)

    return prediction
