from typing import Optional

import pandas as pd
from pandas import DataFrame as D
from pandas import Series as S

from src.models.bm25_model import Bm25Model
from src.models.cross_encoder_model import CrossEncoderModel
from src.utils.constant import CountryCode


def bm25_score(
    row: S, bm25_model_us: Bm25Model, bm25_model_es: Bm25Model, bm25_model_jp: Bm25Model
):

    if row["query_locale"] == CountryCode.US.value:
        return bm25_model_us.score(row["query"], row["product_id"])
    if row["query_locale"] == CountryCode.Spanish.value:
        return bm25_model_es.score(row["query"], row["product_id"])
    if row["query_locale"] == CountryCode.Japanese.value:
        return bm25_model_jp.score(row["query"], row["product_id"])


def run(
    x: Optional[D],
    cross_encoder_model: CrossEncoderModel,
    num_labels: int,
    bm25_model_us: Bm25Model,
    bm25_model_es: Bm25Model,
    bm25_model_jp: Bm25Model,
):

    if x is None:
        return x

    # intialize data frame
    modified_x = pd.DataFrame()

    # get query_id
    modified_x["query_id"] = x["query_id"]

    # get product_id
    modified_x["product_id"] = x["product_id"]

    # query features
    modified_x["query_locale"] = x["query_locale"]

    # product features
    modified_x["product_locale"] = x["product_locale"]

    # query product features
    cross_encoder_predictions = cross_encoder_model.predict_proba(x)
    cross_encoder_columns = [f"cross_encoder_{i}" for i in range(num_labels)]
    cross_encoder_predictions = pd.DataFrame(
        cross_encoder_predictions, columns=cross_encoder_columns
    )

    modified_x = pd.concat([modified_x, cross_encoder_predictions], axis=1)

    modified_x["bm25_score"] = x.apply(
        lambda row: bm25_score(row, bm25_model_us, bm25_model_es, bm25_model_jp), axis=1
    )
    modified_x["same_county"] = (
        modified_x["query_locale"] == modified_x["product_locale"]
    ).astype(int)

    object_columns = modified_x.select_dtypes(["object"]).columns
    modified_x[object_columns] = modified_x[object_columns].apply(
        lambda x: x.astype("category")
    )

    return modified_x
