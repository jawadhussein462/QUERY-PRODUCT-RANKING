"""stack results and create features for final ranking system"""

from typing import Optional

import pandas as pd
from pandas import DataFrame as D
from pandas import Series as S

from src.models.bm25_model import Bm25Model
from src.models.cross_encoder_model import CrossEncoderModel
from src.utils.constant import CountryCode


def bm25_score(
    row: S, bm25_model_us: Bm25Model, bm25_model_es: Bm25Model, bm25_model_jp: Bm25Model
) -> Optional[float]:

    """
     Calculates the BM25 score for a given query and product_id.

    Parameters
     ----------
         - row (pd.Series): The row of the data frame with 'query_locale', 'query' and 'product_id' columns.
         - bm25_model_us (Bm25Model): The BM25 model for US.
         - bm25_model_es (Bm25Model): The BM25 model for Spanish.
         - bm25_model_jp (Bm25Model): The BM25 model for Japanese.

     Returns:
         Optional[float]: The BM25 score for the given query and product_id, or None if the locale is not US, Spanish, or Japanese.
    """

    if row["query_locale"] == CountryCode.US.value:
        return bm25_model_us.score(row["query"], row["product_id"])
    elif row["query_locale"] == CountryCode.Spanish.value:
        return bm25_model_es.score(row["query"], row["product_id"])
    elif row["query_locale"] == CountryCode.Japanese.value:
        return bm25_model_jp.score(row["query"], row["product_id"])

    return None


def run(
    x: Optional[D],
    cross_encoder_model: CrossEncoderModel,
    num_labels: int,
    bm25_model_us: Bm25Model,
    bm25_model_es: Bm25Model,
    bm25_model_jp: Bm25Model,
) -> Optional[D]:

    """Stacks results and creates features for final ranking system.

    Parameters
     ----------

         - x (Optional[pd.DataFrame]): The input data frame with 'query_id', 'product_id', 'query_locale', and 'product_locale' columns.
         - cross_encoder_model (CrossEncoderModel): The cross-encoder model.
         - num_labels (int): The number of labels for the cross-encoder model.
         - bm25_model_us (Bm25Model): The BM25 model for US.
         - bm25_model_es (Bm25Model): The BM25 model for Spanish.
         - bm25_model_jp (Bm25Model): The BM25 model for Japanese.

     Returns:
         Optional[pd.DataFrame]: The modified data frame with added columns, or None if the input data frame is None.
    """

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
