"""Prepare data in order to train the model."""

from typing import Tuple, Any
import time
import os

from pandas import DataFrame as D
from pandas import Series as S

from src.models.bm25_model import Bm25Model
from src.utils.constant import CountryCode


def run(
    product_catalgoue: D,
    lemmatization: bool,
    product_description_column: str,
    product_country_column: str,
) -> Tuple[Bm25Model, Bm25Model, Bm25Model]:

    # Separate languages
    corpus_us = product_catalgoue.apply(
        lambda row: row[product_description_column]
        if row[product_country_column] == CountryCode.US.value
        else "",
        axis=1,
    )

    corpus_es = product_catalgoue.apply(
        lambda row: row[product_description_column]
        if row[product_country_column] == CountryCode.Spanish.value
        else "",
        axis=1,
    )

    corpus_jp = product_catalgoue.apply(
        lambda row: row[product_description_column]
        if row[product_country_column] == CountryCode.Japanese.value
        else "",
        axis=1,
    )

    # Create Bm25
    bm25_model_us = Bm25Model(country=CountryCode.US.value, lemmatization=lemmatization)

    bm25_model_es = Bm25Model(
        country=CountryCode.Spanish.value, lemmatization=lemmatization
    )
    bm25_model_jp = Bm25Model(
        country=CountryCode.Japanese.value, lemmatization=lemmatization
    )

    # Intialize bm25
    bm25_model_us = bm25_model_us.intialize_bm25(corpus_us)
    bm25_model_es = bm25_model_es.intialize_bm25(corpus_es)
    bm25_model_jp = bm25_model_jp.intialize_bm25(corpus_jp)

    return bm25_model_us, bm25_model_es, bm25_model_jp
