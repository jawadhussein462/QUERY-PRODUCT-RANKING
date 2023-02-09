"""Train BM25 model for ranking"""

from typing import Tuple

from pandas import DataFrame as D

from src.models.bm25_model import Bm25Model
from src.utils.constant import CountryCode


def run(
    product_catalgoue: D,
    lemmatization: bool,
    product_description_column: str,
    product_country_column: str,
    product_id_column: str,
) -> Tuple[Bm25Model, Bm25Model, Bm25Model]:

    """
    This function trains BM25 models for three languages: US, Spanish and Japanese.
    The training data is provided as a pandas DataFrame called 'product_catalogue' which
    should contain at least the columns specified in 'product_description_column',
    'product_country_column' and 'product_id_column'. The 'lemmatization' parameter
    indicates whether the text pre-processing should include lemmatization.

    Parameters
    ----------
        - product_catalogue (pandas.DataFrame): A DataFrame containing the training data
        - lemmatization (bool): A flag indicating whether to use lemmatization for text pre-processing
        - product_description_column (str): The name of the column in 'product_catalogue' that
            contains the product descriptions.
        - product_country_column (str): The name of the column in 'product_catalogue' that
            contains the product country codes.
        - product_id_column (str): The name of the column in 'product_catalogue' that
            contains the product ids.

    Returns
    ----------
        Tuple[Bm25Model, Bm25Model, Bm25Model]: A tuple of BM25 models for US, Spanish and Japanese.
    """

    # Separate languages
    condition_us = product_catalgoue[product_country_column] == CountryCode.US.value
    corpus_us = product_catalgoue[product_description_column][condition_us].reset_index(
        drop=True
    )
    product_ids_us = product_catalgoue[product_id_column][condition_us].reset_index(
        drop=True
    )

    condition_es = (
        product_catalgoue[product_country_column] == CountryCode.Spanish.value
    )
    corpus_es = product_catalgoue[product_description_column][condition_es].reset_index(
        drop=True
    )
    product_ids_es = product_catalgoue[product_id_column][condition_es].reset_index(
        drop=True
    )

    condition_jp = (
        product_catalgoue[product_country_column] == CountryCode.Japanese.value
    )
    corpus_jp = product_catalgoue[product_description_column][condition_jp].reset_index(
        drop=True
    )
    product_ids_jp = product_catalgoue[product_id_column][condition_jp].reset_index(
        drop=True
    )

    # Create Bm25
    bm25_model_us = Bm25Model(
        country_code=CountryCode.US.value, lemmatization=lemmatization
    )

    bm25_model_es = Bm25Model(
        country_code=CountryCode.Spanish.value, lemmatization=lemmatization
    )

    bm25_model_jp = Bm25Model(
        country_code=CountryCode.Japanese.value, lemmatization=lemmatization
    )

    # Intialize bm25
    bm25_model_us.intialize_bm25(corpus_us, product_ids_us)
    bm25_model_es.intialize_bm25(corpus_es, product_ids_es)
    bm25_model_jp.intialize_bm25(corpus_jp, product_ids_jp)

    return bm25_model_us, bm25_model_es, bm25_model_jp
