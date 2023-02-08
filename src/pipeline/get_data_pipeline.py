"""
This module is responsible for retrieving the raw data from the file system.
"""

import os
from typing import Tuple

import pandas as pd
from pandas import DataFrame as D


def get_data(file_path: str) -> D:
    """Retrieve data from a CSV file.

    Parameters
    ----------
    file_path: str
        The path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        The data loaded from the CSV file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a CSV file.
    """

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError("The file does not exist.")

    # Check if the file is a CSV file
    if not file_path.endswith(".csv"):
        raise ValueError("The file is not a CSV file.")

    data = pd.read_csv(file_path)
    return data


def run(query_product_path: str, product_catalogue_path: str) -> Tuple[D, D]:

    """Launch all the main steps of the module.

    Parameters
    ----------
    query_product_path: str
        The path to the CSV file containing the query product data.
    product_catalogue_path: str
        The path to the CSV file containing the product catalogue data.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]
        The query product data and the product catalogue data.
    """

    # get data
    query_product_data = get_data(query_product_path)
    product_catalogue = get_data(product_catalogue_path)

    return query_product_data, product_catalogue
