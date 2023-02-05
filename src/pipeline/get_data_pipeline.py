"""Retrieve raw data for the processing."""

import os
from typing import Tuple

import pandas as pd
from pandas import DataFrame as D


def get_data(file_path: str) -> D:

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError("The file does not exist.")

    # Check if the file is a CSV file
    if not file_path.endswith(".csv"):
        raise ValueError("The file is not a CSV file.")

    data = pd.read_csv(file_path)
    return data


def run(
    query_product_path: str, product_catalogue_path: str
) -> Tuple[D, D]:

    """Launch all the mains steps of the module."""

    # get data
    query_product_data = get_data(query_product_path)
    product_catalogue = get_data(product_catalogue_path)

    return query_product_data, product_catalogue
