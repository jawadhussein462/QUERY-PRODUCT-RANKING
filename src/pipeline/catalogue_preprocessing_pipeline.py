"""
Prepare product catalogue data for training by performing several data cleaning steps.

Attributes
---------
    product_catalogue (pandas.DataFrame): The input data to be cleaned.

Returns
-------
    pandas.DataFrame: The cleaned data.
"""

import numpy as np
import regex as re
from pandas import DataFrame as D


def replace_nan_with_default(string: str, default_value: str = "") -> str:
    """
    Replace NaN values with the default value.

    Args
    ----
        string (str): The input string to be checked and replaced.
        default_value (str, optional): The value to replace NaN values with. Defaults to "".

    Returns
    -------
        str: The cleaned string.
    """

    if isinstance(string, float) and np.isnan(string):
        return ""

    else:
        return string


def remove_html_tage(html_text: str) -> str:
    """
    Remove HTML tags from the input text.

    Args
    ----
        html_text (str): The input text with HTML tags.

    Returns
    -------
        str: The cleaned text without HTML tags.
    """

    clean = re.compile("<.*?>")

    plain_text = re.sub(clean, "", html_text)

    return plain_text


def remove_emojis(text: str):
    """
    Remove emojis from the input text.

    Args
    ----
        text (str): The input text with emojis.

    Returns
    -------
        str: The cleaned text without emojis.
    """

    clean_text = re.sub(r"[^\p{L}\s]", "", text)

    return clean_text


def run(product_catalgue: D) -> D:
    """
    Launch all the main steps of the data cleaning process.

    Args
    ----
        product_catalogue (pandas.DataFrame): The input data to be cleaned.

    Returns
    -------
        pandas.DataFrame: The cleaned data.
    """

    df = product_catalgue.copy()

    # Convert to lower case
    df["product_title"] = df["product_title"].str.lower()
    df["product_description"] = df["product_description"].str.lower()

    # Replace NaNs with empty string
    df["product_title"] = df["product_title"].apply(replace_nan_with_default)
    df["product_description"] = df["product_description"].apply(
        replace_nan_with_default
    )

    # Remove Html tags
    df["product_title"] = df["product_title"].apply(remove_html_tage)
    df["product_description"] = df["product_description"].apply(remove_html_tage)

    # Remove emojis
    df["product_title"] = df["product_title"].apply(remove_emojis)
    df["product_description"] = df["product_description"].apply(remove_emojis)

    return df
