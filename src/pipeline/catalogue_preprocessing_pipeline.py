"""Prepare data in order to train the model."""
import numpy as np
import regex as re
from pandas import DataFrame as D


def replace_nan_with_default(string: str, default_value: str = "") -> str:

    if isinstance(string, float) and np.isnan(string):
        return ""

    else:
        return string


def remove_html_tage(html_text: str) -> str:

    clean = re.compile("<.*?>")

    plain_text = re.sub(clean, "", html_text)

    return plain_text


def remove_emojis(text: str):

    clean_text = re.sub(r"[^\p{L}\s]", "", text)

    return clean_text


def run(product_catalgue: D) -> D:

    """Launch all the mains steps of the module."""
    df = product_catalgue.copy()

    # str lower
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

    # lower case
    df["product_title"] = df["product_title"].str.lower()
    df["product_description"] = df["product_description"].str.lower()

    return df
