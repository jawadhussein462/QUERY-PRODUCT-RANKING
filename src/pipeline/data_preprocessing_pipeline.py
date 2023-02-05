"""Prepare data in order to train the model."""

import re
from typing import Dict, Optional, Tuple

import pandas as pd
from pandas import DataFrame as D
from pandas import Series as S


def train_val_split_pandas(x: D, y: S, train_size: float):

    x_train = x.sample(frac=train_size, random_state=200)
    x_val = x.drop(x_train.index)
    y_train = y.loc[x_train.index]
    y_val = y.drop(x_train.index)

    x_train.reset_index(drop=True, inplace=True)
    x_val.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)

    return x_train, x_val, y_train, y_val


def run(
    query_product_df: D,
    product_catalogue: D,
    labels_dict: Dict,
    label_column: str,
    test_set: bool = False,
    train_size: Optional[float] = None,
    sampling_size: Optional[float] = None,
) -> Tuple[D, S]:

    # Merge query product table and product catalogue on product id
    data = pd.merge(query_product_df, product_catalogue, how="left", on="product_id")

    # lower case query
    data["query"] = data["query"].str.lower()

    if sampling_size is not None:
        data = data.sample(frac=sampling_size, ignore_index=True)

    if test_set:

        x = data

        return x, None, None, None

    else:

        # Split data into X and y
        x = data.drop(label_column, axis=1)
        y = data[label_column]

        # Encode labels
        y = y.replace(labels_dict)

        # train val split
        if train_size is not None:

            x_train, x_val, y_train, y_val = train_val_split_pandas(x, y, train_size)

        else:

            x_train = x
            x_val = None
            y_train = y
            y_val = None

        return x_train, x_val, y_train, y_val
