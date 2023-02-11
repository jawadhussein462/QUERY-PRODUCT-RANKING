"""Prepare data in order to train the model."""

from typing import Dict, Optional, Tuple

import pandas as pd
from pandas import DataFrame as D
from pandas import Series as S


def train_val_split_pandas(x: D, y: S, train_size: float) -> Tuple[D, D, S, S]:
    """
    Splits the `x` dataframe and the `y` series into training and validation sets based on `train_size`.

    Parameters
    ----------

        - x (pandas.DataFrame): DataFrame with feature data
        - y (pandas.Series): Series with target data
        - train_size (float): Fraction of the data to be used for training, must be in range [0, 1]

    Returns:
    ----------

        tuple of four elements:
        - The training dataframe (`x_train`)
        - The validation dataframe (`x_val`)
        - The training labels (`y_train`)
        - The validation labels (`y_val`)
    """

    # Use the `sample` method to split the dataframe into training and validation sets
    x_train = x.sample(frac=train_size, random_state=200)
    x_val = x.drop(x_train.index)

    # Select the corresponding labels for each set
    y_train = y.loc[x_train.index]
    y_val = y.drop(x_train.index)

    # Reset the index to avoid any confusion when using the data later
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
) -> Tuple[D, Optional[D], S, Optional[S]]:
    """
    This is the main function that takes several inputs, performs preprocessing on the data, and returns the processed data.

    Parameters
    ----------

    - `query_product_df`: A pandas DataFrame containing the query product information
    - `product_catalogue`: A pandas DataFrame containing the product catalogue information
    - `labels_dict`: A dictionary mapping the categorical labels to numerical values
    - `label_column`: A string representing the name of the column containing the labels in the `data` DataFrame
    - `test_set`: A boolean indicating whether the input data should be considered as a test set (default: False)
    - `train_size`: A float representing the size of the training set (default: None)
    - `sampling_size`: A float representing the fraction of the data to be used (default: None)

    Returns:
    ----------

    A tuple of four elements:
        - The training dataframe (`x_train`)
        - The validation dataframe (`x_val`), if `train_size` is not None
        - The training labels (`y_train`)
        - The validation labels (`y_val`),
    """

    # Merge query product table and product catalogue on product id
    data = pd.merge(query_product_df, product_catalogue, how="left", on="product_id")

    # lower case query
    data["query"] = data["query"].str.lower()

    if sampling_size is not None:
        queries_sampled = (
            data["query"]
            .drop_duplicates()
            .sample(frac=sampling_size, ignore_index=True)
        )
        data = data[data["query"].isin(queries_sampled)].reset_index(drop=True)
        # data = data.sample(frac=sampling_size, ignore_index=True)

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
