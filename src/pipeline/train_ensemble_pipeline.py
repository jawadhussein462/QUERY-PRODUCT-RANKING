"""Prepare data in order to train the model."""

from typing import Dict, Optional

from lightgbm import LGBMRanker
from pandas import DataFrame as D
from pandas import Series as S


def run(
    x_train: D,
    y_train: S,
    x_val: Optional[D],
    y_val: Optional[S],
    query_id_column: str,
    product_id_column: str,
    params: Dict,
):

    # train
    group_train = x_train[query_id_column].value_counts().sort_index().values
    modified_x_train = x_train.sort_values(by=[query_id_column], ignore_index=True)
    modified_x_train = modified_x_train.drop(
        [query_id_column, product_id_column], axis=1
    )

    # define eval_set
    eval_set = [(modified_x_train, y_train)]
    eval_group = [group_train]

    # add validation set
    if x_val is not None and y_val is not None:

        group_val = x_val[query_id_column].value_counts().sort_index().values
        modified_x_val = x_val.sort_values(by=[query_id_column], ignore_index=True)
        modified_x_val = modified_x_val.drop(
            [query_id_column, product_id_column], axis=1
        )

        eval_set.append((modified_x_val, y_val))
        eval_group.append(group_val)

    # initialize model
    model = LGBMRanker(**params)

    # fit the model
    model.fit(
        X=modified_x_train,
        y=y_train,
        group=group_train,
        eval_set=eval_set,
        eval_group=eval_group,
        eval_at=10,
        verbose=10,
    )

    return model
