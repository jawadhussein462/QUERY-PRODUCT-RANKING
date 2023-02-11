# -*- coding: UTF-8 -*-

"""Main of the ranking system.
"""

import argparse
import os
import sys
import warnings

import nltk

sys.path.append(os.getcwd())

from src.configuration import app, data
from src.pipeline import (
    catalogue_preprocessing_pipeline,
    create_features_pipeline,
    data_preprocessing_pipeline,
    get_data_pipeline,
    load_models_pipeline,
    prediction_pipeline,
)


def run():
    """Launch all the mains steps of the module."""

    set_message(message="Run query_product_ranking learning pipeline")

    # Global configuration
    config, device = set_up_config()

    # User options
    args = get_arguments(
        test_path=config.path["inputs"]["query_product_train_path"],
        catalogue_path=config.path["inputs"]["product_catalogue_path"],
        model_load_dir=config.path["inputs"]["model_load_dir"],
        output=config.path["outputs"]["prediction_output"],
    )

    # Retrieving data
    set_message(message="STEP 1: Retrieving data")
    query_product_train, product_catalogue = get_data_pipeline.run(
        query_product_path=args.test_data,
        product_catalogue_path=args.product_catalogue,
    )

    # Create product features
    set_message(message="STEP 2: Pre-processing product features")
    processed_product_catalogue = catalogue_preprocessing_pipeline.run(
        product_catalgue=product_catalogue
    )

    # Pre-process data
    set_message(message="STEP 3: Prepare data")
    x_test, _, _, _ = data_preprocessing_pipeline.run(
        query_product_df=query_product_train,
        product_catalogue=processed_product_catalogue,
        labels_dict=config.data_structure["cross_encoder"]["labels_dict"],
        test_set=True,
        train_size=config.data_structure["data_preparation"]["train_size"],
        sampling_size=config.data_structure["data_preparation"]["sampling_size"],
        label_column=config.data_structure["data_preparation"]["label_column"],
    )

    # Load models
    set_message(message="STEP 4: Load trained models")
    (
        cross_encoder_model,
        ranking_model,
        bm25_model_us,
        bm25_model_es,
        bm25_model_jp,
    ) = load_models_pipeline.run(
        model_load_dir=args.model_load_dir,
        ranking_model_path=config.path["models"]["ranking_model_path"],
        cross_encoder_path=config.path["models"]["cross_encoder"]["cross_encoder_path"],
        bm25_us_path=config.path["models"]["bm25_us_path"],
        bm25_es_path=config.path["models"]["bm25_es_path"],
        bm25_jp_path=config.path["models"]["bm25_jp_path"],
    )

    # Create Hand Crafted Features
    set_message(message="STEP 5: Create Hand Crafted Features")
    modified_x_test = create_features_pipeline.run(
        x=x_test,
        cross_encoder_model=cross_encoder_model,
        num_labels=config.data_structure["cross_encoder"]["num_labels"],
        bm25_model_us=bm25_model_us,
        bm25_model_es=bm25_model_es,
        bm25_model_jp=bm25_model_jp,
    )

    # Prediction
    set_message(message="STEP 6: Make and save predictions")
    predictions = prediction_pipeline.run(
        x=modified_x_test,
        ranking_model=ranking_model,
        prediction_file=args.output,
        query_id_column=config.data_structure["data_preparation"]["query_id_column"],
        product_id_column=config.data_structure["data_preparation"][
            "product_id_column"
        ],
    )

    print(predictions)

    set_message(message="End of query product prediction pipeline")


def get_arguments(
    test_path: str, catalogue_path: str, model_load_dir: str, output: str
) -> argparse.Namespace:
    """Retrieve user parameters.

    Returns:
        argparse.Namespace: Object containing user parameters.
    """
    parser = argparse.ArgumentParser(description="data path")

    parser.add_argument(
        "--test_data",
        type=str,
        default=test_path,
        help="the path to the test data file, in CSV format.",
    )

    parser.add_argument(
        "--product_catalogue",
        type=str,
        default=catalogue_path,
        help="the path to the product catalogue file, in CSV format. This file should be the same as the one used for training",
    )

    parser.add_argument(
        "--model_load_dir",
        type=str,
        default=model_load_dir,
        help="the path to the output file where the predictions will be saved, in CSV format.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=output,
        help="the path to the output file where the predictions will be saved, in CSV format.",
    )

    return parser.parse_args()


def set_up_config() -> data.DataConfig:
    """Retrieve global project config.

    Returns:
        data.DataConfig: Global project config object.
    """

    warnings.filterwarnings("ignore")

    nltk.download("punkt")
    nltk.download("stopwords")

    config = data.DataConfig()

    # get torch device
    device = app.get_device()

    # set seed
    app.set_seed(seed_value=config.model["seed_value"])

    return config, device


def set_message(message: str) -> None:
    """Set message in log files."""
    print(message)


if __name__ == "__main__":
    """Launch the program as following"""
    run()
