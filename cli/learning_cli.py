# -*- coding: UTF-8 -*-

"""Main of the ranking system.
"""

import argparse
import os, sys
from typing import Dict
import warnings

sys.path.append(os.getcwd())

import nltk

from src.pipeline import get_data_pipeline
from src.pipeline import data_preprocessing_pipeline
from src.pipeline import catalogue_preprocessing_pipeline
from src.pipeline import train_cross_encoder_pipeline
from src.pipeline import train_bm25_pipeline
from src.pipeline import create_features_pipeline
from src.pipeline import evaluate_model_pipeline
from src.pipeline import save_pipeline
from src.configuration import data, app


def run():
    """Launch all the mains steps of the module."""

    set_message(message="Run query_product_ranking learning pipeline")

    # User options
    args = get_arguments()

    # Global configuration
    config, device = set_up_config()

    # Retrieving data
    set_message(message="STEP 1: Retrieving data")
    query_product_train, query_product_test, product_catalogue = get_data_pipeline.run(
        data_train_path=config.path["input_data"]["data_train_path"],
        data_test_path=config.path["input_data"]["data_test_path"],
        product_catalogue_path=config.path["input_data"]["product_catalogue_path"],
    )

    # Create product features
    set_message(message="STEP 2: Pre-processing product features")
    processed_product_catalogue = catalogue_preprocessing_pipeline.run(
        product_catalgue=product_catalogue
    )

    # Pre-process data
    set_message(message="STEP 3: Prepare data")
    x_train, x_val, y_train, y_val = data_preprocessing_pipeline.run(
        query_product_df=query_product_train,
        product_catalogue=processed_product_catalogue,
        labels_dict=config.data_structure["cross_encoder"]["labels_dict"],
        test_set=False,
        train_size=config.data_structure["data_preparation"]["train_size"],
        sampling_size=config.data_structure["data_preparation"]["sampling_size"],
        label_column=config.data_structure["data_preparation"]["label_column"],
    )

    x_test, _, _, _ = data_preprocessing_pipeline.run(
        query_product_df=query_product_test,
        product_catalogue=processed_product_catalogue,
        labels_dict=config.data_structure["cross_encoder"]["labels_dict"],
        test_set=True,
        train_size=None,
        sampling_size=config.data_structure["data_preparation"]["sampling_size"],
        label_column=config.data_structure["data_preparation"]["label_column"],
    )

    # Training Cross Encoder model
    set_message(message="STEP 4: Train cross encoder model")
    cross_encoder_model = train_cross_encoder_pipeline.run(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        base_model_name=config.model["cross_encoder"]["model_name"],
        base_model_path=config.path["cross_encoder"]["base_model_path"],
        cross_encoder_tokenizer_path=config.path["cross_encoder"][
            "cross_encoder_tokenizer_path"
        ],
        num_labels=config.data_structure["cross_encoder"]["num_labels"],
        max_seq_length=config.data_structure["cross_encoder"]["max_seq_length"],
        batch_size=config.model["cross_encoder"]["batch_size"],
        epochs=config.model["cross_encoder"]["epochs"],
        lr=config.model["cross_encoder"]["lr"],
        device=device,
    )

    # Training BM25
    set_message(message="STEP 5: Train BM25 model")
    bm25_model_us, bm25_model_es, bm25_model_jp = train_bm25_pipeline.run(
        product_catalgoue=processed_product_catalogue,
        lemmatization=config.model["bm_25"]["lemmatization"],
        product_description_column=config.data_structure["data_preparation"][
            "product_description_column"
        ],
        product_country_column=config.data_structure["data_preparation"][
            "product_country_column"
        ],
    )

    # Create Hand Crafted Features
    set_message(message="STEP 6: Create Hand Crafted Features")
    x_train, x_val, y_train, y_val = create_features_pipeline.run(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        cross_encoder_model=cross_encoder_model,
        bm25_model_us=bm25_model_us,
        bm25_model_es=bm25_model_es,
        bm25_model_jp=bm25_model_jp,
    )

    # Training model
    set_message(message="STEP 7: Train Ensemble model")
    model = train_cross_encoder_pipeline.run(
        x_train=x_train,
        y_train=y_train,
        model_name=config.model["model"],
        params=config.model["params"],
    )

    # Evaluate model
    set_message(message="STEP 8: Evaluate model")
    eval_results = evaluate_model_pipeline.run(x_test=x_test, model=model)

    set_message(message="STEP 9: Save model and evaluation")
    save_pipeline.run(
        model=model,
        evaluation=eval_results,
        model_save_path=config.path["output"]["model_save_path"],
        evaluation_save_path=config.path["output"]["evaluation_save_path"],
    )

    set_message(message="End of query product ranking main pipeline")


def get_arguments() -> argparse.Namespace:
    """Retrieve user parameters.

    Returns:
        argparse.Namespace: Object containing user parameters.
    """
    parser = argparse.ArgumentParser(description="PEPS Table and clustering")

    parser.add_argument(
        "--task",
        type=str,
        choices=["train", "predict", "evaluate"],
        default="train",
        help="specify what task should be performed, train, predict or evaluate",
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
