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
    save_pipeline,
    train_bm25_pipeline,
    train_cross_encoder_pipeline,
    train_ensemble_pipeline,
)


def run():
    """Launch all the mains steps of the module."""

    set_message(message="Run query_product_ranking learning pipeline")

    # Global configuration
    config, device = set_up_config()

    # User options
    args = get_arguments(
        train_path=config.path["input_data"]["query_product_train_path"],
        catalogue_path=config.path["input_data"]["product_catalogue_path"]
    )

    # Retrieving data
    set_message(message="STEP 1: Retrieving data")
    query_product_train, product_catalogue = get_data_pipeline.run(
        query_product_path=args.train_data,
        product_catalogue_path=args.product_catalogue,
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
        product_id_column=config.data_structure["data_preparation"][
            "product_id_column"
        ],
    )

    # Create Hand Crafted Features
    set_message(message="STEP 6: Create Hand Crafted Features")
    modified_x_train = create_features_pipeline.run(
        x=x_train,
        cross_encoder_model=cross_encoder_model,
        num_labels=config.data_structure["cross_encoder"]["num_labels"],
        bm25_model_us=bm25_model_us,
        bm25_model_es=bm25_model_es,
        bm25_model_jp=bm25_model_jp,
    )

    modified_x_val = create_features_pipeline.run(
        x=x_val,
        cross_encoder_model=cross_encoder_model,
        num_labels=config.data_structure["cross_encoder"]["num_labels"],
        bm25_model_us=bm25_model_us,
        bm25_model_es=bm25_model_es,
        bm25_model_jp=bm25_model_jp,
    )

    # Training model
    set_message(message="STEP 7: Train Ensemble model")
    ranking_model = train_ensemble_pipeline.run(
        x_train=modified_x_train,
        y_train=y_train,
        x_val=modified_x_val,
        y_val=y_val,
        query_id_column=config.data_structure["data_preparation"]["query_id_column"],
        product_id_column=config.data_structure["data_preparation"][
            "product_id_column"
        ],
        params=config.model["lgbmranker"],
    )

    # Save model
    set_message(message="STEP 8: Save model and evaluation")
    save_pipeline.run(
        ranking_model=ranking_model,
        cross_encoder_model=cross_encoder_model,
        bm25_model_es=bm25_model_es,
        model_save_path=config.path["output"]["model_save_path"])

    set_message(message="End of query product ranking main pipeline")


def get_arguments(train_path: str,
                  catalogue_path: str) -> argparse.Namespace:
    """Retrieve user parameters.

    Returns:
        argparse.Namespace: Object containing user parameters.
    """
    parser = argparse.ArgumentParser(description="data path")

    parser.add_argument(
        "--train_data",
        type=str,
        default=train_path,
        help="specify the path of the training data",
    )

    parser.add_argument(
        "--product_catalogue",
        type=str,
        default=catalogue_path,
        help="specify the path of the training data",
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
