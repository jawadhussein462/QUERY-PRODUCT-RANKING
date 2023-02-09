"""Save Trained models"""

import os

import dill as pickle
from lightgbm import LGBMRanker

from src.models.cross_encoder_model import CrossEncoderModel
from src.models.bm25_model import Bm25Model


def run(
    ranking_model: LGBMRanker,
    cross_encoder_model: CrossEncoderModel,
    bm25_model_es: Bm25Model,
    bm25_model_us: Bm25Model,
    bm25_model_jp: Bm25Model,
    model_save_dir: str,
    ranking_model_path: str,
    cross_encoder_path: str,
    bm25_path: str,
) -> None:

    """
    This function saves the trained models of ranking model, cross-encoder model, and bm25 models.
    The models are saved as pickle files.
    The function creates the path to save the models if the path does not exist.

    Parameters:
    ----------

        - ranking_model (LGBMRanker): Instance of LGBMRanker class.
        - cross_encoder_model (CrossEncoderModel): Instance of CrossEncoderModel class.
        - bm25_model_es (Bm25Model): Instance of Bm25Model for Spanish language.
        - bm25_model_us (Bm25Model): Instance of Bm25Model for English language.
        - bm25_model_jp (Bm25Model): Instance of Bm25Model for Japanese language.
        - model_save_dir (str): Path to save the models.
        - ranking_model_path (str): Sub-directory name to save the ranking model.
        - cross_encoder_path (str): Sub-directory name to save the cross-encoder model.
        - bm25_path (str): Sub-directory name to save the bm25 models.

    Returns:
    ----------

        None

    """
    # Create paths to save the models
    ranking_model_dir = os.path.join(model_save_dir, ranking_model_path)
    cross_encoder_dir = os.path.join(model_save_dir, cross_encoder_path)
    bm25_dir = os.path.join(model_save_dir, bm25_path)

    # Create the directory if it does not exist
    for path in [ranking_model_dir, cross_encoder_dir, bm25_dir]:
        os.makedirs(path, exist_ok=True)

    # Define file names to save the models
    ranking_model_file = os.path.join(ranking_model_dir, "ranking_model.pkl")
    cross_encoder_file = os.path.join(cross_encoder_dir, "cross_encoder.pkl")
    bm25_file_us = os.path.join(bm25_dir, "bm25_model_us.pkl")
    bm25_file_es = os.path.join(bm25_dir, "bm25_model_es.pkl")
    bm25_file_jp = os.path.join(bm25_dir, "bm25_model_jp.pkl")

    # Save models
    for model, file_path in zip(
        [
            ranking_model,
            cross_encoder_model,
            bm25_model_us,
            bm25_model_es,
            bm25_model_jp,
        ],
        [
            ranking_model_file,
            cross_encoder_file,
            bm25_file_us,
            bm25_file_es,
            bm25_file_jp,
        ],
    ):

        with open(file_path, "wb") as file_name:
            pickle.dump(model, file_name)
