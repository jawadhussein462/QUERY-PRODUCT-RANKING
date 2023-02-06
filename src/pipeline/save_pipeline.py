"""Save Trained models"""

import os
import pickle

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

    # Create paths
    ranking_model_path = os.path.join(model_save_dir, ranking_model_path)
    cross_encoder_path = os.path.join(model_save_dir, cross_encoder_path)
    bm25_path = os.path.join(model_save_dir, bm25_path)

    bm25_path_us = os.path.join(bm25_path, "bm25_model_us.pkl")
    bm25_path_es = os.path.join(bm25_path, "bm25_model_es.pkl")
    bm25_path_jp = os.path.join(bm25_path, "bm25_model_jp.pkl")

    ranking_model_path = os.path.join(ranking_model_path, "ranking_model.pkl")

    # Save cross encoder model
    cross_encoder_model.save_model(model_path=cross_encoder_path)

    # Save bm25 models
    bm25_model_us.save_bm25(bm25_path_us)
    bm25_model_es.save_bm25(bm25_path_es)
    bm25_model_jp.save_bm25(bm25_path_jp)

    # Save ranking model
    with open(ranking_model_path, "wb") as file:
        pickle.dump(ranking_model, file)
