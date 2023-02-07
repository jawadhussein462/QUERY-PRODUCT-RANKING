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
    ranking_model_dir = os.path.join(model_save_dir, ranking_model_path)
    cross_encoder_dir = os.path.join(model_save_dir, cross_encoder_path)
    bm25_dir = os.path.join(model_save_dir, bm25_path)

    # Create path if not exist
    for path in [ranking_model_dir, cross_encoder_dir, bm25_dir]:
        os.makedirs(path, exist_ok=True)

    # Define file name
    ranking_model_file = os.path.join(model_save_dir, "ranking_model.pkl")
    cross_encoder_file = os.path.join(model_save_dir, "cross_encoder.pth")
    bm25_us_file = os.path.join(bm25_path, "bm25_model_us.pkl")
    bm25_es_file = os.path.join(bm25_path, "bm25_model_es.pkl")
    bm25_jp_file = os.path.join(bm25_path, "bm25_model_jp.pkl")

    # Save cross encoder model
    cross_encoder_model.save_model(path=cross_encoder_file)

    # Save ranking model
    with open(ranking_model_file, "wb") as file_name:
        pickle.dump(ranking_model, file_name)

    # Save bm25 models
    bm25_model_us.save_bm25(bm25_us_file)
    bm25_model_es.save_bm25(bm25_es_file)
    bm25_model_jp.save_bm25(bm25_jp_file)
