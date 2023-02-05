"""Save Trained models"""

from lightgbm import LGBMRanker

from src.models.cross_encoder_model import CrossEncoderModel
from src.models.bm25_model import Bm25Model


def run(
    ranking_model: LGBMRanker,
    cross_encoder_model: CrossEncoderModel,
    bm25_model_es: Bm25Model,
    bm25_model_us: Bm25Model,
    bm25_model_jp: Bm25Model,
    ranking_model_save_path: str,
    cross_encoder_save_path: str,
    bm25_model_save_path: str,
) -> NotImplemented:

    pass
