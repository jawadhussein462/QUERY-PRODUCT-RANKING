from transformers import (
    AlbertModel,
    AlbertTokenizer,
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
)

from src.utils.constant import CrossEncoderModelName


class CrossEncoderModelFactory:

    MODEL_CLASSES = {
        CrossEncoderModelName.Bert.value: (
            BertModel,
            BertTokenizer,
            "bert-base-multilingual-cased",
        ),
        CrossEncoderModelName.Albert.value: (
            AlbertModel,
            AlbertTokenizer,
            "bert-base-multilingual-cased",
        ),
        CrossEncoderModelName.Distilbert.value: (
            DistilBertModel,
            DistilBertTokenizer,
            "bert-base-multilingual-cased",
        ),
    }

    def create_model(self, model_type, *args, **kwargs):
        model_class, tokenizer_class, model_name = self.MODEL_CLASSES[model_type]
        model = model_class.from_pretrained(model_name, *args, **kwargs)
        tokenizer = tokenizer_class.from_pretrained(model_name, *args, **kwargs)
        return model, tokenizer

    def save_model(
        self, model=None, tokenizer=None, model_path=None, tokenizer_path=None
    ):
        if model is not None:
            model.save_pretrained(model_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(tokenizer_path)

    def load_model(self, model_type: str, model_path: str, tokenizer_path: str):
        model_class, tokenizer_class, _ = self.MODEL_CLASSES[model_type]
        model = model_class.from_pretrained(model_path)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        return model, tokenizer
