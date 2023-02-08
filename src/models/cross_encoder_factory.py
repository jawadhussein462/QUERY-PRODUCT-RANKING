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
    """
    CrossEncoderModelFactory is a class for creating, saving, and loading models for cross-encoder NLP tasks.

    The class has three main methods: `create_model`, `save_model`, and `load_model`.

    `create_model` creates a model and tokenizer given a `model_type` from the `CrossEncoderModelName` enum.
    `save_model` saves the given `model` and `tokenizer` to disk.
    `load_model` loads a saved `model` and `tokenizer` from disk given a `model_type` and file paths.

    The class also has a class-level dictionary `MODEL_CLASSES` that maps `model_type` to the appropriate model
    class, tokenizer class, and pre-trained model name.
    """

    MODEL_CLASSES = {
        CrossEncoderModelName.Bert.value: (
            BertModel,
            BertTokenizer,
            "bert-base-multilingual-cased",
        ),
        CrossEncoderModelName.Albert.value: (
            AlbertModel,
            AlbertTokenizer,
            "albert-base-multilingual-xl",
        ),
        CrossEncoderModelName.Distilbert.value: (
            DistilBertModel,
            DistilBertTokenizer,
            "distilbert-base-multilingual-cased",
        ),
    }

    def create_model(self, model_type, *args, **kwargs):
        """
        Parameters
        ----------
        model_type : str
            A string that corresponds to a value in the `CrossEncoderModelName` enum.
        args : Positional arguments
            Positional arguments to pass to the `from_pretrained` method.
        kwargs : Keyword arguments
            Keyword arguments to pass to the `from_pretrained` method.

        Returns
        -------
        tuple
            A tuple of the created `model` and `tokenizer`.
        """
        model_class, tokenizer_class, model_name = self.MODEL_CLASSES[model_type]
        model = model_class.from_pretrained(model_name, *args, **kwargs)
        tokenizer = tokenizer_class.from_pretrained(model_name, *args, **kwargs)
        return model, tokenizer

    def save_model(
        self, model=None, tokenizer=None, model_path=None, tokenizer_path=None
    ):
        """
        Saves the model and tokenizer to the specified paths.

        Parameters
        ----------
        model : transformers.PreTrainedModel, optional
            The model to save, by default `None`.
        tokenizer : transformers.PreTrainedTokenizer, optional
            The tokenizer to save, by default `None`.
        model_path : str, optional
            The path to save the model to, by default `None`.
        tokenizer_path : str, optional
            The path to save the tokenizer to, by default `None`.
        """
        if model is not None:
            model.save_pretrained(model_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(tokenizer_path)

    def load_model(self, model_type: str, model_path: str, tokenizer_path: str):

        """
        Loads the model and tokenizer from the specified paths.

        Parameters
        ----------
        model_type : str
            A string that corresponds to a value in the `CrossEncoderModelName` enum.
        model_path : str
            The path to load the model from.
        tokenizer_path : str
            The path to load the tokenizer from.

        Returns
        -------
        tuple
            A tuple of the loaded `model` and `tokenizer`.
        """

        model_class, tokenizer_class, _ = self.MODEL_CLASSES[model_type]
        model = model_class.from_pretrained(model_path)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        return model, tokenizer
