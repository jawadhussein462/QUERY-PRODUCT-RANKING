from typing import Optional

from pandas import Series as S
from rank_bm25 import BM25Okapi
from spacy.tokenizer import Tokenizer

from src.utils.support import get_spacy_from_country


class Bm25Model:
    """
    A BM25 ranking model to rank products based on their text description and a query.

    The language of the product descriptions and queries are determined by the `country_code` argument, which is used
    to get the appropriate SpaCy NLP object and stop words for that language. The `lemmatization` argument determines
    whether the tokenized words should be lemmatized or not.

    The `intialize_bm25` method must be called with the product descriptions and product IDs before the model can be used
    to rank products. The model can be saved to disk using the `save_bm25` method, and loaded again later using the
    `load_bm25` method. The `score` method is used to get the ranking score for a single product given a query.

    Attributes
    ----------

        - corpus (pandas.Series): The product descriptions.
        - corpus_tokenized (pandas.Series): The tokenized product descriptions.
        - country_code (str): The country code indicating the language of the product descriptions and queries.
        - lemmatization (bool): Whether the tokenized words should be lemmatized or not.
        - bm25 (rank_bm25.BM25Okapi): The BM25 ranking model.
        - spacy_nlp(spacy.language.Language): The SpaCy NLP object for the specified language.
        - stop_words (set): The set of stop words for the specified language.
        - tokenizer (spacy.tokenizer.Tokenizer): The SpaCy tokenizer for the specified language.
        - product_ids (pandas.Series): The product IDs corresponding to the product descriptions.
    """

    def __init__(self, country_code: str = "us", lemmatization: bool = True):
        """
        Initialize the Bm25Model.

        Attributes
        ----------
        country_code: str, optional (default="us")
            The country code of the language used for the model.
        lemmatization: bool, optional (default=True)
            Whether to perform lemmatization on the tokenized words.

        Returns
        -------
        None
        """

        self.corpus: Optional[S] = None
        self.corpus_tokenized: Optional[S] = None
        self.product_ids: Optional[S] = None
        self.country_code: str = country_code
        self.lemmatization: bool = lemmatization

        self.bm25: Optional[BM25Okapi] = None
        self.spacy_nlp, self.stop_words = get_spacy_from_country(country_code)
        self.tokenizer: Tokenizer = Tokenizer(self.spacy_nlp.vocab)

    def tokenization(self, text: str):
        """
        Tokenize a given text.

        Attributes
        ----------
        text: str
            The input text to be tokenized.

        Returns
        -------
        list
            A list of tokenized words.
        """
        # tokenize
        tokens = self.tokenizer(text)

        # Remove stop words
        tokens = [token for token in tokens if token.text not in self.stop_words]

        if self.lemmatization:
            tokens = [token.lemma_ for token in tokens]
        else:
            tokens = [token.text for token in tokens]

        return tokens

    def intialize_bm25(self, corpus: S, product_ids: S):
        """
        Initialize the BM25Okapi model with the given corpus.

        Attributes
        ----------
        corpus: pandas.Series
            The corpus used to initialize the BM25Okapi model.
        product_ids: pandas.Series
            The product ids associated with the given corpus.

        Returns
        -------
        None
        """
        self.corpus = corpus
        self.product_ids = product_ids
        self.corpus_tokenized = self.corpus.apply(self.tokenization)
        self.bm25 = BM25Okapi(self.corpus_tokenized)

    def score(self, query: str, product_id: str):
        """
        Get the BM25 score for a given query and product id.

        Attributes
        ----------
        query: str
            The input query.
        product_id: str
            The product id associated with the corpus.

        Returns
        -------
        float
            The BM25 score for the given query and product id.
        """
        index = (
            self.product_ids[self.product_ids == product_id].index[0]
            if product_id in self.product_ids.values
            else None
        )
        tokenized_query = self.tokenization(query)
        scores = (
            self.bm25.get_batch_scores(tokenized_query, [index])
            if index is not None
            else [0]
        )

        return scores[0]
