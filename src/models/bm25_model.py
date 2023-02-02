from typing import List, Optional
import re

from pandas import Series as S
from rank_bm25 import BM25Okapi
import spacy
from spacy.tokenizer import Tokenizer

from src.utils.support import get_spacy_from_country


class Bm25Model:
    def __init__(self, country: str = "english", lemmatization: bool = True):

        self.corpus = None
        self.corpus_tokenized = None
        self.country = country
        self.lemmatization = lemmatization

        self.bm25: Optional[BM25Okapi] = None
        self.spacy_nlp, self.stop_words = get_spacy_from_country(country)
        self.tokenizer = Tokenizer(self.spacy_nlp.vocab)

    def tokenization(self, text: str):

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

        self.corpus = corpus
        self.product_ids = product_ids
        self.corpus_tokenized = self.corpus.apply(self.tokenization)
        self.bm25 = BM25Okapi(self.corpus_tokenized)

    def score(self, query: str, product_id: str):

        index = (
            self.product_ids[self.product_ids == product_id].index[0]
            if product_id
            else None
        )
        tokenized_query = self.tokenization(query)
        scores = (
            self.bm25.get_batch_scores(tokenized_query, [index])
            if index is not None
            else [0]
        )

        return scores[0]
