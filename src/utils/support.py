import os
import csv
from typing import Any

import pandas as pd
from pandas import DataFrame as D
import spacy


def get_spacy_from_country(country: str) -> Any:

    if country == "us":
        spacy_model = spacy.load("en_core_web_sm")
        spacy_stop_words = spacy_model.Defaults.stop_words

    if country == "es":
        spacy_model = spacy.load("es_core_news_sm")
        spacy_stop_words = spacy_model.Defaults.stop_words

    if country == "jp":
        spacy_model = spacy.load("ja_core_news_sm")
        spacy_stop_words = spacy_model.Defaults.stop_words

    return spacy_model, spacy_stop_words
