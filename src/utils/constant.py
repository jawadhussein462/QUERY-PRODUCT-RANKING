from enum import Enum


class CrossEncoderModelName(Enum):
    Bert = "bert"
    Distilbert = "distilbert"
    Albert = "albert"


class CountryCode(Enum):
    US = "us"
    Spanish = "es"
    Japanese = "jp"
