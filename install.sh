#!/bin/bash

#install requirements.txt
pip3 install -r requirements.txt

echo "project dependencies has been downloaded successfully"

# Download English model
python3 -m spacy download en_core_web_sm

# Download Spanish model
python3 -m spacy download es_core_news_sm

# Download Japanese model
python3 -m spacy download ja_core_news_sm

echo "spacy models have been successfully downloaded"