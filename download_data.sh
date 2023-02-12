#!/bin/bash

mkdir -p data
aicrowd login
aicrowd dataset download -c esci-challenge-for-improving-product-search -o data 0 1 2 3