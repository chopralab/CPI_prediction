#!/bin/bash

: ${DATASET:=human}
: ${radius:=2}  # >=0.
: ${ngram:=3}  # >=1.

python preprocess_test_data.py $DATASET $radius $ngram
