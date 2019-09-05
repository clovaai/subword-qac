#!/usr/bin/env bash

INPUT="data/aol/full/train.query.txt"
SPM_DIR="spm"
MAX_NUM_SENTS=100000000

mkdir -p ${SPM_DIR}/char
spm_train --input=$INPUT \
    --input_sentence_size ${MAX_NUM_SENTS} \
    --model_type char \
    --model_prefix ${SPM_DIR}/char/spm

VOCAB_SIZE=256

for MODEL_TYPE in "unigram" "bpe" ; do
    mkdir -p ${SPM_DIR}/${MODEL_TYPE}/${VOCAB_SIZE}
    spm_train --input=$INPUT \
        --input_sentence_size ${MAX_NUM_SENTS} \
        --model_type ${MODEL_TYPE} \
        --vocab_size ${VOCAB_SIZE} \
        --model_prefix ${SPM_DIR}/${MODEL_TYPE}/${VOCAB_SIZE}/spm
done
