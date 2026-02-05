#!/bin/bash
set -e

# Configuration
CHUNK_WORDS=400
OVERLAP_WORDS=40
PROMPT_RATIO=0.1

# Create SFT datasets for all authors

poetry run python create_data_sft.py \
  --input-dir data_books/darwin \
  --output data_sft/darwin.json \
  --chunk-words $CHUNK_WORDS \
  --overlap-words $OVERLAP_WORDS \
  --prompt-ratio $PROMPT_RATIO

poetry run python create_data_sft.py \
  --input-dir data_books/dickens \
  --output data_sft/dickens.json \
  --chunk-words $CHUNK_WORDS \
  --overlap-words $OVERLAP_WORDS \
  --prompt-ratio $PROMPT_RATIO

poetry run python create_data_sft.py \
  --input-dir data_books/dostoevsky \
  --output data_sft/dostoevsky.json \
  --chunk-words $CHUNK_WORDS \
  --overlap-words $OVERLAP_WORDS \
  --prompt-ratio $PROMPT_RATIO

poetry run python create_data_sft.py \
  --input-dir data_books/doyle \
  --output data_sft/doyle.json \
  --chunk-words $CHUNK_WORDS \
  --overlap-words $OVERLAP_WORDS \
  --prompt-ratio $PROMPT_RATIO

poetry run python create_data_sft.py \
  --input-dir data_books/fitzgerald \
  --output data_sft/fitzgerald.json \
  --chunk-words $CHUNK_WORDS \
  --overlap-words $OVERLAP_WORDS \
  --prompt-ratio $PROMPT_RATIO

poetry run python create_data_sft.py \
  --input-dir data_books/twain \
  --output data_sft/twain.json \
  --chunk-words $CHUNK_WORDS \
  --overlap-words $OVERLAP_WORDS \
  --prompt-ratio $PROMPT_RATIO
