# ImpersonateGPT

Text generation with LoRA adapters fine-tuned to mimic famous writers' styles.

## Overview

This project fine-tunes LoRA adapters on a base language model (Gemma-3-270M) to emulate the writing styles of classic authors. Training data comes from [Project Gutenberg](https://www.gutenberg.org) books.

## Architecture

- **Frontend:** Streamlit app deployed on Google Cloud Run
- **Backend:** FastAPI inference server on Modal
- **Training:** Supervised fine-tuning (SFT) with LoRA on Modal GPUs
- **CI/CD:** GitHub Actions for automated deployment

## Project Structure

```
├── app.py                      # Streamlit frontend
├── backend.py                  # Modal FastAPI inference server
├── train_sft.py                # Modal training script
├── create_data_sft.py          # Generate training data from books
├── create_splits.py            # Split data into train/val sets
├── data_books/                 # Raw Project Gutenberg texts
├── data_sft/                   # Datasets properly formatted for training
├── Dockerfile                  # Cloud Run container config
├── .github/workflows/          # CI/CD pipeline
└── pyproject.toml              # Poetry dependencies
```

## Setup

Install dependencies with [Poetry](https://python-poetry.org):

```bash
poetry install
```

## Data Preparation

1. **Create SFT datasets** from raw books:

```bash
./create_data_sft.sh  # Processes all authors in data_books/
```

Or for a single author:

```bash
poetry run python create_data_sft.py \
  --input-dir data_books/twain \
  --output data_sft/twain.json \
  --chunk-words 400 \
  --overlap-words 40 \
  --prompt-ratio 0.1
```

2. **Split into train/val sets:**

```bash
poetry run python create_splits.py \
  --input data_sft/twain.json \
  --train-ratio 0.9
```

## Training

Train a LoRA adapter on Modal:

```bash
modal run train_sft.py \
  --model-path gemma-3-270m \
  --data-path-train data_sft/twain_train.json \
  --data-path-val data_sft/twain_val.json \
  --name twain \
  --lora-r 16 \
  --lora-alpha 32 \
  --learning-rate 0.0002 \
  --num-epochs 1
```

Training logs are sent to Weights & Biases.

## Deployment

Deploy backend and frontend via GitHub Actions:

```bash
# Trigger workflow manually from GitHub UI
```

The workflow:
1. Deploys Modal backend (`backend.py`)
2. Builds Docker image for Streamlit app
3. Deploys to Google Cloud Run with `BACKEND_URL` env var

## Local Development

Run the app locally (requires `BACKEND_URL` env var):

```bash
export BACKEND_URL="https://your-modal-backend.modal.run"
poetry run streamlit run app.py
```
