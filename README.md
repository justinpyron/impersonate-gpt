# impersonate-gpt
Fine-tuned GPT2 that emulates famous writers.

# Project Organization
```
├── README.md                   <- Overview
├── app.py                      <- Streamlit web app frontend
├── model_*.pt                  <- Weights of fine-tuned models
├── flask_app.py                <- Flask app for serving inferences
├── Dockerfile                  <- Docker image for Flask app
├── impersonate_dataset.py      <- Custom PyTorch Dataset subclass for book data
├── impersonate_trainer.py      <- Class for training models
├── launch_train_session.py     <- Script to run training sessions
├── train_configs.py            <- Configs for the different models (one per author)
├── pyproject.toml              <- Poetry config specifying Python environment dependencies
├── poetry.lock                 <- Locked dependencies to ensure consistent installs
├── .pre-commit-config.yaml     <- Linting configs
```

# Installation
This project uses [Poetry](https://python-poetry.org/docs/) to manage its Python environment.

1. Install Poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies
```
poetry install
```

# Usage
A Streamlit web app is the frontend for interacting with the fine-tuned models.

The app can be accessed at [impersonate-gpt.streamlit.app](https://impersonate-gpt.streamlit.app).

Alternatively, the app can be run locally with
```
poetry run streamlit run app.py
```
