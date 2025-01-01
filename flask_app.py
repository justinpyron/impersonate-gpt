import socket

import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_CARD = "openai-community/gpt2"
WEIGHTS_DARWIN = "model_darwin_20250101T0035.pt"
WEIGHTS_DOSTOEVSKY = "model_dostoevsky_20250101T0202.pt"
WEIGHTS_FITZGERALD = "model_fitzgerald_20241231T2306.pt"
WEIGHTS_TWAIN = "model_twain_20241231T2357.pt"


app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD)


def load_model(weights_path: str) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        MODEL_CARD,
        state_dict=torch.load(weights_path, weights_only=True),
    )


def generate(
    model: torch.nn.Module,
    seed_text: str,
    temperature: float,
    num_tokens: int,
) -> str:
    out = model.generate(
        **tokenizer(seed_text, return_tensors="pt"),
        max_new_tokens=num_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
    )
    return tokenizer.decode(out[0].tolist())


@app.route("/health")
def health_check():
    return jsonify({"status": "Server is running."})


@app.route("/darwin", methods=["POST"])
def darwin() -> str:
    data = request.get_json()
    model = load_model(WEIGHTS_DARWIN)
    out = generate(model, data["text"], data["temperature"], data["num_tokens"])
    return out


@app.route("/dostoevsky", methods=["POST"])
def dostoevsky() -> str:
    data = request.get_json()
    model = load_model(WEIGHTS_DOSTOEVSKY)
    out = generate(model, data["text"], data["temperature"], data["num_tokens"])
    return out


@app.route("/fitzgerald", methods=["POST"])
def fitzgerald() -> str:
    data = request.get_json()
    model = load_model(WEIGHTS_FITZGERALD)
    out = generate(model, data["text"], data["temperature"], data["num_tokens"])
    return out


@app.route("/twain", methods=["POST"])
def twain() -> str:
    data = request.get_json()
    model = load_model(WEIGHTS_TWAIN)
    out = generate(model, data["text"], data["temperature"], data["num_tokens"])
    return out


if __name__ == "__main__":
    app.run(debug="local" in socket.gethostname(), host="0.0.0.0", port=8080)
