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


def load_model(weights_path: str = None) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        MODEL_CARD,
        state_dict=None
        if weights_path is None
        else torch.load(weights_path, weights_only=True),
    )


def generate(
    model: torch.nn.Module,
    seed_text: str,
    temperature: float,
    num_tokens: int,
) -> str:
    generated_tokens = model.generate(
        **tokenizer(seed_text, return_tensors="pt"),
        max_new_tokens=num_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
    )[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


def serve(model_name: str = None) -> str:
    model = model_dict.get(model_name)
    data = request.get_json()
    out = generate(model, data["text"], data["temperature"], data["num_tokens"])
    return out


@app.route("/health")
def health_check():
    return jsonify({"status": "Server is running."})


@app.route("/gpt2", methods=["POST"])
def gpt2():
    return jsonify({"generated_text": serve("gpt2")})


@app.route("/darwin", methods=["POST"])
def darwin():
    return jsonify({"generated_text": serve("darwin")})


@app.route("/dostoevsky", methods=["POST"])
def dostoevsky():
    return jsonify({"generated_text": serve("dostoevsky")})


@app.route("/fitzgerald", methods=["POST"])
def fitzgerald():
    return jsonify({"generated_text": serve("fitzgerald")})


@app.route("/twain", methods=["POST"])
def twain():
    return jsonify({"generated_text": serve("twain")})


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD)
    model_dict = {
        "gpt2": load_model(),
        "darwin": load_model(WEIGHTS_DARWIN),
        "dostoevsky": load_model(WEIGHTS_DOSTOEVSKY),
        "fitzgerald": load_model(WEIGHTS_FITZGERALD),
        "twain": load_model(WEIGHTS_TWAIN),
    }
    app.run(debug="local" in socket.gethostname(), host="0.0.0.0", port=8080)
