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


def load_model(weights_path: str = None) -> torch.nn.Module:
    if weights_path is None:
        state_dict = None
    else:
        state_dict = torch.load(weights_path, weights_only=True)
    return AutoModelForCausalLM.from_pretrained(MODEL_CARD, state_dict=state_dict)


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


def serve(weights_path: str = None) -> str:
    data = request.get_json()
    model = load_model(weights_path)
    out = generate(model, data["text"], data["temperature"], data["num_tokens"])
    return out


@app.route("/health")
def health_check():
    return jsonify({"status": "Server is running."})


@app.route("/gpt2", methods=["POST"])
def gpt2() -> str:
    return jsonify({"generated_text": serve()})


@app.route("/darwin", methods=["POST"])
def darwin() -> str:
    return jsonify({"generated_text": serve(WEIGHTS_DARWIN)})


@app.route("/dostoevsky", methods=["POST"])
def dostoevsky() -> str:
    return jsonify({"generated_text": serve(WEIGHTS_DOSTOEVSKY)})


@app.route("/fitzgerald", methods=["POST"])
def fitzgerald() -> str:
    return jsonify({"generated_text": serve(WEIGHTS_FITZGERALD)})


@app.route("/twain", methods=["POST"])
def twain() -> str:
    return jsonify({"generated_text": serve(WEIGHTS_TWAIN)})


if __name__ == "__main__":
    app.run(debug="local" in socket.gethostname(), host="0.0.0.0", port=8080)
