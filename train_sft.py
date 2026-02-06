"""
SFT training with LoRA on Modal.

Launches a fire-and-forget training job that runs in the cloud. The job
continues even if you disconnect. Logs are sent to Weights & Biases.

Usage:
    modal run train_sft.py \
        --model-path <path-in-volume> \
        --data-path-train <path-in-volume> \
        --data-path-val <path-in-volume> \
        --output-dir <path-in-volume> \
        [--lora-r 16] \
        [--lora-alpha 32] \
        [--learning-rate 0.0002] \
        [--num-epochs 3] \
        [--batch-size 4] \
        [--max-seq-length 2048] \
        [--gradient-accumulation-steps 4]
"""

import json

import modal

# =============================================================================
# Configuration
# =============================================================================

APP_NAME = "impersonate-gpt-sft"
VOLUME_NAME = "impersonate-gpt"
VOLUME_MOUNT_PATH = "/data"
GPU = "A10"
WANDB_ENTITY = "pyron"
WANDB_PROJECT = "impersonate-gpt-sft"

# =============================================================================
# Modal Setup
# =============================================================================

app = modal.App(APP_NAME)

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "wandb",
)

volume = modal.Volume.from_name(VOLUME_NAME)


# =============================================================================
# Training
# =============================================================================


def load_sft_dataset(path: str):
    """Load a JSON file of SFTExamples into a HuggingFace Dataset."""
    from datasets import Dataset

    with open(path, "r") as f:
        data = json.load(f)

    return Dataset.from_dict(
        {
            "prompt": [ex["prompt"] for ex in data],
            "completion": [ex["completion"] for ex in data],
        }
    )


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    gpu=GPU,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    model_path: str,
    data_path_train: str,
    data_path_val: str,
    output_dir: str,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    max_seq_length: int,
    gradient_accumulation_steps: int,
):
    """Run SFT training with LoRA on a frozen base model."""
    import os

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Configure wandb
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    # Data
    dataset_train = load_sft_dataset(f"{VOLUME_MOUNT_PATH}/{data_path_train}")
    dataset_val = load_sft_dataset(f"{VOLUME_MOUNT_PATH}/{data_path_val}")

    # Model and tokenizer
    model_path_full = f"{VOLUME_MOUNT_PATH}/{model_path}"
    tokenizer = AutoTokenizer.from_pretrained(model_path_full)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path_full)

    # LoRA
    config_lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config
    output_dir_full = f"{VOLUME_MOUNT_PATH}/{output_dir}"
    config_training = SFTConfig(
        output_dir=output_dir_full,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=config_training,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        processing_class=tokenizer,
        peft_config=config_lora,
    )
    trainer.train()
    trainer.save_model(output_dir_full)
    volume.commit()


# =============================================================================
# CLI Entrypoint
# =============================================================================


@app.local_entrypoint()
def main(
    model_path: str,
    data_path_train: str,
    data_path_val: str,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
    max_seq_length: int = 2048,
    gradient_accumulation_steps: int = 4,
):
    """Launch SFT training on Modal. All paths are relative to the volume root."""
    print("=" * 80)
    print("Launching SFT training job on Modal...")
    print(f"  Model: {model_path}")
    print(f"  Train data: {data_path_train}")
    print(f"  Val data: {data_path_val}")
    print(f"  Output: {output_dir}")
    print(f"  LoRA config: r={lora_r}, alpha={lora_alpha}")
    print(
        f"  Training: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}"
    )
    print("-" * 80)

    call = train.spawn(
        model_path=model_path,
        data_path_train=data_path_train,
        data_path_val=data_path_val,
        output_dir=output_dir,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    print()
    print(f"✓ Training job launched successfully!")
    print(f"  Job ID: {call.object_id}")
    print()
    print("You can safely close this terminal. The job will continue running on Modal.")
    print()
    print("To check job status:")
    print(f"  • Web UI: https://modal.com/apps")
    print(f"  • CLI: modal app logs {app.name}")
    print("=" * 80)
