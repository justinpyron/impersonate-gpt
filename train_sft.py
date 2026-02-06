"""
SFT training with LoRA on Modal.

Usage:
    modal run train_sft.py \
        --model-path <path-in-volume> \
        --train-data <path-in-volume> \
        --val-data <path-in-volume> \
        --output-dir <path-in-volume>
"""

import json

import modal

# =============================================================================
# Configuration
# =============================================================================

VOLUME_NAME = "PLACEHOLDER"
VOLUME_MOUNT_PATH = "/vol"
GPU = "PLACEHOLDER"  # e.g. modal.gpu.A100(count=1)
TIMEOUT_SECONDS = 3600

# =============================================================================
# Modal Setup
# =============================================================================

app = modal.App("PLACEHOLDER")

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
