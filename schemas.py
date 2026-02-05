from pathlib import Path

from pydantic import BaseModel


class SFTExample(BaseModel):
    """A single example for SFT training."""

    id: str  # Unique identifier
    path: Path  # Path to the original book
    prompt: str  # Seed/context text
    completion: str  # Continuation text
    created_at: str  # UTC timestamp in YYYYMMDDTHHMMSSZ format


class DPOExample(BaseModel):
    """A single example for DPO training."""

    path: Path  # Path to the original book
    prompt: str  # Seed/context text
    chosen: str  # Real continuation from the book
    rejected: str  # Generated continuation from base model
