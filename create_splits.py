"""
Script to split examples into train/val sets, stratified by book.

Takes a JSON file of examples (SFTExample or DPOExample) and splits them
into train and val sets, ensuring each book contributes to both sets.

Output files are created in the same directory as the input file with
_train.json and _val.json suffixes.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def split_examples(
    examples: list[dict],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Split examples into train and val sets, stratified by book.

    Args:
        examples: List of example dicts (must have 'path' field)
        train_ratio: Fraction of examples for training (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_examples, val_examples)
    """
    # Group examples by book path
    by_book: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_book[ex["path"]].append(ex)

    # Create seeded random instance
    rng = random.Random(seed)

    train_examples = []
    val_examples = []

    # For each book, shuffle and split
    for path, book_examples in sorted(by_book.items()):
        rng.shuffle(book_examples)
        split_idx = int(len(book_examples) * train_ratio)
        # Ensure at least 1 example in each split if possible
        if split_idx == 0 and len(book_examples) > 1:
            split_idx = 1
        elif split_idx == len(book_examples) and len(book_examples) > 1:
            split_idx = len(book_examples) - 1

        train_examples.extend(book_examples[:split_idx])
        val_examples.extend(book_examples[split_idx:])

    return train_examples, val_examples


def main():
    parser = argparse.ArgumentParser(
        description="Split examples into train/val sets, stratified by book."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of examples for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Load input JSON
    print(f"Loading: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        examples = json.load(f)

    print(f"Total examples: {len(examples)}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Seed: {args.seed}")
    print()

    # Count examples per book
    by_book: dict[str, int] = defaultdict(int)
    for ex in examples:
        by_book[ex["path"]] += 1

    print(f"Books: {len(by_book)}")
    for path, count in sorted(by_book.items()):
        print(f"  {Path(path).name}: {count} examples")
    print()

    # Split examples
    train_examples, val_examples = split_examples(examples, args.train_ratio, args.seed)

    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples: {len(val_examples)}")

    # Create output paths in same directory as input
    input_dir = args.input.parent
    input_stem = args.input.stem  # filename without extension
    train_path = input_dir / f"{input_stem}_train.json"
    val_path = input_dir / f"{input_stem}_val.json"

    # Write output files
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_examples, f, indent=4)
    print(f"Saved: {train_path}")

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_examples, f, indent=4)
    print(f"Saved: {val_path}")


if __name__ == "__main__":
    main()
