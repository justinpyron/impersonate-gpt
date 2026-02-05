"""
Script to create SFT training data from raw book files.

Transforms raw book text files into SFTExample objects by:
1. Stripping Project Gutenberg header/footer
2. Chunking text into segments
3. Splitting each chunk into prompt/completion pairs
"""

import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from banned_words import banned_words
from schemas import SFTExample


def strip_gutenberg(text: str) -> str:
    """
    Remove Project Gutenberg header and footer from text.

    Args:
        text: Raw text containing Gutenberg boilerplate

    Returns:
        Clean text with header/footer removed

    Raises:
        ValueError: If START or END markers are not found
    """
    start_pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*"
    end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .+? \*\*\*"

    start_match = re.search(start_pattern, text)
    end_match = re.search(end_pattern, text)

    if not start_match:
        raise ValueError("Could not find Gutenberg START marker")
    if not end_match:
        raise ValueError("Could not find Gutenberg END marker")

    # Extract text between markers
    content = text[start_match.end() : end_match.start()]
    return content.strip()


def chunk_text(text: str, chunk_words: int, overlap_words: int = 0) -> list[str]:
    """
    Split text into chunks of approximately chunk_words size.

    Args:
        text: Clean text to chunk
        chunk_words: Target number of words per chunk
        overlap_words: Number of words to overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []

    if len(words) == 0:
        return chunks

    stride = chunk_words - overlap_words
    if stride <= 0:
        raise ValueError("overlap_words must be less than chunk_words")

    i = 0
    while i < len(words):
        # Get chunk_words starting from position i
        chunk_end = min(i + chunk_words, len(words))
        chunk_words_list = words[i:chunk_end]

        # Try to extend to a sentence boundary (., !, ?)
        # Look ahead up to 20 words for a sentence ending
        if chunk_end < len(words):
            for j in range(chunk_end, min(chunk_end + 20, len(words))):
                word = words[j]
                chunk_words_list.append(word)
                if word.endswith((".", "!", "?", '."', '!"', '?"', ".'", "!'", "?'")):
                    break

        chunk = " ".join(chunk_words_list)
        chunks.append(chunk)

        i += stride

        # Stop if we've consumed all words
        if chunk_end >= len(words):
            break

    return chunks


def split_chunk(chunk: str, prompt_ratio: float) -> tuple[str, str]:
    """
    Split a chunk into prompt and completion at approximately prompt_ratio.

    Args:
        chunk: Text chunk to split
        prompt_ratio: Fraction of chunk that becomes prompt (0.0 to 1.0)

    Returns:
        Tuple of (prompt, completion)
    """
    if not 0.0 < prompt_ratio < 1.0:
        raise ValueError("prompt_ratio must be between 0 and 1 (exclusive)")

    words = chunk.split()
    split_index = int(len(words) * prompt_ratio)

    # Ensure we have at least one word on each side
    split_index = max(1, min(split_index, len(words) - 1))

    prompt = " ".join(words[:split_index])
    completion = " " + " ".join(words[split_index:])

    return prompt, completion


def map_book_to_sft_examples(
    path: Path,
    chunk_words: int,
    overlap_words: int,
    prompt_ratio: float,
    created_at: str,
    min_chunk_words: int = 20,
) -> list[SFTExample]:
    """
    Process a single book file into a list of SFTExample objects.

    Args:
        path: Path to the book file
        chunk_words: Target words per chunk
        overlap_words: Words of overlap between chunks
        prompt_ratio: Fraction of chunk that becomes prompt
        created_at: UTC timestamp in ISO format
        min_chunk_words: Minimum words required for a valid chunk

    Returns:
        List of SFTExample objects
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    clean_text = strip_gutenberg(text)
    chunks = chunk_text(clean_text, chunk_words, overlap_words)

    examples = []
    for chunk in chunks:
        # Skip chunks that are too short
        if len(chunk.split()) < min_chunk_words:
            continue

        prompt, completion = split_chunk(chunk, prompt_ratio)

        # Skip if completion contains banned words (case-insensitive)
        completion_lower = completion.lower()
        if any(banned_word in completion_lower for banned_word in banned_words):
            continue

        example = SFTExample(
            id=str(uuid.uuid4())[:4],
            path=path,
            prompt=prompt,
            completion=completion,
            created_at=created_at,
        )
        examples.append(example)

    return examples


def create_sft_dataset(
    paths: list[Path],
    chunk_words: int,
    overlap_words: int,
    prompt_ratio: float,
) -> list[SFTExample]:
    """
    Process multiple book files into a combined list of SFTExample objects.

    Args:
        paths: List of paths to book files
        chunk_words: Target words per chunk
        overlap_words: Words of overlap between chunks
        prompt_ratio: Fraction of chunk that becomes prompt

    Returns:
        Combined list of SFTExample objects from all books
    """
    all_examples = []
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for path in paths:
        print(f"Processing: {path.name}")
        try:
            examples = map_book_to_sft_examples(
                path, chunk_words, overlap_words, prompt_ratio, created_at
            )
            all_examples.extend(examples)
            print(f"  -> {len(examples)} examples")
        except ValueError as e:
            print(f"  -> Error: {e}")

    return all_examples


def main():
    parser = argparse.ArgumentParser(
        description="Create SFT training data from raw book files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw book .txt files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=500,
        help="Target words per chunk (default: 500)",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=0,
        help="Words of overlap between chunks (default: 0)",
    )
    parser.add_argument(
        "--prompt-ratio",
        type=float,
        default=0.5,
        help="Fraction of chunk that becomes prompt (default: 0.5)",
    )

    args = parser.parse_args()

    # 1. Check if output file already exists
    if args.output.exists():
        raise FileExistsError(f"Output file already exists: {args.output}")

    # 2. Find all .txt files in input directory
    paths = sorted(args.input_dir.glob("*.txt"))
    if not paths:
        print(f"No .txt files found in {args.input_dir}")
        return

    # 3. Print parameters
    print(f"Found {len(paths)} book files")
    print(
        "Parameters:",
        f"\n  chunk_words   = {args.chunk_words}",
        f"\n  overlap_words = {args.overlap_words}",
        f"\n  prompt_ratio  = {args.prompt_ratio}",
    )
    print()

    # 4. Create dataset
    examples = create_sft_dataset(
        paths=paths,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        prompt_ratio=args.prompt_ratio,
    )

    print()
    print(f"Total examples: {len(examples)}")

    # 5. Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 6. Write to JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([ex.model_dump(mode="json") for ex in examples], f, indent=4)

    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
