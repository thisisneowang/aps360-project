"""Build and save a training vocabulary for the IM2LATEX dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import build_token_sequences_for_vocab
from tokenizer import LatexTokenizer
from vocab import Vocab

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "im2latex"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "vocab.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LaTeX vocabulary from IM2LATEX train split.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to directory containing im2latex_*.lst files and formula_images/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write vocab JSON.",
    )
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum token frequency to include.")
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Optional maximum total vocab size (including special tokens).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "validate", "test"],
        help="Dataset split used for vocab construction (recommended: train).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = LatexTokenizer()
    token_sequences = build_token_sequences_for_vocab(
        data_root=args.data_root,
        tokenizer=tokenizer,
        split=args.split,
        apply_latex_cleaning=True,
    )

    vocab = Vocab.build(
        token_sequences=token_sequences,
        min_freq=args.min_freq,
        max_size=args.max_size,
    )
    vocab.save(args.output)

    lengths = [len(seq) for seq in token_sequences]
    total_tokens = sum(lengths)
    avg_len = total_tokens / max(1, len(lengths))
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    print(f"Built vocab with {len(vocab)} entries.")
    print(f"Saved to: {args.output}")
    print(f"Sequences used: {len(token_sequences)}")
    print(f"Minimum tokenized formula length: {min_len}")
    print(f"Maximum tokenized formula length: {max_len}")
    print(f"Average tokenized formula length: {avg_len:.2f}")
    print("First 20 vocab tokens:")
    print(vocab.itos[:20])


if __name__ == "__main__":
    main()

