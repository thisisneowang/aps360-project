"""Train a CNN+LSTM image-to-LaTeX baseline with greedy decoding.

Example:
    python train_baseline.py --epochs 3 --batch-size 16 --lr 1e-3
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Set-up the current file directory paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline_model import BaselineIm2LatexModel, count_trainable_parameters
from preprocessing import FormulaImageTransform
from tokenization.dataset import Im2LatexDataset, make_autoregressive_collate_fn
from tokenization.tokenizer import LatexTokenizer
from tokenization.vocab import Vocab

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "im2latex"
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "data" / "processed" / "vocab.json"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "checkpoints"

LABEL_PAD_ID = -100


def initialize_metrics_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])


def append_metrics_row(
    path: Path,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
        ])

# For reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# GPU training is the best
def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Just loop through and check whether the output tokens match
def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    pred = logits.argmax(dim=-1)
    mask = labels.ne(LABEL_PAD_ID)
    correct = ((pred == labels) & mask).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)

# Handle special tokens as well. Otherwise, just translate normally.
def ids_to_latex_text(ids: list[int], vocab: Vocab, tokenizer: LatexTokenizer, eos_id: int) -> str:
    tokens: list[str] = []
    for idx in ids:
        token = vocab.id_to_token(idx)
        if token == "<eos>" or idx == eos_id:
            break
        if token == "<pad>" or token == "<bos>":
            continue
        tokens.append(token)
    return tokenizer.detokenize(tokens)


def train_one_epoch(
    model: BaselineIm2LatexModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    grad_clip: float,
    print_every: int,
    max_batches: int | None,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    steps = 0

    for batch_idx, batch in enumerate(loader):
        # If somehow we've exceeded max_batches, exit early
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["images"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(images, decoder_input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=LABEL_PAD_ID,
        )

        # Setting to none is actually a (minor) optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        correct, total = token_accuracy(logits.detach(), labels)
        total_correct += correct
        total_tokens += total
        steps += 1

        # Print loss and token accuracy updates at fixed batch intervals
        if print_every > 0 and (batch_idx + 1) % print_every == 0:
            running_loss = total_loss / max(1, steps)
            running_acc = total_correct / max(1, total_tokens)
            print(
                f"  [train] step={batch_idx + 1} "
                f"loss={running_loss:.4f} token_acc={running_acc:.4f}"
            )

    avg_loss = total_loss / max(1, steps)
    avg_acc = total_correct / max(1, total_tokens)
    return avg_loss, avg_acc

# Samething as above but for validation. Of course, use no_grad since weights aren't updated here.
@torch.no_grad()
def evaluate(
    model: BaselineIm2LatexModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    steps = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["images"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(images, decoder_input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=LABEL_PAD_ID,
        )

        total_loss += loss.item()
        correct, total = token_accuracy(logits, labels)
        total_correct += correct
        total_tokens += total
        steps += 1

    avg_loss = total_loss / max(1, steps)
    avg_acc = total_correct / max(1, total_tokens)
    return avg_loss, avg_acc

# (If desired) print the output of the model against the ground-truth tokens at the end of an epoch
@torch.no_grad()
def preview_predictions(
    model: BaselineIm2LatexModel,
    loader: DataLoader,
    device: torch.device,
    vocab: Vocab,
    tokenizer: LatexTokenizer,
    max_decode_len: int,
    num_samples: int,
) -> None:
    model.eval()

    try:
        batch = next(iter(loader))
    except StopIteration:
        print("  [preview] skipped (empty loader)")
        return

    images = batch["images"].to(device)
    labels = batch["labels"]
    image_names = batch["image_names"]

    decode_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    generated = decode_model.generate(images[:num_samples], max_len=max_decode_len).cpu()

    print("  [preview] greedy-decoded samples")
    for i in range(min(num_samples, generated.size(0))):
        pred_ids = generated[i].tolist()

        # Labels are shifted targets; remove pad marker (-100) for readable target text.
        target_ids = [idx for idx in labels[i].tolist() if idx != LABEL_PAD_ID]

        pred_text = ids_to_latex_text(pred_ids, vocab=vocab, tokenizer=tokenizer, eos_id=vocab.eos_id)
        target_text = ids_to_latex_text(target_ids, vocab=vocab, tokenizer=tokenizer, eos_id=vocab.eos_id)

        print(f"    - image: {image_names[i]}")
        print(f"      pred  : {pred_text[:220]}")
        print(f"      target: {target_text[:220]}")

# Add a TON of options! These can be enabled from a Powershell terminal like in VSCode
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CNN+LSTM baseline for image-to-LaTeX.")

    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Path for per-epoch metrics CSV (default: <save_dir>/baseline_metrics.csv).",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--encoder-base-channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=100)

    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-decode-len", type=int, default=128)
    parser.add_argument("--preview-samples", type=int, default=2)

    parser.add_argument(
        "--normalize-images",
        action="store_true",
        help="Use mean/std normalization in image preprocessing.",
    )

    return parser

# Now pute verything together (along with the appropriate datasets)
def main() -> None:
    args = build_arg_parser().parse_args()
    metrics_file = args.metrics_file if args.metrics_file is not None else args.save_dir / "baseline_metrics.csv"

    if not args.vocab.exists():
        raise FileNotFoundError(f"Vocab file not found: {args.vocab}")

    set_seed(args.seed)
    device = pick_device(args.device)

    vocab = Vocab.load(args.vocab)
    tokenizer = LatexTokenizer()
    image_transform = FormulaImageTransform(normalize=args.normalize_images)

    train_dataset = Im2LatexDataset(
        data_root=args.data_root,
        split="train",
        tokenizer=tokenizer,
        vocab=vocab,
        image_transform=image_transform,
        apply_latex_cleaning=True,
        add_bos=True,
        add_eos=True,
    )
    val_dataset = Im2LatexDataset(
        data_root=args.data_root,
        split="val",
        tokenizer=tokenizer,
        vocab=vocab,
        image_transform=image_transform,
        apply_latex_cleaning=True,
        add_bos=True,
        add_eos=True,
    )

    collate_fn = make_autoregressive_collate_fn(pad_id=vocab.pad_id, label_pad_id=LABEL_PAD_ID)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    model = BaselineIm2LatexModel(
        vocab_size=len(vocab),
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        in_channels=1,
        encoder_base_channels=args.encoder_base_channels,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
    ).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 88)
    print("BASELINE TRAINING")
    print("=" * 88)
    print(f"device            : {device}")
    print(f"train_size        : {len(train_dataset)}")
    print(f"val_size          : {len(val_dataset)}")
    print(f"vocab_size        : {len(vocab)}")
    print(f"trainable_params  : {count_trainable_parameters(model):,}")
    print(f"batch_size        : {args.batch_size}")
    print(f"epochs            : {args.epochs}")
    print(f"metrics_file      : {metrics_file}")
    print()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    initialize_metrics_file(metrics_file)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Measure the time to train every epoch
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            print_every=args.print_every,
            max_batches=args.max_train_batches,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=args.max_val_batches,
        )

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_token_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_token_acc={val_acc:.4f} | "
            f"time={elapsed:.1f}s"
        )
        append_metrics_row(
            path=metrics_file,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        preview_predictions(
            model=model,
            loader=val_loader,
            device=device,
            vocab=vocab,
            tokenizer=tokenizer,
            max_decode_len=args.max_decode_len,
            num_samples=args.preview_samples,
        )

        model_for_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        # Save the last model whenever possible
        ckpt_last = args.save_dir / "baseline_last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_for_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "vocab_size": len(vocab),
            },
            ckpt_last,
        )
        # Always update the best model saved so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_best = args.save_dir / "baseline_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_for_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "vocab_size": len(vocab),
                    "best_val_loss": best_val_loss,
                },
                ckpt_best,
            )
            print(f"  saved new best checkpoint -> {ckpt_best}")

        print(f"  saved last checkpoint     -> {ckpt_last}")
        print()

    print("Training complete.")


if __name__ == "__main__":
    main()

