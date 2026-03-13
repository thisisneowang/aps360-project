"""Train ViT+Transformer image-to-LaTeX model.

Example:
    python train_transformer.py --epochs 15 --batch-size 16 --lr 3e-4 --device cuda
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import FormulaImageTransform
from tokenization.dataset import Im2LatexDataset, make_autoregressive_collate_fn
from tokenization.tokenizer import LatexTokenizer
from tokenization.vocab import Vocab
from transformer_model.vit_transformer_model import (
    VitTransformerIm2LatexModel,
    count_trainable_parameters,
)

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "im2latex"
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "data" / "processed" / "vocab.json"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "checkpoints"

LABEL_PAD_ID = -100

# Save the four horsemen of training curves
def initialize_metrics_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

# Write the four horsemen of training curves
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    pred = logits.argmax(dim=-1)
    mask = labels.ne(LABEL_PAD_ID)
    correct = ((pred == labels) & mask).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


# Convert id mappings (from vocabulary) back to their corresponding tokens
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
    model: VitTransformerIm2LatexModel | torch.nn.DataParallel,
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

    # Primary training loop
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["images"].to(device, non_blocking=True)
        decoder_input_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(images, decoder_input_ids)
        # CrossEntropyLoss is standard in autoregression generation against the entire vocabulary
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=LABEL_PAD_ID,
        ) 

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

        # Print in regular batch steps
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


@torch.no_grad()
def evaluate(
    model: VitTransformerIm2LatexModel | torch.nn.DataParallel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    steps = 0

    # Primary evaluation loop on the validation set.
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["images"].to(device, non_blocking=True)
        decoder_input_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

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


@torch.no_grad()
def preview_predictions(
    model: VitTransformerIm2LatexModel | torch.nn.DataParallel,
    loader: DataLoader,
    device: torch.device,
    vocab: Vocab,
    tokenizer: LatexTokenizer,
    max_decode_len: int,
    num_samples: int,
    decode_method: str,
    beam_size: int,
    beam_length_penalty: float,
) -> None:
    model.eval()

    try:
        batch = next(iter(loader))
    except StopIteration:
        print("  [preview] skipped (empty loader)")
        return

    images = batch["images"].to(device, non_blocking=True)
    labels = batch["labels"]
    image_names = batch["image_names"]

    decode_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    # Note: no teacher forcing here! This differs from evaluation and training.
    generated = decode_model.generate(
        images[:num_samples],
        max_len=max_decode_len,
        decode_method=decode_method,
        beam_size=beam_size,
        beam_length_penalty=beam_length_penalty,
    ).cpu()

    # Always use beam search for better accuracy (generally)
    if decode_method == "beam":
        print(f"  [preview] beam-decoded samples (B={beam_size})")
    else:
        print("  [preview] greedy-decoded samples")

    # Convert each token into their corresponding text, and then print it out
    for i in range(min(num_samples, generated.size(0))):
        pred_ids = generated[i].tolist()
        target_ids = [idx for idx in labels[i].tolist() if idx != LABEL_PAD_ID]

        pred_text = ids_to_latex_text(pred_ids, vocab=vocab, tokenizer=tokenizer, eos_id=vocab.eos_id)
        target_text = ids_to_latex_text(target_ids, vocab=vocab, tokenizer=tokenizer, eos_id=vocab.eos_id)

        print(f"    - image: {image_names[i]}")
        print(f"      pred  : {pred_text[:220]}")
        print(f"      target: {target_text[:220]}")

# Give the ability o stack a ton of argument options for the user (probably won't use all of them -- but good practice for later in career)
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ViT+Transformer for image-to-LaTeX.")

    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=8)

    # Internal dimension of transformer model
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--encoder-layers", type=int, default=10)
    parser.add_argument("--encoder-heads", type=int, default=6)
    # Tells you how large the MLP is in each encoder transformer block
    parser.add_argument("--encoder-mlp-ratio", type=float, default=4.0)

    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--decoder-heads", type=int, default=6)
    # Similarly tells you how large the MLP (or FFN -- same thing) is in each decoder transformer block
    parser.add_argument("--decoder-ffn-dim", type=int, default=1536)

    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=100)

    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    # A safeguard to protect against extremely large decoding/generation lengths, but likely shouldn't be completely necessary
    parser.add_argument("--max-decode-len", type=int, default=384)
    parser.add_argument("--max-generate-len", type=int, default=192)
    parser.add_argument("--preview-samples", type=int, default=2)

    parser.add_argument(
        "--decode-method",
        type=str,
        choices=["greedy", "beam"],
        default="beam",
        help="Decoding strategy for preview generation.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=20,
        help="Beam width used when --decode-method beam.",
    )

    # Don't overtly prefer shorter sequences: because they don't add more negative logprobs
    parser.add_argument(
        "--beam-length-penalty",
        type=float,
        default=0.0,
        help="Length normalization penalty for beam search (0 disables).",
    )

    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume training from a saved checkpoint.",
    )

    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Path for per-epoch metrics CSV (default: <save_dir>/vit_transformer_metrics.csv).",
    )

    parser.add_argument(
        "--normalize-images",
        action="store_true",
        help="Use mean/std normalization in image preprocessing.",
    )

    return parser

# Combine all the above functions into a single, cohesive unit. Again, this is relatively similar to what was done for the baseline model.
def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.vocab.exists():
        raise FileNotFoundError(f"Vocab file not found: {args.vocab}")

    if args.decode_method == "beam" and args.beam_size < 1:
        raise ValueError("beam_size must be >= 1")

    # Load a pre-defined metric file directory if provided
    metrics_file = args.metrics_file if args.metrics_file is not None else args.save_dir / "vit_transformer_metrics.csv"

    set_seed(args.seed)
    device = pick_device(args.device)


    # Load a vocab file if provided
    vocab = Vocab.load(args.vocab)
    tokenizer = LatexTokenizer()
    image_transform = FormulaImageTransform(
        out_h=args.image_height,
        out_w=args.image_width,
        normalize=args.normalize_images,
    )

    if args.max_decode_len < 2:
        raise ValueError("max_decode_len must be >= 2")

    # Cap the amount of formula tokens so that the length of decoder_input_ids never exceeds max_decode_len.
    max_formula_tokens = args.max_decode_len - 1

    # Load training and validation datasets
    train_dataset = Im2LatexDataset(
        data_root=args.data_root,
        split="train",
        tokenizer=tokenizer,
        vocab=vocab,
        image_transform=image_transform,
        apply_latex_cleaning=True,
        add_bos=True,
        add_eos=True,
        max_formula_tokens=max_formula_tokens,
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
        max_formula_tokens=max_formula_tokens,
    )

    # Pad sequences to uniform sequence length
    collate_fn = make_autoregressive_collate_fn(pad_id=vocab.pad_id, label_pad_id=LABEL_PAD_ID)

    pin_memory = device.type == "cuda"
    use_persistent_workers = args.num_workers > 0

    # Actually load the data
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
        collate_fn=collate_fn,
    )

    # Load the model with all the user input arguments
    model: VitTransformerIm2LatexModel | torch.nn.DataParallel
    model = VitTransformerIm2LatexModel(
        vocab_size=len(vocab),
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        pad_id=vocab.pad_id,
        in_channels=1,
        image_height=args.image_height,
        image_width=args.image_width,
        patch_size=args.patch_size,
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        encoder_mlp_ratio=args.encoder_mlp_ratio,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ffn_dim=args.decoder_ffn_dim,
        dropout=args.dropout,
        max_decode_len=args.max_decode_len,
    ).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")

        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model_for_load = model.module if isinstance(model, torch.nn.DataParallel) else model
        model_for_load.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"Resumed from: {args.resume_checkpoint}")
        print(f"  start_epoch      : {start_epoch}")
        print(f"  best_val_loss_so_far: {best_val_loss:.4f}")

    print("=" * 88)
    print("VIT+TRANSFORMER TRAINING")
    print("=" * 88)
    print(f"device            : {device}")
    print(f"train_size        : {len(train_dataset)}")
    print(f"val_size          : {len(val_dataset)}")
    print(f"vocab_size        : {len(vocab)}")
    print(f"trainable_params  : {count_trainable_parameters(model):,}")
    print(f"batch_size        : {args.batch_size}")
    print(f"epochs            : {args.epochs}")
    print(f"decoder_max_len   : {args.max_decode_len}")
    print(f"generate_max_len  : {args.max_generate_len}")
    print(f"max_formula_tokens: {max_formula_tokens}")
    print(f"decode_method     : {args.decode_method}")
    print(f"metrics_file      : {metrics_file}")
    if args.decode_method == "beam":
        print(f"beam_size         : {args.beam_size}")
        print(f"beam_len_penalty  : {args.beam_length_penalty}")
    print()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_checkpoint is None:
        initialize_metrics_file(metrics_file)
    elif not metrics_file.exists():
        initialize_metrics_file(metrics_file)

    if start_epoch > args.epochs:
        print(
            "Nothing to train: start_epoch is greater than epochs. "
            f"start_epoch={start_epoch}, epochs={args.epochs}."
        )
        return

    # Main training loop for each epoch
    for epoch in range(start_epoch, args.epochs + 1):
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
        append_metrics_row(
            path=metrics_file,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_token_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_token_acc={val_acc:.4f} | "
            f"time={elapsed:.1f}s"
        )

        preview_predictions(
            model=model,
            loader=val_loader,
            device=device,
            vocab=vocab,
            tokenizer=tokenizer,
            max_decode_len=args.max_generate_len,
            num_samples=args.preview_samples,
            decode_method=args.decode_method,
            beam_size=args.beam_size,
            beam_length_penalty=args.beam_length_penalty,
        )

        model_for_save = model.module if isinstance(model, torch.nn.DataParallel) else model

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_best = args.save_dir / "vit_transformer_best.pt"
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
        # Always save the last transformer model from the final epoch
        ckpt_last = args.save_dir / "vit_transformer_last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_for_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "vocab_size": len(vocab),
                "best_val_loss": best_val_loss,
            },
            ckpt_last,
        )

        print(f"  saved last checkpoint     -> {ckpt_last}")
        print()

    print("Training complete.")


if __name__ == "__main__":
    main()

