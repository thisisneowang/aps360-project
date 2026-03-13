"""ViT + Transformer encoder-decoder final model for image-to-LaTeX conversion.


"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed2D(nn.Module):
    """The first step in a ViT: convert an image into a sequence of non-overlapping smaller 'patch' tokens."""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    # First apply a convolution (proj) to chop up the image into patches
    # Then, count how many rows and columsn there are
    # Next, perform some shape trickery (shown below)
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        # [B, C, H, W] -> [B, D, Gh, Gw] -> [B, Gh*Gw, D]
        x = self.proj(images)
        grid_h, grid_w = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return tokens, grid_h, grid_w

# Could have used RoPE if you were feeling ambitious. But this might be a bit easier.
class Learned2DPositionalEncoding(nn.Module):
    """Learning 2D positional embeddings for patch grids in a ViT."""

    def __init__(self, d_model: int, max_grid_h: int, max_grid_w: int) -> None:
        super().__init__()
        self.max_grid_h = max_grid_h
        self.max_grid_w = max_grid_w
        self.row_embed = nn.Embedding(max_grid_h, d_model)
        self.col_embed = nn.Embedding(max_grid_w, d_model)

    def forward(self, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
        if grid_h > self.max_grid_h or grid_w > self.max_grid_w:
            raise ValueError(
                "Patch grid larger than positional encoding capacity: "
                f"got ({grid_h}, {grid_w}), "
                f"max=({self.max_grid_h}, {self.max_grid_w})."
            )

        # Learn row and column positional embeddings separately
        row_ids = torch.arange(grid_h, device=device)
        col_ids = torch.arange(grid_w, device=device)

        # Generally, can concatenate these positional embeddings to vector representations
        row = self.row_embed(row_ids)[:, None, :]  # [Gh, 1, D]
        col = self.col_embed(col_ids)[None, :, :]  # [1, Gw, D]

        # Summarize those orthogonal positional embeddings
        pos = row + col  # [Gh, Gw, D]
        return pos.reshape(1, grid_h * grid_w, -1)


class VisionTransformerEncoder(nn.Module):
    """ViT encoder that returns visual token memory for cross-attention."""

    def __init__(
        self,
        in_channels: int = 1,
        image_height: int = 64,
        image_width: int = 256,
        patch_size: int = 8,
        d_model: int = 384,
        depth: int = 10,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # In-case something went wrong
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError(
                "image_height and image_width must be divisible by patch_size. "
                f"Got image=({image_height}, {image_width}), patch={patch_size}."
            )

        # Generate patch embeddings
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size,
        )

        # Throw in the positional embeddings
        max_grid_h = image_height // patch_size
        max_grid_w = image_width // patch_size
        self.pos_embed = Learned2DPositionalEncoding(
            d_model=d_model,
            max_grid_h=max_grid_h,
            max_grid_w=max_grid_w,
        )
        self.input_dropout = nn.Dropout(dropout)

        # Cook up the transformer 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, grid_h, grid_w = self.patch_embed(images)
        pos = self.pos_embed(grid_h, grid_w, device=tokens.device)
        x = self.input_dropout(tokens + pos)
        x = self.encoder(x)
        x = self.norm(x)

        summary = x.mean(dim=1)
        return x, summary

# Similar to ViT, just with less patching. Thankfully PyTorch just has libraries for transformers (mostly).
class TransformerLatexDecoder(nn.Module):
    """Autoregressive Transformer decoder over visual memory tokens."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_id: int = 0,
    ) -> None:
        super().__init__()

        self.pad_id = pad_id
        self.max_len = max_len

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.input_dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying often improves LM-style decoders and lowers params.
        self.output_proj.weight = self.token_embed.weight

    # Make sure that the transformer can't attend to tokens that are too ahead yet! So add a triangular matrix filled with neg. infinities.
    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1,
        )

    # Otherwise, it's a matter of stacking many transformer blocks
    def forward(self, decoder_input_ids: torch.Tensor, encoder_seq: torch.Tensor) -> torch.Tensor:
        batch_size, tgt_len = decoder_input_ids.shape

        if tgt_len > self.max_len:
            raise ValueError(
                f"decoder_input_ids length {tgt_len} exceeds decoder max_len {self.max_len}."
            )

        pos_ids = torch.arange(tgt_len, device=decoder_input_ids.device)
        pos_ids = pos_ids.unsqueeze(0).expand(batch_size, tgt_len)

        tgt = self.token_embed(decoder_input_ids) + self.pos_embed(pos_ids)
        tgt = self.input_dropout(tgt)

        tgt_mask = self._causal_mask(tgt_len, device=decoder_input_ids.device)
        tgt_key_padding_mask = decoder_input_ids.eq(self.pad_id)

        # Note: because we use memory=encoder_seq, we're specifically attending to the tokens provided by the ViT
        decoded = self.decoder(
            tgt=tgt,
            memory=encoder_seq,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        decoded = self.norm(decoded)
        logits = self.output_proj(decoded)
        return logits

    # Just FC + softmax and find the single argmax (or one of multiple).
    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_seq: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        batch_size = encoder_seq.size(0)
        device = encoder_seq.device

        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self.forward(generated, encoder_seq)
            next_token = logits[:, -1, :].argmax(dim=-1)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(eos_id)

            if bool(finished.all().item()):
                break

        # Return tokens after BOS.
        return generated[:, 1:]

    @staticmethod
    def _length_normalized_score(score: float, length: int, length_penalty: float) -> float:
        if length_penalty <= 0.0:
            return score
        return score / (((5.0 + float(length)) / 6.0) ** length_penalty)

    '''Beam search is simple if you think about it. The idea is that when you just start with <BOS>, consider all V 
    condiitonal probabilities and select the B top ones. That gives you B starting prefixes. Now, repeat the following
    loop: 
    
    Consider all possible B * V probabilities. That means for each of the B beam prefixes, use their output token 
    embedding to calculate what the set of probabilities among the vocabulary would be. Repeat this for each beam prefix,
    giving you B * V probabilities in total, and select the top B out of all of those. That could lead to some of the
    previous prefixes being deleted, but that's basically how you implement the model.'''
    @torch.no_grad()
    def beam_search_decode(
        self,
        encoder_seq: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        beam_size: int = 20,
        length_penalty: float = 0.0,
    ) -> torch.Tensor:
        if beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {beam_size}")

        batch_size = encoder_seq.size(0)
        device = encoder_seq.device
        results: list[torch.Tensor] = []

        for i in range(batch_size):
            # Doing it a batch item at a time
            memory_i = encoder_seq[i : i + 1]

            # Store information about each beam as a tuple of the form (tokens, cumulative_logprob, ended)
            beams: list[tuple[torch.Tensor, float, bool]] = [
                (torch.tensor([bos_id], dtype=torch.long, device=device), 0.0, False)
            ]

            for _ in range(max_len):
                # List of all prefix beams to look at for now
                candidates: list[tuple[torch.Tensor, float, bool]] = []

                for tokens, score, ended in beams:
                    # If some prefix already predicted <EOS>, skip over it.
                    if ended:
                        candidates.append((tokens, score, True))
                        continue

                    # To run the decoder on our current prefix and encoder memory
                    logits = self.forward(tokens.unsqueeze(0), memory_i)[:, -1, :].squeeze(0)
                    # Get each logit form the last output of forward
                    log_probs = torch.log_softmax(logits, dim=-1)

                    # Don't necessarily need to look towards every token in the vocabulary.
                    # Might be sufficient to take the best 'beam_size' choices (or vocab_size, whatever is smaller)
                    topk = min(beam_size, int(log_probs.numel()))
                    # And then, of course, take the top B ones
                    top_scores, top_ids = torch.topk(log_probs, k=topk, dim=-1)

                    # This updates all the candidate prefix beams with their new scores, tokens, etc.
                    for token_logprob, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                        next_token = torch.tensor([token_id], dtype=torch.long, device=device)
                        new_tokens = torch.cat([tokens, next_token], dim=0)
                        new_score = score + float(token_logprob) # autoregressive prob turns into addition after a logarithm
                        new_ended = token_id == eos_id
                        candidates.append((new_tokens, new_score, new_ended))

                # Apply a score to beam search sequences based on their length normalization penalty
                def rank_key(item: tuple[torch.Tensor, float, bool]) -> float:
                    seq_tokens, seq_score, _ = item
                    
                    content_len = max(1, int(seq_tokens.numel() - 1))
                    return self._length_normalized_score(seq_score, content_len, length_penalty)

                # Sort each candidate by that normalized score and select the best beam_size of them
                candidates.sort(key=rank_key, reverse=True)
                beams = candidates[:beam_size]

                # In case all beams have ended, then there's nothing to do anymore.
                if all(ended for _, _, ended in beams):
                    break

            # Ideally just choose from beams who have already completed first.
            ended_beams = [item for item in beams if item[2]]
            final_pool = ended_beams if ended_beams else beams
            # Then, select the one best beam using a length-normalized score (somewhat similar in spirit to the above)
            best_tokens, _, _ = max(
                final_pool,
                key=lambda item: self._length_normalized_score(
                    item[1],
                    max(1, int(item[0].numel() - 1)),
                    length_penalty,
                ),
            )

            # Remove the <BOS> token at the end of each token sequence.
            results.append(best_tokens[1:])

        if not results:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)

        # Padding each seuqence to a common length. So first find their max overall sequence length.
        max_out_len = max(int(seq.numel()) for seq in results)
        if max_out_len == 0:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)

        # Then inflate a padding tensor, and then pad the original tensor!
        out = torch.full((batch_size, max_out_len), fill_value=self.pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(results):
            if seq.numel() > 0:
                out[i, : seq.numel()] = seq
        return out

# Concate ViT-Transformer decoder into a final pipeline
class VitTransformerIm2LatexModel(nn.Module):
    """ViT encoder + Transformer decoder model for IM2LaTeX."""

    def __init__(
        self,
        vocab_size: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        in_channels: int = 1,
        image_height: int = 64,
        image_width: int = 256,
        patch_size: int = 8,
        d_model: int = 384,
        encoder_layers: int = 10,
        encoder_heads: int = 6,
        encoder_mlp_ratio: float = 4.0,
        decoder_layers: int = 6,
        decoder_heads: int = 6,
        decoder_ffn_dim: int = 1536,
        dropout: float = 0.1,
        max_decode_len: int = 256,
    ) -> None:
        super().__init__()

        self.bos_id = bos_id
        self.eos_id = eos_id

        # Load the two main transformer models
        self.encoder = VisionTransformerEncoder(
            in_channels=in_channels,
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
            d_model=d_model,
            depth=encoder_layers,
            num_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            dropout=dropout,
        )
        self.decoder = TransformerLatexDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            ffn_dim=decoder_ffn_dim,
            dropout=dropout,
            max_len=max_decode_len,
            pad_id=pad_id,
        )

    def forward(self, images: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        encoder_seq, _ = self.encoder(images)
        return self.decoder(decoder_input_ids=decoder_input_ids, encoder_seq=encoder_seq)

    # Used only during validation/testing/inference for autoregressive generation
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: int = 192,
        decode_method: str = "greedy",
        beam_size: int = 1,
        beam_length_penalty: float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        encoder_seq, _ = self.encoder(images)

        method = decode_method.lower()
        if method not in {"greedy", "beam"}:
            raise ValueError(f"decode_method must be 'greedy' or 'beam', got {decode_method!r}")

        if method == "beam" or beam_size > 1:
            return self.decoder.beam_search_decode(
                encoder_seq=encoder_seq,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=beam_length_penalty,
            )

        return self.decoder.greedy_decode(
            encoder_seq=encoder_seq,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            max_len=max_len,
        )


def count_trainable_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters, which should be in the tens of millions range."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
