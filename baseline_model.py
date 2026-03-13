"""Baseline image-to-LaTeX model.

Architecture:
- CNN encoder (4 downsampling conv blocks)
- Plain LSTM decoder (autoregressive, no attention)
- Greedy decoding during inference
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """CNN doubles number of channels with each layer while roughly halving feature map dimensions."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        blocks = []
        prev = in_channels
        for ch in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
            )
            prev = ch

        self.conv_blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(channels[-1], hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This returns a tuple of (encoder_sequence, encoder_summary).

        images: [B, C, H, W]
        encoder_sequence: [B, T, HIDDEN_DIM]
        encoder_summary:  [B, HIDDEN_DIM]
        """
        x = self.conv_blocks(images)
        x = self.dropout(self.proj(x))

        # Collapse height into channel features and keep width as sequence axis.
        x = x.mean(dim=2)  # [B, HIDDEN, W']
        seq = x.transpose(1, 2).contiguous()  # [B, W', HIDDEN]
        summary = seq.mean(dim=1)  # [B, HIDDEN]
        return seq, summary


class AutoregressiveLSTMDecoder(nn.Module):
    """A simple non-attention-based autoregressive LSTM decoder."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # embedding matrix to project each token in the vocabulary into embedding space
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # used to transform the final output of the CNN into LSTM initialization
        self.init_h = nn.Linear(hidden_dim, hidden_dim * num_layers)
        self.init_c = nn.Linear(hidden_dim, hidden_dim * num_layers)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def _init_state(self, encoder_summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden/cell state from encoder summary."""
        batch_size = encoder_summary.size(0)

        h0 = self.init_h(encoder_summary)
        c0 = self.init_c(encoder_summary)

        h0 = h0.view(batch_size, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = c0.view(batch_size, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        return h0, c0

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_seq: torch.Tensor,
        encoder_summary: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced decoding. Should only be used during training and nowhere else

        Just take the inputs, embed them, and input those embeddings into the LSTM decoder
        decoder_input_ids: [B, T]
        returns logits: [B, T, VOCAB]
        """

        hidden, cell = self._init_state(encoder_summary)
        embedded = self.input_dropout(self.embedding(decoder_input_ids))
        outputs, _ = self.lstm(embedded, (hidden, cell))
        logits = self.output_proj(outputs)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_seq: torch.Tensor,
        encoder_summary: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        """Greedy decoding only. That's O(|V|) per step to find argmax.

        This returns predicted token IDs (without <BOS>, of course) with shape [B, <=max_len].
        """

        batch_size = encoder_summary.size(0)
        device = encoder_summary.device

        hidden, cell = self._init_state(encoder_summary)
        current = torch.full((batch_size,), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        outputs = []
        # take the current embedding, feed it to the next LSTM timestep
        # and then find the token which has the highest value logit (greedy decoding)
        # append that token to the current prefix
        for _ in range(max_len):
            emb = self.embedding(current).unsqueeze(1)
            output, (hidden, cell) = self.lstm(emb, (hidden, cell))

            logits_t = self.output_proj(output.squeeze(1))
            next_token = logits_t.argmax(dim=-1)

            outputs.append(next_token)
            finished = finished | next_token.eq(eos_id)

            # if a sequence has already finished, just produce EOS. otherwise, keep decoding.
            current = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            if bool(finished.all().item()):
                break

        if not outputs:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)
        return torch.stack(outputs, dim=1)


class BaselineIm2LatexModel(nn.Module):
    """CNN encoder + LSTM decoder baseline model."""

    def __init__(
        self,
        vocab_size: int,
        bos_id: int,
        eos_id: int,
        in_channels: int = 1,
        encoder_base_channels: int = 64,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
        decoder_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Attach all previous functions into a single, cohesive unit

        self.bos_id = bos_id
        self.eos_id = eos_id

        self.encoder = CNNEncoder(
            in_channels=in_channels,
            base_channels=encoder_base_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.decoder = AutoregressiveLSTMDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=decoder_layers,
            dropout=dropout,
        )

    def forward(self, images: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits for teacher-forced training.

        images: [B, C, H, W]
        decoder_input_ids: [B, T]
        returns logits: [B, T, VOCAB]
        """
        encoder_seq, encoder_summary = self.encoder(images)
        logits = self.decoder(decoder_input_ids, encoder_seq, encoder_summary)
        return logits

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int = 128) -> torch.Tensor:
        """Generate sequences with greedy decoding."""
        self.eval()
        encoder_seq, encoder_summary = self.encoder(images)
        return self.decoder.greedy_decode(
            encoder_seq=encoder_seq,
            encoder_summary=encoder_summary,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            max_len=max_len,
        )


def count_trainable_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters in the overall model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
