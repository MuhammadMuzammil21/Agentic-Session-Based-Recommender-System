"""models/encoder.py — GRU + Self-Attention session encoder and training harness.

SessionEncoder:   embed → GRU → SelfAttentionLayer → session representation.
NextItemTrainer:  training loop, evaluation, checkpointing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from config.settings import Config
from data.vocab import PAD_IDX, UNK_IDX
from models.attention import SelfAttentionLayer
from models.embeddings import ItemEmbedding

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SessionEncoder
# ══════════════════════════════════════════════════════════════════════════════


class SessionEncoder(nn.Module):
    """GRU-based session encoder with multi-head self-attention pooling.

    Pipeline:
        input_ids [B, L]
            → ItemEmbedding     → [B, L, D]
            → PackedGRU         → all_hiddens [B, L, H]
            → SelfAttentionLayer → session_repr [B, H], attn_weights [B, L]

    Args:
        vocab_size:  Vocabulary size (including PAD/UNK tokens).
        embed_dim:   Item embedding dimensionality.
        hidden_dim:  GRU hidden state dimensionality.
        num_heads:   Number of attention heads.
        dropout:     Dropout probability used in embedding and attention layers.
        padding_idx: PAD token index (default 0).

    Example:
        enc = SessionEncoder(vocab_size=500, embed_dim=64, hidden_dim=128,
                             num_heads=4, dropout=0.2)
        ids     = torch.randint(1, 500, (4, 10))
        lengths = torch.tensor([10, 8, 6, 3])
        repr_, weights, hiddens = enc(ids, lengths)
        # repr_:   [4, 128]
        # weights: [4, 10]
        # hiddens: [4, 10, 128]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        self.item_embedding = ItemEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout=dropout,
            padding_idx=padding_idx,
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.attention = SelfAttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)

        # Projection from hidden space → embedding space for scoring.
        # Required when hidden_dim != embed_dim; identity equivalent when equal.
        if hidden_dim != embed_dim:
            self.projection: nn.Module = nn.Linear(hidden_dim, embed_dim, bias=False)
        else:
            self.projection = nn.Identity()

        logger.debug(
            "SessionEncoder: vocab=%d, embed=%d, hidden=%d, heads=%d, proj=%s",
            vocab_size,
            embed_dim,
            hidden_dim,
            num_heads,
            "Linear" if hidden_dim != embed_dim else "Identity",
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Tensor,
        lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode a batch of padded item sequences.

        Args:
            input_ids: LongTensor [B, L] — left-padded item indices.
            lengths:   LongTensor [B]    — true sequence length per sample.

        Returns:
            session_repr:  FloatTensor [B, H] — final session representation.
            attn_weights:  FloatTensor [B, L] — per-token attention weights.
            all_hiddens:   FloatTensor [B, L, H] — all GRU hidden states.
        """
        B, L = input_ids.shape

        # 1. Embed
        embedded = self.item_embedding(input_ids)          # [B, L, D]

        # 2. Pack padded sequence (lengths must be on CPU for pack_padded_sequence)
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        # 3. GRU
        packed_out, _ = self.gru(packed)
        all_hiddens, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=L,
        )                                                  # [B, L, H]
        all_hiddens = self.dropout(all_hiddens)

        # 4. Build padding mask: 1 for real tokens, 0 for PAD
        # input_ids != padding_idx accounts for left-padding
        mask = (input_ids != self.padding_idx).float()    # [B, L]

        # 5. Attention pooling
        session_repr, attn_weights = self.attention(all_hiddens, mask)

        return session_repr, attn_weights, all_hiddens

    # ── Utility ───────────────────────────────────────────────────────────────

    def predict_scores(
        self,
        session_repr: Tensor,
        item_embeddings: Tensor,
    ) -> Tensor:
        """Score all items for a batch of session representations.

        Args:
            session_repr:    FloatTensor [B, H] — session embeddings.
            item_embeddings: FloatTensor [V, D] — full embedding weight matrix.

        Returns:
            FloatTensor [B, V] — unnormalised similarity scores.
        """
        # Project from hidden space → embedding space, then dot with embeddings.
        # [B, H] → [B, D] → [B, V]
        proj_repr = self.projection(session_repr)        # [B, D]
        return torch.matmul(proj_repr, item_embeddings.t())  # [B, V]


# ══════════════════════════════════════════════════════════════════════════════
# NextItemTrainer
# ══════════════════════════════════════════════════════════════════════════════


class NextItemTrainer:
    """Training harness for the SessionEncoder next-item prediction task.

    Loss: cross-entropy over all vocabulary items (full softmax).

    Args:
        encoder:    The SessionEncoder model to train.
        vocab_size: Total vocabulary size (number of items).
        cfg:        Project Config object.

    Example:
        trainer = NextItemTrainer(encoder, vocab_size=5000, cfg=cfg)
        loss = trainer.train_epoch(train_loader, optimizer, device)
        metrics = trainer.evaluate(val_loader, device, k_values=[5, 10, 20])
    """

    LOG_EVERY: int = 100  # log batch loss every N steps

    def __init__(
        self,
        encoder: SessionEncoder,
        vocab_size: int,
        cfg: Config,
    ) -> None:
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.cfg = cfg

    # ── Training ──────────────────────────────────────────────────────────────

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> float:
        """Run one training epoch.

        Args:
            dataloader: DataLoader yielding batches with keys
                        ``input_ids``, ``lengths``, ``target``.
            optimizer:  PyTorch optimiser (already configured).
            device:     Device to run computations on.

        Returns:
            Mean cross-entropy loss over the epoch.
        """
        self.encoder.train()
        total_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)   # [B, L]
            lengths   = batch["lengths"].to(device)     # [B]
            targets   = batch["target"].to(device)      # [B]

            optimizer.zero_grad()

            session_repr, _, _ = self.encoder(input_ids, lengths)   # [B, H]
            item_embs = self.encoder.item_embedding.get_all_embeddings().to(device)
            logits = self.encoder.predict_scores(session_repr, item_embs)  # [B, V]

            loss = F.cross_entropy(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (step + 1) % self.LOG_EVERY == 0:
                logger.info(
                    "  step %d/%d — loss=%.4f",
                    step + 1,
                    len(dataloader),
                    loss.item(),
                )

        mean_loss = total_loss / max(n_batches, 1)
        logger.info("train_epoch complete — mean_loss=%.4f", mean_loss)
        return mean_loss

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        dataloader: DataLoader,
        device: torch.device,
        k_values: List[int],
    ) -> Dict[str, float]:
        """Evaluate Recall@K and MRR@K on a held-out set.

        Args:
            dataloader: DataLoader with same batch format as train.
            device:     Compute device.
            k_values:   List of K values, e.g. [5, 10, 20].

        Returns:
            Dict like ``{"Recall@5": 0.31, "MRR@5": 0.18, ...}``.
        """
        self.encoder.eval()
        recall: Dict[int, float] = {k: 0.0 for k in k_values}
        mrr:    Dict[int, float] = {k: 0.0 for k in k_values}
        n_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                lengths   = batch["lengths"].to(device)
                targets   = batch["target"].to(device)

                session_repr, _, _ = self.encoder(input_ids, lengths)
                item_embs = self.encoder.item_embedding.get_all_embeddings().to(device)
                logits = self.encoder.predict_scores(session_repr, item_embs)  # [B, V]

                # PAD and UNK are not real items; never recommend them.
                logits[:, PAD_IDX] = float("-inf")
                logits[:, UNK_IDX] = float("-inf")

                for k in k_values:
                    top_k = torch.topk(logits, k, dim=-1).indices     # [B, k]
                    for b in range(targets.size(0)):
                        gt = targets[b].item()
                        preds = top_k[b].tolist()
                        if gt in preds:
                            recall[k] += 1.0
                            mrr[k] += 1.0 / (preds.index(gt) + 1)

                n_samples += targets.size(0)

        results: Dict[str, float] = {}
        for k in k_values:
            results[f"Recall@{k}"] = recall[k] / max(n_samples, 1)
            results[f"MRR@{k}"]    = mrr[k]    / max(n_samples, 1)

        logger.info(
            "evaluate: n=%d  %s",
            n_samples,
            "  ".join(f"{k}={v:.4f}" for k, v in results.items()),
        )
        return results

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        epoch: int,
        val_recall: float,
        path: str,
    ) -> None:
        """Save encoder state dict to a checkpoint file.

        Args:
            epoch:      Current epoch index (0-based).
            val_recall: Validation Recall@10 used to name the file.
            path:       File path for the checkpoint.

        Note:
            Checkpoint naming convention: epoch_NNN_recallX.XXXX.pt
        """
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "val_recall": val_recall,
            "model_state_dict": self.encoder.state_dict(),
        }
        torch.save(payload, ckpt_path)
        logger.info("Checkpoint saved → %s", ckpt_path)

    def load_checkpoint(self, path: str) -> None:
        """Load encoder weights from a checkpoint file.

        Args:
            path: Path to a checkpoint created by :meth:`save_checkpoint`.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(payload["model_state_dict"])
        logger.info(
            "Checkpoint loaded ← %s (epoch=%d, val_recall=%.4f)",
            ckpt_path,
            payload.get("epoch", -1),
            payload.get("val_recall", float("nan")),
        )
