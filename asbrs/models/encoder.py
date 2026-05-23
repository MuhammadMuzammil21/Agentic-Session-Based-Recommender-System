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

        This is the heart of the model — it walks a session through:
          (1) item-id → 64-number embedding lookup
          (2) the GRU, which reads items left→right and updates a memory
              vector at each step
          (3) the attention layer, which weighted-averages all GRU
              memories into ONE summary vector representing the session

        Args:
            input_ids: LongTensor [B, L] — batch of B sessions, each L=50
                       integer item ids. Older items at the LEFT (padded
                       with 0s), most recent at the RIGHT.
            lengths:   LongTensor [B]    — true (non-PAD) length of each
                       session in the batch.

        Returns:
            session_repr:  FloatTensor [B, H] — one 128-number summary per
                           session. This is the "memory of the whole
                           shopping visit" used to score items.
            attn_weights:  FloatTensor [B, L] — how much the attention
                           layer focused on each of the L positions.
                           Sums to 1.0 over real (non-PAD) positions.
            all_hiddens:   FloatTensor [B, L, H] — the GRU's memory vector
                           at every step (kept for debugging / future use).
        """
        # B = batch size (how many sessions we're processing at once)
        # L = sequence length (always 50 here — the padded length)
        B, L = input_ids.shape

        # ── STEP 1: turn integer ids into embedding vectors ───────────────────
        # input_ids is just integers like [[0, 0, ..., 17, 42, 88], ...].
        # ItemEmbedding looks each integer up in its [vocab_size × 64] table
        # and replaces it with the row of 64 numbers stored at that index.
        # After this line we have one 64-number vector per item position.
        embedded = self.item_embedding(input_ids)          # [B, L, D] = [B, 50, 64]

        # ── STEP 2: tell PyTorch to skip PAD positions when running the GRU ───
        # `pack_padded_sequence` is an optimisation: instead of feeding all 50
        # positions (most of which are zeros / PAD), we tell the GRU
        # "this session only has `lengths[i]` real items at the END; ignore
        # the leading PAD tokens." That's both faster AND keeps the PAD tokens
        # from polluting the GRU's hidden state.
        lengths_cpu = lengths.clamp(min=1).cpu()  # pack_padded needs CPU tensor
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,  # we don't pre-sort by length
        )

        # ── STEP 3: the GRU walks the sequence and updates its memory ────────
        # The GRU starts with a zero "memory vector" (hidden state) of size 128.
        # It reads the first item, mixes it with the zero memory → memory v1.
        # It reads the second item, mixes it with memory v1     → memory v2.
        # …and so on. Each step's output is one 128-number memory snapshot.
        # `packed_out` contains all those snapshots. The `_` we discard is just
        # the final hidden state (we don't need it — attention uses all snapshots).
        packed_out, _ = self.gru(packed)

        # Unpack: restore the [B, 50, 128] shape (PAD positions get zero filler).
        # all_hiddens[b, t, :] = GRU's memory vector at position t for session b.
        all_hiddens, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=L,
        )                                                  # [B, L, H] = [B, 50, 128]

        # Standard regularisation — randomly zero some values to prevent overfitting.
        all_hiddens = self.dropout(all_hiddens)

        # ── STEP 4: build a mask telling attention which positions are real ──
        # mask[b, t] = 1.0 if position t in session b is a real item, 0.0 if PAD.
        # The attention layer uses this so PAD positions get zero weight.
        mask = (input_ids != self.padding_idx).float()    # [B, L]

        # ── STEP 5: attention pools the 50 memory vectors into ONE summary ───
        # The GRU gave us 50 memory snapshots per session. We need ONE vector
        # to represent the whole session for scoring. Attention does this as a
        # *learned* weighted average: it decides which snapshots are most
        # informative and weights them more heavily.
        # Returns:
        #   session_repr [B, H]  — the one 128-number session summary.
        #   attn_weights [B, L]  — how much each input position contributed,
        #                          shown as the bar chart in the demo UI.
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
