"""Train entity embeddings for geo_level_* features.

Trains a small supervised PyTorch MLP with embedding layers for the three
geo_level columns, then extracts the learned embedding matrices for use as
dense features in LGB/XGB alongside GLMM encoding.

Architecture: geo_level_{1,2,3} → embedding → concat → BN → MLP → softmax(3)

Trained on full training data. This introduces mild indirect target leakage
(embeddings encode aggregate damage patterns per geo ID), which is standard
practice for embedding-based feature engineering in tabular competitions.

Usage:
    python src/geo_embeddings.py --config configs/config.yaml

Outputs (in models/):
    geo_emb_1.npy  — (vocab1, dim1) float32 embedding matrix for geo_level_1_id
    geo_emb_2.npy  — (vocab2, dim2) float32 embedding matrix for geo_level_2_id
    geo_emb_3.npy  — (vocab3, dim3) float32 embedding matrix for geo_level_3_id
    geo_enc_1.pkl  — LabelEncoder mapping raw IDs to embedding indices
    geo_enc_2.pkl  — LabelEncoder mapping raw IDs to embedding indices
    geo_enc_3.pkl  — LabelEncoder mapping raw IDs to embedding indices
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from utils.seed import set_global_seed

logger = logging.getLogger(__name__)

UNKNOWN_IDX: int = 0  # embedding index 0 reserved for unseen / rare categories


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GeoEmbeddingModel(nn.Module):
    """Supervised embedding MLP for geo_level_{1,2,3} categorical features.

    Architecture:
        geo_i → Embedding(vocab_i, dim_i) → concat → BN → [Linear → ReLU → Dropout]* → Linear(n_classes)

    Args:
        vocab_sizes: Vocab sizes (incl. UNKNOWN slot at index 0) for each geo column.
        emb_dims: Embedding dimensions for each geo column.
        hidden_sizes: Hidden layer sizes for the MLP head.
        n_classes: Number of output classes.
        dropout: Dropout rate applied after each hidden layer.
    """

    def __init__(
        self,
        vocab_sizes: tuple[int, int, int],
        emb_dims: tuple[int, int, int],
        hidden_sizes: list[int],
        n_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb1 = nn.Embedding(vocab_sizes[0], emb_dims[0], padding_idx=UNKNOWN_IDX)
        self.emb2 = nn.Embedding(vocab_sizes[1], emb_dims[1], padding_idx=UNKNOWN_IDX)
        self.emb3 = nn.Embedding(vocab_sizes[2], emb_dims[2], padding_idx=UNKNOWN_IDX)

        emb_total = sum(emb_dims)
        layers: list[nn.Module] = [nn.BatchNorm1d(emb_total)]
        in_size = emb_total
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        g1: torch.Tensor,
        g2: torch.Tensor,
        g3: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            g1: (batch,) integer indices for geo_level_1_id.
            g2: (batch,) integer indices for geo_level_2_id.
            g3: (batch,) integer indices for geo_level_3_id.

        Returns:
            Logits of shape (batch, n_classes).
        """
        x = torch.cat([self.emb1(g1), self.emb2(g2), self.emb3(g3)], dim=1)
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _encode_geo_col(
    series: pd.Series,
    le: LabelEncoder,
) -> np.ndarray:
    """Map raw geo IDs to 1-based embedding indices. Unseen IDs → UNKNOWN_IDX (0).

    Args:
        series: Raw geo ID column (any dtype, converted to str).
        le: Fitted LabelEncoder.

    Returns:
        Integer index array of shape (n,).
    """
    raw = series.astype(str).values
    known = set(le.classes_)
    indices = np.zeros(len(raw), dtype=np.int64)
    mask = np.array([v in known for v in raw])
    if mask.any():
        indices[mask] = le.transform(raw[mask]) + 1  # +1: shift known → 1..N
    return indices


def train_geo_embeddings(cfg: dict, models_dir: Path) -> None:
    """Train the geo embedding model and save matrices + encoders.

    Args:
        cfg: Full config dict.
        models_dir: Directory where outputs are saved.
    """
    seed = cfg["seed"]
    set_global_seed(seed)

    emb_cfg = cfg["features"]["geo_embedding"]
    epochs: int = emb_cfg["epochs"]
    batch_size: int = emb_cfg["batch_size"]
    lr: float = emb_cfg["learning_rate"]
    dropout: float = emb_cfg["dropout"]
    hidden_sizes: list[int] = emb_cfg["hidden_sizes"]
    emb_dims_map: dict[str, int] = emb_cfg["dims"]

    geo_cols: list[str] = cfg["features"]["geo_cols"]
    target_col: str = cfg["data"]["target_col"]
    n_classes: int = len(cfg["features"]["classes"])

    # -- Load full training data --
    train_values = pd.read_csv(cfg["data"]["train_values"], encoding="utf-8")
    train_labels = pd.read_csv(cfg["data"]["train_labels"], encoding="utf-8")
    df = train_values.merge(train_labels, on=cfg["data"]["id_col"])

    y = (df[target_col].values - 1).astype(np.int64)  # 0-indexed

    # -- Fit LabelEncoders and encode each geo column --
    encoders: list[LabelEncoder] = []
    geo_index_arrays: list[np.ndarray] = []
    vocab_sizes: list[int] = []
    emb_dims: list[int] = []

    for col in geo_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        indices = _encode_geo_col(df[col], le)
        vocab_sizes.append(len(le.classes_) + 1)  # +1 for UNKNOWN slot
        emb_dims.append(emb_dims_map[col])
        encoders.append(le)
        geo_index_arrays.append(indices)
        logger.info(
            "%s: %d unique → vocab_size=%d, emb_dim=%d",
            col,
            len(le.classes_),
            vocab_sizes[-1],
            emb_dims[-1],
        )

    # -- Build DataLoader --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training geo embeddings on {device}.", flush=True)

    ds = TensorDataset(
        torch.tensor(geo_index_arrays[0], dtype=torch.long),
        torch.tensor(geo_index_arrays[1], dtype=torch.long),
        torch.tensor(geo_index_arrays[2], dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # -- Build model --
    model = GeoEmbeddingModel(
        vocab_sizes=tuple(vocab_sizes),  # type: ignore[arg-type]
        emb_dims=tuple(emb_dims),  # type: ignore[arg-type]
        hidden_sizes=hidden_sizes,
        n_classes=n_classes,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )

    # -- Training loop --
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for b_g1, b_g2, b_g3, b_y in loader:
            b_g1 = b_g1.to(device)
            b_g2 = b_g2.to(device)
            b_g3 = b_g3.to(device)
            b_y = b_y.to(device)
            optimizer.zero_grad()
            logits = model(b_g1, b_g2, b_g3)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b_y)
            correct += (logits.argmax(1) == b_y).sum().item()
            total += len(b_y)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs}: loss={epoch_loss / total:.4f},"
                f" acc={correct / total:.4f}",
                flush=True,
            )

    # -- Extract embedding matrices --
    model.eval()
    with torch.inference_mode():
        emb_matrices = [
            model.emb1.weight.cpu().numpy().astype(np.float32),
            model.emb2.weight.cpu().numpy().astype(np.float32),
            model.emb3.weight.cpu().numpy().astype(np.float32),
        ]

    # -- Save embeddings and encoders --
    for col, enc, emb_matrix in zip(geo_cols, encoders, emb_matrices):
        tag = col.replace("geo_level_", "").replace("_id", "")
        emb_path = models_dir / f"geo_emb_{tag}.npy"
        enc_path = models_dir / f"geo_enc_{tag}.pkl"
        np.save(emb_path, emb_matrix)
        joblib.dump(enc, enc_path)
        print(f"  Saved {emb_path}  shape={emb_matrix.shape}", flush=True)
        print(f"  Saved {enc_path}", flush=True)

    print(f"\nDone. Geo embeddings saved to {models_dir}/", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train geo entity embeddings")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models_dir = Path(cfg["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Training geo entity embeddings...", flush=True)
    train_geo_embeddings(cfg, models_dir)


if __name__ == "__main__":
    main()
