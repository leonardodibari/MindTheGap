"""SchNet construction, training, checkpointing, and inference."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet

from .utils import seed_everything


class DenseRadiusInteractionGraph(nn.Module):
    """Pure-PyTorch fallback when the optional optimized radius graph is unavailable."""

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, pos, batch):
        distances = torch.cdist(pos, pos)
        mask = batch[:, None].eq(batch[None, :]) & distances.gt(0) & distances.lt(self.cutoff)
        edge_index = mask.nonzero(as_tuple=False).T.contiguous()
        row, column = edge_index
        return edge_index, distances[row, column]


def optional_interaction_graph(cutoff: float):
    """Use PyG's optimized graph when available, otherwise the exact dense fallback."""
    try:
        import pyg_lib  # noqa: F401
        return None
    except ImportError:
        return DenseRadiusInteractionGraph(cutoff)


def make_dataloader(
    graphs: Sequence,
    indices,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 42,
    device: torch.device | None = None,
) -> DataLoader:
    """Build a deterministic graph dataloader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(graphs, np.asarray(indices).tolist()), batch_size=batch_size,
        shuffle=shuffle, num_workers=0,
        pin_memory=bool(device is not None and device.type == "cuda"),
        generator=generator,
    )


def build_model(config: dict[str, Any], device: torch.device) -> SchNet:
    """Construct the exact SchNet architecture encoded by a visible notebook config."""
    return SchNet(
        hidden_channels=config["hidden_channels"],
        num_filters=config["hidden_channels"],
        num_interactions=config["num_interactions"],
        num_gaussians=25,
        cutoff=config["cutoff"],
        interaction_graph=optional_interaction_graph(config["cutoff"]),
        max_num_neighbors=32,
        readout="mean",
    ).to(device)


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return targets and predictions in loader order."""
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.z, batch.pos, batch.batch).view(-1)
        predictions.append(output.detach().cpu())
        targets.append(batch.y.view(-1).detach().cpu())
    return torch.cat(targets).numpy(), torch.cat(predictions).numpy()


def save_checkpoint(model, config: dict[str, Any], epoch: int, validation_mae: float, path: Path) -> None:
    """Save a SchNet checkpoint using the existing schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch,
                "validation_mae": float(validation_mae)}, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[SchNet, dict[str, Any]]:
    """Load the exact architecture and state stored in a checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["config"]


def train_schnet(
    config: dict[str, Any],
    graphs: Sequence,
    train_indices,
    validation_indices,
    checkpoint_path: Path,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train with the notebook's unchanged MAE loss, clipping, and early stopping."""
    seed_everything(seed, deterministic_torch=True)
    train_loader = make_dataloader(
        graphs, train_indices, config["batch_size"], shuffle=True, seed=seed, device=device
    )
    validation_loader = make_dataloader(
        graphs, validation_indices, config["batch_size"] * 2, seed=seed, device=device
    )
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
    )
    criterion = nn.L1Loss()
    history = {"train_mae": [], "validation_mae": []}
    best_mae, best_epoch, stale_epochs = np.inf, 0, 0
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        total_absolute_error, total_samples = 0., 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output, target = model(batch.z, batch.pos, batch.batch).view(-1), batch.y.view(-1)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_absolute_error += float(loss.detach()) * len(target)
            total_samples += len(target)
        train_mae = total_absolute_error / total_samples
        validation_true, validation_prediction = predict(model, validation_loader, device)
        validation_mae = mean_absolute_error(validation_true, validation_prediction)
        history["train_mae"].append(train_mae)
        history["validation_mae"].append(validation_mae)
        if validation_mae < best_mae - config["min_delta"]:
            best_mae, best_epoch, stale_epochs = validation_mae, epoch, 0
            save_checkpoint(model, config, epoch, validation_mae, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                break
    training_seconds = time.perf_counter() - started
    model, _ = load_checkpoint(checkpoint_path, device)
    return {"model": model, "history": history, "best_validation_mae": best_mae,
            "best_epoch": best_epoch, "training_seconds": training_seconds,
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}

