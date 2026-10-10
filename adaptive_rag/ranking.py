"""Small PyTorch learning-to-rank and budget-routing heads, safe tensor files."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file


class Ranker(nn.Module):
    def __init__(self, dimension: int = 384):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension * 2 + 3, 64), nn.ReLU(), nn.Dropout(.1),
                                     nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)

    @torch.inference_mode()
    def predict(self, features: np.ndarray) -> np.ndarray:
        self.eval()
        return self(torch.from_numpy(features)).sigmoid().numpy()


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(7, 32), nn.Tanh(), nn.Linear(32, 16),
                                     nn.Tanh(), nn.Linear(16, 3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    @torch.inference_mode()
    def predict(self, features: np.ndarray) -> np.ndarray:
        self.eval()
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None]
        return self(torch.from_numpy(features)).argmax(dim=1).numpy()


def save_head(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().contiguous().cpu() for key, value in model.state_dict().items()}, str(path))


def load_head(model: nn.Module, path: str | Path) -> nn.Module:
    model.load_state_dict(load_file(str(path), device="cpu"))
    model.eval()
    return model


def fit_ranker(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
               seed: int = 42, epochs: int = 60) -> tuple[Ranker, dict]:
    torch.manual_seed(seed)
    model = Ranker((train_x.shape[1] - 3) // 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    x, y = torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))
    vx, vy = torch.from_numpy(val_x), torch.from_numpy(val_y.astype(np.float32))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum() / (y == 1).sum())))
    best, state, best_epoch = float("inf"), None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(x))
        for start in range(0, len(x), 128):
            index = indices[start:start + 128]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[index]), y[index])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(vx), vy))
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            state = {key: value.clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            history.append({"epoch": epoch, "validation_bce": val_loss})
    model.load_state_dict(state)
    model.eval()
    return model, {"selected_epoch": best_epoch, "validation_bce": best, "history": history,
                   "train_pairs": len(x), "validation_pairs": len(vx), "parameters": sum(p.numel() for p in model.parameters())}


def fit_router(features: np.ndarray, labels: np.ndarray, seed: int = 42) -> Router:
    torch.manual_seed(seed)
    model = Router()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.03)
    x, y = torch.from_numpy(features), torch.from_numpy(labels.astype(np.int64))
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model
