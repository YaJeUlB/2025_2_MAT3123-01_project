from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    seed: int = 0
    device: str = "cpu"

    batch_size: int = 256
    epochs_base: int = 10
    epochs_spec: int = 3
    lr_base: float = 1e-3
    lr_spec: float = 1e-4

    tau: float = 0.9
    val_size: int = 10000  # FashionMNIST train=60000 -> 50k train_base, 10k val_gate


# -----------------------------
# Model
# -----------------------------
class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)  # 28->14

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.2, training=self.training)
        return self.fc2(x)


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: str) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total += bs
    return total_loss / total


@torch.no_grad()
def eval_with_conf_split(model: nn.Module, loader: DataLoader, device: str, tau: float) -> dict:
    """
    Evaluate model accuracy and also accuracy on:
      - low-confidence set (conf < tau), where conf is computed from THIS model
      - high-confidence set (conf >= tau)
    """
    model.eval()
    correct, total = 0, 0
    low_correct, low_total = 0, 0
    high_correct, high_total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        probs = F.softmax(model(x), dim=1)
        conf, pred = probs.max(dim=1)

        total += x.size(0)
        correct += (pred == y).sum().item()

        low_mask = conf < tau
        high_mask = ~low_mask

        if low_mask.any():
            low_total += low_mask.sum().item()
            low_correct += (pred[low_mask] == y[low_mask]).sum().item()
        if high_mask.any():
            high_total += high_mask.sum().item()
            high_correct += (pred[high_mask] == y[high_mask]).sum().item()

    return {
        "overall_acc": correct / total,
        "low_acc": (low_correct / low_total) if low_total > 0 else float("nan"),
        "high_acc": (high_correct / high_total) if high_total > 0 else float("nan"),
        "mu_Utau": low_total / total,
        "low_count": low_total,
        "high_count": high_total,
        "total": total,
    }


@torch.no_grad()
def low_conf_indices_on_subset(base: nn.Module, subset_loader: DataLoader, device: str, tau: float) -> list[int]:
    """
    Return positions (0..len(subset)-1) in a subset where base confidence < tau.
    """
    base.eval()
    idxs: list[int] = []
    seen = 0

    for x, _y in subset_loader:
        x = x.to(device)
        probs = F.softmax(base(x), dim=1)
        conf = probs.max(dim=1).values

        mask = conf < tau
        pos = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
        idxs.extend([seen + j for j in pos])
        seen += x.size(0)

    return idxs


@torch.no_grad()
def eval_gated(base: nn.Module, spec: nn.Module, loader: DataLoader, device: str, tau: float) -> dict:
    """
    Gate by BASE confidence:
      if base_conf >= tau -> base prediction
      else -> specialist prediction
    Report low/high split defined by base_conf.
    """
    base.eval()
    spec.eval()

    correct, total = 0, 0
    low_correct, low_total = 0, 0
    high_correct, high_total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        base_probs = F.softmax(base(x), dim=1)
        base_conf, base_pred = base_probs.max(dim=1)

        spec_pred = spec(x).argmax(dim=1)

        use_base = base_conf >= tau
        final_pred = torch.where(use_base, base_pred, spec_pred)

        total += x.size(0)
        correct += (final_pred == y).sum().item()

        low_mask = ~use_base
        high_mask = use_base

        if low_mask.any():
            low_total += low_mask.sum().item()
            low_correct += (final_pred[low_mask] == y[low_mask]).sum().item()
        if high_mask.any():
            high_total += high_mask.sum().item()
            high_correct += (final_pred[high_mask] == y[high_mask]).sum().item()

    return {
        "overall_acc": correct / total,
        "low_acc": (low_correct / low_total) if low_total > 0 else float("nan"),
        "high_acc": (high_correct / high_total) if high_total > 0 else float("nan"),
        "mu_Utau": low_total / total,
        "low_count": low_total,
        "high_count": high_total,
        "total": total,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = CFG()
    set_seed(cfg.seed)

    transform = transforms.Compose([transforms.ToTensor()])

    full_train = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)

    # Split train into train_base and val_gate
    val_size = cfg.val_size
    train_size = len(full_train) - val_size
    gen = torch.Generator().manual_seed(cfg.seed)
    train_base, val_gate = random_split(full_train, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_base, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_gate, batch_size=cfg.batch_size, shuffle=False,)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # 1) Train base model
    base = BaseCNN().to(cfg.device)
    opt_base = torch.optim.Adam(base.parameters(), lr=cfg.lr_base)

    for ep in range(1, cfg.epochs_base + 1):
        loss = train_one_epoch(base, train_loader, opt_base, cfg.device)
        print(f"[Base] Epoch {ep}/{cfg.epochs_base} | loss={loss:.4f}")

    # Evaluate base on test
    base_stats = eval_with_conf_split(base, test_loader, cfg.device, tau=cfg.tau)
    print("\n=== Base (test, conf split by base) ===")
    for k, v in base_stats.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")

    # 2) Build low-confidence subset from val_gate (using base confidence)
    hard_positions = low_conf_indices_on_subset(base, val_loader, cfg.device, cfg.tau)
    print(f"\nLow-conf set size (val_gate): {len(hard_positions)} / {len(val_gate)}  (tau={cfg.tau})")

    if len(hard_positions) == 0:
        print("Low-conf set is empty. Increase tau or adjust training.")
        return

    hard_subset = Subset(val_gate, hard_positions)
    hard_loader = DataLoader(hard_subset, batch_size=cfg.batch_size, shuffle=True)

    # 3) Specialist = base model fine-tuned on low-conf subset
    spec = BaseCNN().to(cfg.device)
    spec.load_state_dict(base.state_dict())
    opt_spec = torch.optim.Adam(spec.parameters(), lr=cfg.lr_spec)

    for ep in range(1, cfg.epochs_spec + 1):
        loss = train_one_epoch(spec, hard_loader, opt_spec, cfg.device)
        print(f"[Spec] Epoch {ep}/{cfg.epochs_spec} | loss={loss:.4f}")

    # 4) Gated evaluation on test (gate by base confidence)
    gated_stats = eval_gated(base, spec, test_loader, cfg.device, tau=cfg.tau)
    print("\n=== Gated Base+Specialist (test, gate by base conf) ===")
    for k, v in gated_stats.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")


if __name__ == "__main__":
    main()
