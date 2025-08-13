# Device-aware training with memristor LUT using snap-only projection.
# Dataset: MNIST for sanity check. Switch to your MRI loaders later if needed.

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# LUT utilities
# =========================================================
def load_memristor_values(pot_path: str, dep_path: str = None):
    """
    Read LUT text files (one float per line).
    Returns:
        pot_vals (torch.FloatTensor), dep_vals (torch.FloatTensor or None)
    """
    def _read_txt(p):
        with open(p, "r") as f:
            vals = [float(x.strip()) for x in f if x.strip()]
        return torch.tensor(vals, dtype=torch.float32)

    if not os.path.exists(pot_path):
        raise FileNotFoundError(f"Potentiation file not found: {pot_path}")
    pot_vals = _read_txt(pot_path)

    dep_vals = None
    if dep_path is not None and os.path.exists(dep_path):
        dep_vals = _read_txt(dep_path)

    return pot_vals, dep_vals


def build_states_from_lut(pot_vals: torch.Tensor,
                          dep_vals: torch.Tensor = None,
                          zero_center: bool = True) -> torch.Tensor:
    """
    Build a symmetric state vector from pot/dep LUTs.
    Negative branch uses -flip(depression) or -flip(pot) if depression is missing.
    Positive branch uses +potentiation.
    Optionally zero-center the states for stability.
    """
    if dep_vals is not None:
        states_neg = -torch.flip(dep_vals, dims=[0])
    else:
        states_neg = -torch.flip(pot_vals, dims=[0])

    states_pos = pot_vals.clone()
    states = torch.cat([states_neg, states_pos], dim=0)

    # Remove duplicates and sort
    states = torch.unique(states)
    states, _ = torch.sort(states)

    if zero_center:
        states = states - states.mean()

    return states


# =========================================================
# Quantizer using an explicit state vector
# =========================================================
class MemristorQuantizerFromTensor:
    def __init__(self, states: torch.Tensor, device='cpu'):
        states = torch.unique(states).to(torch.float32).to(device)
        states, _ = torch.sort(states)
        self.states = states
        self.S = self.states.numel()

    def snap_to_state(self, w_fp32: torch.Tensor):
        """
        For each weight value, find the nearest state.
        Returns:
            snapped_weight, index_tensor
        """
        w = w_fp32.unsqueeze(-1)             # [..., 1]
        d = torch.abs(w - self.states)       # [..., S]
        idx = torch.argmin(d, dim=-1)        # [...]
        w_snapped = self.states[idx]
        return w_snapped, idx

    def indices_to_weight(self, idx: torch.Tensor):
        return self.states[idx]

    def step_indices(self, idx: torch.Tensor, delta_steps: torch.Tensor):
        # Not used in snap-only mode, but kept for completeness
        if delta_steps.dtype.is_floating_point:
            delta_steps = torch.round(delta_steps)
        new_idx = torch.clamp(idx + delta_steps.to(idx.dtype), 0, self.S - 1)
        return new_idx


# =========================================================
# Device-aware Linear (snap-only projection)
# =========================================================
class DeviceAwareLinear(nn.Module):
    """
    Linear layer with:
      - FP32 master parameters
      - Forward uses device-snapped weights with an STE pass-through
      - After optimizer.step(), call project_after_step() to snap FP32 to nearest device states
    """
    def __init__(self, in_features, out_features, bias=True, quantizer=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.quantizer = quantizer

        self.weight_fp32 = nn.Parameter(torch.empty(out_features, in_features))
        # Init small so that initial snap lands near zero state
        nn.init.uniform_(self.weight_fp32, a=-0.02, b=0.02)

        if bias:
            self.bias_fp32 = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_fp32', None)

        self.register_buffer('weight_idx', None)

    def _ensure_indices(self):
        if self.weight_idx is None:
            with torch.no_grad():
                w_snap, idx = self.quantizer.snap_to_state(self.weight_fp32)
                self.weight_fp32.copy_(w_snap)
                self.weight_idx = idx

    def forward(self, x):
        self._ensure_indices()
        w_device = self.quantizer.indices_to_weight(self.weight_idx)
        # STE pass-through
        w = w_device + (self.weight_fp32 - self.weight_fp32.detach())
        b = self.bias_fp32 if self.use_bias and self.bias_fp32 is not None else None
        return torch.nn.functional.linear(x, w, b)

    @torch.no_grad()
    def project_after_step(self):
        """
        Snap-only projection:
        Snap FP32 weights to nearest device states and update indices accordingly.
        """
        self._ensure_indices()
        w_snapped, idx = self.quantizer.snap_to_state(self.weight_fp32)
        self.weight_fp32.copy_(w_snapped)
        self.weight_idx = idx


# =========================================================
# Simple CNN for MNIST
# =========================================================
class BaselineCNN_Device(nn.Module):
    """
    For MNIST (1x28x28):
      conv(1->32) -> BN(freeze) -> ReLU -> MaxPool
      conv(32->64) -> BN(freeze) -> ReLU -> MaxPool
      flatten -> FC(128) -> DeviceAwareLinear(num_classes)
    """
    def __init__(self, quantizer, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = DeviceAwareLinear(128, num_classes, bias=True, quantizer=quantizer)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# =========================================================
# Train and Eval
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device, log_every=200):
    model.train()
    correct = total = 0
    running_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for i, (x, y) in enumerate(pbar, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        # Grad clip for stability
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        optimizer.step()

        # Snap-only projection for the device-aware layer
        model.fc2.project_after_step()

        running_loss += float(loss.item()) * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if i % log_every == 0:
            g = model.fc2.weight_fp32.grad
            if g is not None and torch.isfinite(g).all():
                mean_g = g.abs().mean().item()
                max_g = g.abs().max().item()
            else:
                mean_g = max_g = float('nan')
            pbar.set_postfix(loss=running_loss/total,
                             acc=100.0*correct/total,
                             mean_g=f"{mean_g:.2e}",
                             max_g=f"{max_g:.2e}")
    return running_loss/total, 100.0*correct/total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = 0
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        running_loss += float(loss.item()) * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss/total, 100.0*correct/total


# =========================================================
# main
# =========================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LUTs from your paths
    pot_vals, dep_vals = load_memristor_values(
        "./value2/C1D1_potentiation.txt",
        "./value2/C1D1_depression.txt"
    )
    print(f"[LUT] potentiation: {pot_vals.shape} (min={pot_vals.min():.6f}, max={pot_vals.max():.6f})")
    if dep_vals is not None:
        print(f"[LUT] depression  : {dep_vals.shape} (min={dep_vals.min():.6f}, max={dep_vals.max():.6f})")
    else:
        print("[LUT] depression: None (using mirrored potentiation)")

    # Build states and quantizer
    states = build_states_from_lut(pot_vals, dep_vals, zero_center=True)
    quant = MemristorQuantizerFromTensor(states, device=device)

    # Data: MNIST sanity check
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, pin_memory=True)

    # Model and optimizer
    model = BaselineCNN_Device(quantizer=quant, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'weight_decay': 0.0},
        {'params': model.fc1.parameters(),       'weight_decay': 0.0},
        {'params': [model.fc2.weight_fp32, model.fc2.bias_fp32], 'weight_decay': 0.0},
    ], lr=1e-3)

    epochs = 5
    best_acc = 0.0
    prev_idx = None

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, log_every=200
        )

        # Index movement and histogram logging
        with torch.no_grad():
            model.fc2._ensure_indices()
            idx = model.fc2.weight_idx.view(-1)
            moved = None if prev_idx is None else (idx != prev_idx).sum().item()
            prev_idx = idx.clone()

            hist = torch.bincount(idx, minlength=quant.S).float()
            used_states = int((hist > 0).sum().item())
            topk = torch.topk(hist, k=min(5, quant.S)).indices.tolist()
            print(f"[Index] moved={moved if moved is not None else 'N/A'}, used_states={used_states}/{quant.S}")
            print(f"[Index] top states: {topk} (counts: {[int(hist[i].item()) for i in topk]})")

        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        print(f"[Train] loss={tr_loss:.4f}, acc={tr_acc:.2f}% | [Test] loss={te_loss:.4f}, acc={te_acc:.2f}%")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "best_mnist_memristor.pth")
            print("==> Best model saved!")

    print(f"\nBest Test Acc: {best_acc:.2f}%")

    # To switch to your MRI dataset later:
    # - Replace the MNIST loaders with your MRI loaders.
    # - Set num_classes=2 in BaselineCNN_Device.
    # - Keep BN affine disabled at first and use light augmentations.


if __name__ == "__main__":
    main()

