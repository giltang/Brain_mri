#n_mri_memristor_4class_multi_device_plot_6profiles.py
# Train the same snap-only memristor-aware model across multiple device profiles
# Dataset: Brain MRI 4 classes (glioma, meningioma, notumor, pituitary)
# Produces a matplotlib plot of validation accuracy per epoch per device.

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    if dep_vals is not None:
        states_neg = -torch.flip(dep_vals, dims=[0])
    else:
        states_neg = -torch.flip(pot_vals, dims=[0])

    states_pos = pot_vals.clone()
    states = torch.cat([states_neg, states_pos], dim=0)

    states = torch.unique(states)
    states, _ = torch.sort(states)

    if zero_center:
        states = states - states.mean()

    return states


# =========================================================
# Quantizer
# =========================================================
class MemristorQuantizerFromTensor:
    def __init__(self, states: torch.Tensor, device='cpu'):
        states = torch.unique(states).to(torch.float32).to(device)
        states, _ = torch.sort(states)
        self.states = states
        self.S = self.states.numel()

    def snap_to_state(self, w_fp32: torch.Tensor):
        w = w_fp32.unsqueeze(-1)
        d = torch.abs(w - self.states)
        idx = torch.argmin(d, dim=-1)
        w_snapped = self.states[idx]
        return w_snapped, idx

    def indices_to_weight(self, idx: torch.Tensor):
        return self.states[idx]


# =========================================================
# Device-aware Linear (snap-only projection)
# =========================================================
class DeviceAwareLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, quantizer=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.quantizer = quantizer

        self.weight_fp32 = nn.Parameter(torch.empty(out_features, in_features))
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
        self._ensure_indices()
        w_snapped, idx = self.quantizer.snap_to_state(self.weight_fp32)
        self.weight_fp32.copy_(w_snapped)
        self.weight_idx = idx


# =========================================================
# CNN for Brain MRI (grayscale, 4 classes)
# =========================================================
class BaselineCNN_Device(nn.Module):
    def __init__(self, quantizer, image_size=128, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (image_size // 4) * (image_size // 4), 128)
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
# Dataset helpers: filter and remap classes
# =========================================================
def filter_classes(dataset: datasets.ImageFolder, allowed):
    indices = [i for i, (_, label) in enumerate(dataset.samples)
               if dataset.classes[label] in allowed]
    return Subset(dataset, indices)

class RemappedSubset(Dataset):
    def __init__(self, subset: Subset, class_to_new_label: dict):
        self.subset = subset
        self.class_to_new_label = class_to_new_label

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, original_label = self.subset[idx]
        label_name = self.subset.dataset.classes[original_label]
        new_label = self.class_to_new_label[label_name]
        return x, new_label


# =========================================================
# Train and Eval
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device, log_every=50):
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

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        optimizer.step()
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
# One full train for a given device profile
# =========================================================
def run_for_device(device_name: str,
                   pot_path: str,
                   dep_path: str,
                   data_root: str,
                   image_size: int,
                   batch_size: int,
                   epochs: int,
                   lr: float,
                   device: torch.device):
    print(f"\n===== Device profile: {device_name} =====")
    pot_vals, dep_vals = load_memristor_values(pot_path, dep_path)
    print(f"[LUT] potentiation: {pot_vals.shape} (min={pot_vals.min():.6f}, max={pot_vals.max():.6f})")
    if dep_vals is not None:
        print(f"[LUT] depression  : {dep_vals.shape} (min={dep_vals.min():.6f}, max={dep_vals.max():.6f})")
    else:
        print("[LUT] depression: None (using mirrored potentiation)")

    states = build_states_from_lut(pot_vals, dep_vals, zero_center=True)
    quant = MemristorQuantizerFromTensor(states, device=device)

    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_full = datasets.ImageFolder(root=os.path.join(data_root, "Training"), transform=train_transform)
    val_full   = datasets.ImageFolder(root=os.path.join(data_root, "Testing"),  transform=val_transform)

    allowed = ["glioma", "meningioma", "notumor", "pituitary"]
    class_to_new = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}

    train_subset = filter_classes(train_full, allowed)
    val_subset   = filter_classes(val_full, allowed)
    train_dataset = RemappedSubset(train_subset, class_to_new)
    val_dataset   = RemappedSubset(val_subset, class_to_new)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, pin_memory=True)

    model = BaselineCNN_Device(quantizer=quant, image_size=image_size, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'weight_decay': 0.0},
        {'params': model.fc1.parameters(),       'weight_decay': 0.0},
        {'params': [model.fc2.weight_fp32, model.fc2.bias_fp32], 'weight_decay': 0.0},
    ], lr=lr)

    best_acc = 0.0
    best_path = f"best_mri_memristor_{device_name}_4cls.pth"
    prev_idx = None
    val_acc_per_epoch = []

    for epoch in range(1, epochs + 1):
        print(f"\n=== {device_name} Epoch {epoch}/{epochs} ===")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, log_every=50)

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

        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        val_acc_per_epoch.append(va_acc)
        print(f"[Train] loss={tr_loss:.4f}, acc={tr_acc:.2f}% | [Val] loss={va_loss:.4f}, acc={va_acc:.2f}%")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"==> Best model saved to {best_path}")

    print(f"Best Val Acc for {device_name}: {best_acc:.2f}%")
    return best_acc, best_path, val_acc_per_epoch


# =========================================================
# main: run all device profiles and compare, then plot
# =========================================================
def main():
    set_seed(42)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = "./dataset/brain_mri"
    image_size = 128
    batch_size = 16
    epochs = 30
    lr = 1e-3

    # Update paths here if your file names differ
    paths = {
        "C1D1": {
            "pot": "./value2/C1D1_potentiation.txt",
            "dep": "./value2/C1D1_depression.txt",
        },
        "C4D1": {
            "pot": "./value2/C4D1_potentiation.txt",
            "dep": "./value2/C4D1_depression.txt",
        },
        "C1D8": {
            "pot": "./value2/C1D8_potentiation.txt",
            "dep": "./value2/C1D8_depression.txt",
        },
        "Identical_spikes": {
            "pot": "./value2/Identical spikes_potentiation.txt",
            "dep": "./value2/Identical spikes_depression.txt",
        },
        "Non_spikes_1": {
            "pot": "./value2/Non spikes_1_potentiation.txt",
            "dep": "./value2/Non spikes_1_depression.txt",
        },
        "Non_spikes_2": {
            "pot": "./value2/Non spikes_2_potentiation.txt",
            "dep": "./value2/Non spikes_2_depression.txt",
        },
    }

    results = {}
    acc_curves = {}

    for name, p in paths.items():
        try:
            acc, path, curve = run_for_device(
                device_name=name,
                pot_path=p["pot"],
                dep_path=p["dep"],
                data_root=data_root,
                image_size=image_size,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                device=torch_device
            )
            results[name] = {"best_val_acc": acc, "ckpt": path}
            acc_curves[name] = curve
        except FileNotFoundError as e:
            print(f"[Skip {name}] {e}")
        except Exception as e:
            print(f"[Error {name}] {e}")

    print("\n===== Summary =====")
    for k, v in results.items():
        curve = acc_curves.get(k, [])
        mean_acc = np.mean(curve) if len(curve) > 0 else 0.0
        print(f"{k}: best_val_acc={v['best_val_acc']:.2f}%, mean_val_acc={mean_acc:.2f}% ckpt={v['ckpt']}")


    # Plot accuracy curves for all devices that ran successfully
    if len(acc_curves) > 0:
        plt.figure(figsize=(9, 6))
        for name, curve in acc_curves.items():
            if len(curve) > 0:
                plt.plot(range(1, len(curve) + 1), curve, marker='o', label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy (%)")
        plt.title("Validation Accuracy per Epoch for Each Device")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("device_accuracy_comparison.png", dpi=150)
        try:
            plt.show()
        except Exception:
            pass
    else:
        print("No curves to plot.")

if __name__ == "__main__":
    main()


