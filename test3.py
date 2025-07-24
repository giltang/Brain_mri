import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np

# ---- 멤리스터 LUT 처리 ----
def load_values(pot_path, dep_path):
    def read_file(path):
        with open(path, 'r') as f:
            return [float(line.strip()) for line in f if line.strip()]

    p_vals = read_file(pot_path)
    d_vals = read_file(dep_path)

    combined = []
    for p in p_vals:
        for d in d_vals:
            combined.append(p - d)

    combined = torch.tensor(sorted(set(combined)), dtype=torch.float32)
    return combined

# ---- 멤리스터-aware Linear Layer ----
class MemristorLinear(nn.Module):
    def __init__(self, in_features, out_features, values):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.values = values
        self.w_idx = nn.Parameter(torch.randint(0, len(values), (out_features, in_features)), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        self.input_cache = x.detach()
        weight = self.values[self.w_idx]
        return F.linear(x, weight, self.bias)

    def index_step(self, grad_output, input_cache, lr):
        with torch.no_grad():
            grad_input = input_cache
            grad_output = grad_output
            grad_w = grad_output.T @ grad_input / grad_output.size(0)
            current_val = self.values[self.w_idx]
            updated_val = current_val - lr * grad_w

            # LUT 최적화: 반복문으로 argmin
            new_idx = torch.zeros_like(self.w_idx)
            for i in range(self.out_features):
                for j in range(self.in_features):
                    diff = torch.abs(self.values - updated_val[i, j])
                    new_idx[i, j] = torch.argmin(diff)
            self.w_idx.copy_(new_idx)

# ---- 멤리스터-aware 네트워크 ----
class MemristorNet(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.fc1 = MemristorLinear(112 * 112 * 3, 512, values)
        self.fc2 = MemristorLinear(512, 4, values)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        self.hidden_act = x.detach()
        x = self.fc2(x)
        return x

    def index_step(self, grad_output, input_cache, lr):
        self.fc2.index_step(grad_output, self.hidden_act, lr)
        grad_hidden = grad_output @ self.fc2.values[self.fc2.w_idx]
        self.fc1.index_step(grad_hidden, input_cache, lr)

# ---- 경로 설정 및 전처리 ----
base_dir = "./dataset/brain_mri"
train_dir = os.path.join(base_dir, "Training")
val_dir = os.path.join(base_dir, "Testing")

train_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = ImageFolder(root=train_dir, transform=train_transform)
val_dataset = ImageFolder(root=val_dir, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ---- 멤리스터 LUT 불러오기 ----
pot_path = "./value2/C1D1_potentiation.txt"
dep_path = "./value2/C1D1_depression.txt"
values = load_values(pot_path, dep_path)

# ---- 모델, 손실함수, 학습 ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MemristorNet(values).to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.01

best_acc = 0.0
for epoch in range(1, 11):
    model.train()
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"[Epoch {epoch}]")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs_flat = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        model.zero_grad()
        grad_output = torch.autograd.grad(loss, outputs, retain_graph=True)[0].detach()
        model.index_step(grad_output, inputs_flat.detach(), lr)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        acc = correct / total * 100
        loop.set_postfix(loss=loss.item(), acc=acc)

    # 검증
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total * 100
    print(f"Validation Accuracy: {val_acc:.2f}%")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_memristor_mri_model.pth")
        print("==> Best model saved!")
