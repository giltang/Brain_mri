# brain_mri_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import GPUtil

# ---- Configurations ----
data_root = './dataset/brain_mri'
pot_path = './value2/C1D1_potentiation.txt'
dep_path = './value2/C1D1_depression.txt'
num_classes = 4
batch_size = 16
epochs = 10
lr = 0.01
image_size = 224

# ---- Memristor LUT Loading ----
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

# ---- Memristor-aware Linear Layer ----
class MemristorLinear(nn.Module):
    def __init__(self, in_features, out_features, values):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.values = values.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.w_idx = nn.Parameter(torch.randint(0, len(values), (out_features, in_features), device=self.values.device), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        self.input_cache = x.detach()
        weight = self.values[self.w_idx]
        return nn.functional.linear(x, weight, self.bias)

    def index_step(self, grad_output, input_cache, lr):
        with torch.no_grad():
            grad_w = grad_output.T @ input_cache / grad_output.size(0)
            current_val = self.values[self.w_idx]
            updated_val = current_val - lr * grad_w

            new_idx = torch.empty_like(self.w_idx)
            for i in range(self.out_features):
                for j in range(self.in_features):
                    diff = torch.abs(self.values - updated_val[i, j])
                    new_idx[i, j] = torch.argmin(diff)
            self.w_idx.copy_(new_idx)

# ---- CNN + Memristor Model Definition ----
class MemristorCNN(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = MemristorLinear(32 * (image_size // 4) * (image_size // 4), 512, values)
        self.fc2 = MemristorLinear(512, num_classes, values)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        self.hidden_act = x.detach()
        x = self.fc2(x)
        return x

    def index_step(self, grad_output, input_cache, lr):
        self.fc2.index_step(grad_output, self.hidden_act, lr)
        grad_hidden = grad_output @ self.fc2.values[self.fc2.w_idx]
        self.fc1.index_step(grad_hidden, input_cache, lr)

# ---- Data Transforms ----
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ---- Datasets ----
train_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'Training'), transform=train_transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'Testing'), transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ---- Load LUT ----
values = load_values(pot_path, dep_path)

# ---- Model, Loss, Device ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MemristorCNN(values).to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.01

# ---- Training ----
best_acc = 0.0
for epoch in range(1, epochs + 1):
    model.train()
    correct = 0
    total = 0
    print("Start training loop")
    print(f"Total training images: {len(train_dataset)}")

    loop = tqdm(train_loader, desc=f"[Epoch {epoch}]")

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs_flat = model.flatten(model.conv(inputs))
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        model.zero_grad()
        grad_output = torch.autograd.grad(loss, outputs, retain_graph=True)[0].detach()
        model.index_step(grad_output, inputs_flat.detach(), lr)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        acc = correct / total * 100

        # GPU memory usage
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            mem_usage = f"{gpu.memoryUsed}/{gpu.memoryTotal} MB"
        else:
            mem_usage = "N/A"

        loop.set_postfix(loss=loss.item(), acc=acc, gpu_mem=mem_usage)

    # ---- Validation ----
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

