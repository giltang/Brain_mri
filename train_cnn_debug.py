# train_cnn_debug.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Simple CNN for 4-class classification
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load data using ImageFolder
def get_dataloaders(train_dir, test_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder(train_dir, transform=transform)
    test_set = datasets.ImageFolder(test_dir, transform=transform)

    print(f"Class mapping from ImageFolder: {train_set.class_to_idx}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

# Training loop
def train(model, loader, optimizer, criterion, device):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * labels.size(0)
    return running_loss / total, correct / total

# Evaluation loop
def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)
    return running_loss / total, correct / total

# Main training entry point
def main():
    train_dir = "./dataset/brain_mri/Training"
    test_dir = "./dataset/brain_mri/Testing"
    batch_size = 16
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(train_dir, test_dir, batch_size)
    model = SimpleCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Test Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%")

    torch.save(model.state_dict(), "debug_cnn.pth")

if __name__ == "__main__":
    main()

