import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ── Config ─────────────────────────────
DATA_DIR = "./data"
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ───────────────────────────────────────

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
train_data = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)
val_data   = datasets.ImageFolder(f"{DATA_DIR}/val", transform=transform)
print(train_data.classes)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE)

# Load EfficientNet
model = models.efficientnet_b0(pretrained=True)

# Modify final layer
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model = model.to(DEVICE)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ── Training Loop ───────────────────────
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), "efficientnet_model.pth")

print("✅ EfficientNet Training Complete!")