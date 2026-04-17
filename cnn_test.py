import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

# Config
DATA_DIR = "./data/test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load data
test_data = datasets.ImageFolder(DATA_DIR, transform=transform)
test_loader = DataLoader(test_data, batch_size=16)

# Load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("cnn_model.pth"))
model = model.to(DEVICE)
model.eval()

# Test
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")