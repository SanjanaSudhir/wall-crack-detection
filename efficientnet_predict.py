import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# ── Config ─────────────────────────────
MODEL_PATH = "efficientnet_model.pth"
IMAGE_PATH = "test_image2.jpg"   # change this
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ───────────────────────────────────────

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load model
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    outputs = model(image)
    _, pred = torch.max(outputs, 1)

# Labels (IMPORTANT: match ImageFolder order)
classes = ["cracked", "not_cracked"]

result = classes[pred.item()]

# Output
if result == "cracked":
    print("🟥 Result: CRACKED")
else:
    print("🟩 Result: NO CRACK")