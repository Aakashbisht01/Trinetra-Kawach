# train_model.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    ...
    epoch_time = time.time() - start_time
    print(f"⏱ Elapsed time: {epoch_time/60:.2f} minutes")

# ------------------------
# 1. Setup
# ------------------------
DATA_DIR = "data/video_samples"   # contains 'normal' and 'distress' subfolders
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "video_action_model.pth")
LABELS_PATH = os.path.join(MODEL_DIR, "video_action_labels.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------
# 2. Data Preprocessing
# ------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),       # resize frames
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])  # normalize
])

# Torchvision automatically assigns labels based on subfolder names
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------
# 3. Model
# ------------------------
model = models.resnet18(pretrained=True)   # transfer learning
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)      # binary classification
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------------
# 4. Training Loop
# ------------------------
for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {total_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}%")

# ------------------------
# 5. Save Model + Labels
# ------------------------
torch.save(model.state_dict(), MODEL_PATH)

# Save label mapping (index -> class name)
label_map = {v: k for k, v in dataset.class_to_idx.items()}
# Example: {"0": "distress", "1": "normal"}

with open(LABELS_PATH, "w") as f:
    json.dump(label_map, f)

print(f"✅ Training complete. Model saved at {MODEL_PATH}")
print(f"✅ Labels saved at {LABELS_PATH}")
