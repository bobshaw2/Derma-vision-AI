
import os
import copy
import random
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

from model import create_model, CLASS_CODES

# Configuration
DATA_DIR = "dataset"
CSV_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
MODEL_PATH = "models/skin_model.pth"

EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Read metadata
df = pd.read_csv(CSV_PATH)

image_lookup = {}

for folder in [
    "HAM10000_images_part_1",
    "HAM10000_images_part_2"
]:
    path = os.path.join(DATA_DIR, folder)

    for filename in os.listdir(path):
        if filename.lower().endswith(".jpg"):
            image_id = os.path.splitext(filename)[0]
            image_lookup[image_id] = os.path.join(
                path, filename
            )

df["path"] = df["image_id"].map(image_lookup)
df = df.dropna(subset=["path"]).reset_index(drop=True)

label_map = {
    code: i for i, code in enumerate(CLASS_CODES)
}

df["label"] = df["dx"].map(label_map)

# Split by lesion_id to prevent leakage
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=SEED
)

train_val_idx, test_idx = next(
    splitter.split(df, groups=df["lesion_id"])
)

train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=SEED
)

train_idx, val_idx = next(
    splitter2.split(
        train_val_df,
        groups=train_val_df["lesion_id"]
    )
)

train_df = train_val_df.iloc[train_idx].copy()
val_df = train_val_df.iloc[val_idx].copy()

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

# Image preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

class SkinDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)

        label = int(row["label"])

        return image, label

train_dataset = SkinDataset(train_df, train_transform)
val_dataset = SkinDataset(val_df, eval_transform)
test_dataset = SkinDataset(test_df, eval_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0
)

# Model
model = create_model(pretrained=True).to(device)

# Class weights for imbalance
counts = np.bincount(
    train_df["label"].to_numpy(),
    minlength=len(CLASS_CODES)
)

weights = len(train_df) / (
    len(CLASS_CODES) * np.maximum(counts, 1)
)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_accuracy = -1.0
best_state = None

# Training
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    model.eval()
    predictions = []
    actual = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            predictions.extend(predicted.cpu().tolist())
            actual.extend(labels.tolist())

    accuracy = accuracy_score(actual, predictions)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {running_loss / len(train_dataset):.4f} | "
        f"Validation Accuracy: {accuracy:.4f}",
        flush=True
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_state = copy.deepcopy(model.state_dict())

# Save best model
os.makedirs("models", exist_ok=True)

torch.save(
    {
        "state_dict": best_state,
        "class_codes": CLASS_CODES
    },
    MODEL_PATH
)

print("Best validation accuracy:", best_accuracy)
print("Model saved:", MODEL_PATH)

# Evaluate once on held-out test data
model.load_state_dict(best_state)
model.eval()

test_predictions = []
test_actual = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        test_predictions.extend(predicted.cpu().tolist())
        test_actual.extend(labels.tolist())

print("\nTest classification report:")
print(
    classification_report(
        test_actual,
        test_predictions,
        labels=list(range(len(CLASS_CODES))),
        target_names=CLASS_CODES,
        zero_division=0
    )
)

print("\nTraining and evaluation completed!")
