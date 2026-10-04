import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = r"D:\FYDP\daraz_experiment\output_v4\features"
OUTPUT_DIR = r"D:\FYDP\daraz_experiment\output_v4\early_fusion"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8

TEXT_DIM = 768
IMAGE_DIM = 2048
PROJECTION_DIM = 256


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sys.stdout.reconfigure(encoding="utf-8")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("EARLY FUSION - MULTIMODAL FAKE REVIEW DETECTION")
print("=" * 70)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD FEATURES
# ============================================================

def load_split(split):

    text_path = os.path.join(
        FEATURE_DIR,
        f"{split}_text_features.npy"
    )

    image_path = os.path.join(
        FEATURE_DIR,
        f"{split}_image_features.npy"
    )

    metadata_path = os.path.join(
        FEATURE_DIR,
        f"{split}_metadata.csv"
    )

    text_features = np.load(text_path)
    image_features = np.load(image_path)
    metadata = pd.read_csv(metadata_path)

    print(f"\n{split.upper()} SET")
    print(f"Text features : {text_features.shape}")
    print(f"Image features: {image_features.shape}")
    print(f"Metadata      : {metadata.shape}")

    return text_features, image_features, metadata


train_text, train_image, train_meta = load_split("train")
val_text, val_image, val_meta = load_split("validation")
test_text, test_image, test_meta = load_split("test")


# ============================================================
# VERIFY DIMENSIONS
# ============================================================

assert train_text.shape[1] == TEXT_DIM
assert val_text.shape[1] == TEXT_DIM
assert test_text.shape[1] == TEXT_DIM

assert train_image.shape[1] == IMAGE_DIM
assert val_image.shape[1] == IMAGE_DIM
assert test_image.shape[1] == IMAGE_DIM

assert len(train_text) == len(train_image) == len(train_meta)
assert len(val_text) == len(val_image) == len(val_meta)
assert len(test_text) == len(test_image) == len(test_meta)


# ============================================================
# LABEL CONVERSION
# ============================================================

# def convert_labels(metadata):

#     labels = (
#         metadata["final_label"]
#         .astype(str)
#         .str.strip()
#         .str.upper()
#     )

#     mapping = {
#         "REAL": 0,
#         "FAKE": 1
#     }

#     converted = labels.map(mapping)

#     if converted.isna().any():
#         print("\nERROR: Unknown labels found:")
#         print(labels[converted.isna()].value_counts())
#         raise ValueError("Unknown label found.")

#     return converted.to_numpy(dtype=np.int64)
def convert_labels(metadata):

    labels = (
        metadata["final_label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    mapping = {
        "TRUE": 0,
        "FALSE": 1,
        "REAL": 0,
        "FAKE": 1
    }

    converted = labels.map(mapping)

    if converted.isna().any():
        print("\nERROR: Unknown labels found:")
        print(labels[converted.isna()].value_counts())
        raise ValueError("Unknown label found.")

    return converted.to_numpy(dtype=np.int64)


y_train = convert_labels(train_meta)
y_val = convert_labels(val_meta)
y_test = convert_labels(test_meta)


print("\nLABEL DISTRIBUTION")

print(
    "Train:",
    dict(zip(*np.unique(y_train, return_counts=True)))
)

print(
    "Validation:",
    dict(zip(*np.unique(y_val, return_counts=True)))
)

print(
    "Test:",
    dict(zip(*np.unique(y_test, return_counts=True)))
)


# ============================================================
# CONVERT TO PYTORCH TENSORS
# ============================================================

train_text = torch.tensor(train_text, dtype=torch.float32)
train_image = torch.tensor(train_image, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

val_text = torch.tensor(val_text, dtype=torch.float32)
val_image = torch.tensor(val_image, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

test_text = torch.tensor(test_text, dtype=torch.float32)
test_image = torch.tensor(test_image, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    train_text,
    train_image,
    y_train
)

val_dataset = TensorDataset(
    val_text,
    val_image,
    y_val
)

test_dataset = TensorDataset(
    test_text,
    test_image,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# EARLY FUSION MODEL
# ============================================================

class EarlyFusionModel(nn.Module):

    def __init__(
        self,
        text_dim=768,
        image_dim=2048,
        projection_dim=256
    ):

        super().__init__()

        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Image projection
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

    def forward(self, text_features, image_features):

        # Project each modality
        text_representation = self.text_projection(
            text_features
        )

        image_representation = self.image_projection(
            image_features
        )

        # EARLY FUSION
        fused_features = torch.cat(
            [text_representation, image_representation],
            dim=1
        )

        # Classification
        output = self.classifier(
            fused_features
        )

        return output


model = EarlyFusionModel(
    text_dim=TEXT_DIM,
    image_dim=IMAGE_DIM,
    projection_dim=PROJECTION_DIM
).to(DEVICE)


print("\nMODEL")
print(model)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(y_train.cpu().numpy())

class_weights = (
    len(y_train)
    / (len(class_counts) * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:")
print(class_weights)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for text_batch, image_batch, labels in train_loader:

        text_batch = text_batch.to(DEVICE)
        image_batch = image_batch.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            text_batch,
            image_batch
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


def evaluate(loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for text_batch, image_batch, labels in loader:

            text_batch = text_batch.to(DEVICE)
            image_batch = image_batch.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(
                text_batch,
                image_batch
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    return (
        total_loss / total,
        correct / total,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

history = []

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc, _, _ = evaluate(
        val_loader
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "validation_loss": val_loss,
        "validation_accuracy": val_acc
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # Early stopping based on validation loss
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            model.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "best_early_fusion_model.pt"
            )
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}."
            )

            break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        os.path.join(
            OUTPUT_DIR,
            "best_early_fusion_model.pt"
        ),
        map_location=DEVICE
    )
)

print(
    f"\nBest epoch: {best_epoch}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

val_loss, val_acc, val_labels, val_predictions = evaluate(
    val_loader
)

print("\nFINAL VALIDATION")
print(f"Loss     : {val_loss:.4f}")
print(f"Accuracy : {val_acc:.4f}")


# ============================================================
# FINAL TEST
# ============================================================

test_loss, test_acc, test_labels, test_predictions = evaluate(
    test_loader
)

print("\n" + "=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_results = test_meta.copy()

test_results["true_label"] = np.where(
    test_labels == 0,
    "REAL",
    "FAKE"
)

test_results["predicted_label"] = np.where(
    test_predictions == 0,
    "REAL",
    "FAKE"
)

test_results["correct"] = (
    test_labels == test_predictions
)

test_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print("\nEARLY FUSION TEST METRICS")
print("-" * 40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=["REAL", "FAKE"],
        zero_division=0
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "best_epoch": best_epoch
}

pd.DataFrame(
    [metrics]
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "early_fusion_metrics.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=["REAL", "FAKE"],
    columns=["REAL", "FAKE"]
)

cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    ),
    encoding="utf-8-sig"
)


print("\n" + "=" * 70)
print("EARLY FUSION COMPLETED")
print("=" * 70)

print(f"\nResults saved to:")
print(OUTPUT_DIR)