import os
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = r"D:\FYDP\daraz_experiment\output_v4\features"
OUTPUT_DIR = r"D:\FYDP\daraz_experiment\output_v4\late_fusion"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8

TEXT_DIM = 768
IMAGE_DIM = 2048

HIDDEN_DIM = 128


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sys.stdout.reconfigure(encoding="utf-8")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("LATE FUSION - MULTIMODAL FAKE REVIEW DETECTION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


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
    print(
        f"Text features : {text_features.shape}"
    )
    print(
        f"Image features: {image_features.shape}"
    )
    print(
        f"Metadata      : {metadata.shape}"
    )

    return (
        text_features,
        image_features,
        metadata
    )


train_text, train_image, train_meta = load_split(
    "train"
)

val_text, val_image, val_meta = load_split(
    "validation"
)

test_text, test_image, test_meta = load_split(
    "test"
)


# ============================================================
# VERIFY FEATURE DIMENSIONS
# ============================================================

assert train_text.shape[1] == TEXT_DIM
assert val_text.shape[1] == TEXT_DIM
assert test_text.shape[1] == TEXT_DIM

assert train_image.shape[1] == IMAGE_DIM
assert val_image.shape[1] == IMAGE_DIM
assert test_image.shape[1] == IMAGE_DIM


# ============================================================
# VERIFY SAMPLE ALIGNMENT
# ============================================================

assert len(train_text) == len(train_image)
assert len(train_text) == len(train_meta)

assert len(val_text) == len(val_image)
assert len(val_text) == len(val_meta)

assert len(test_text) == len(test_image)
assert len(test_text) == len(test_meta)


# ============================================================
# LABEL CONVERSION
# ============================================================

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

        print(
            "\nERROR: Unknown labels found:"
        )

        print(
            labels[converted.isna()].value_counts()
        )

        raise ValueError(
            "Unknown label found."
        )

    return converted.to_numpy(
        dtype=np.int64
    )


y_train = convert_labels(train_meta)
y_val = convert_labels(val_meta)
y_test = convert_labels(test_meta)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nLABEL DISTRIBUTION")

print(
    "Train:",
    dict(
        zip(
            *np.unique(
                y_train,
                return_counts=True
            )
        )
    )
)

print(
    "Validation:",
    dict(
        zip(
            *np.unique(
                y_val,
                return_counts=True
            )
        )
    )
)

print(
    "Test:",
    dict(
        zip(
            *np.unique(
                y_test,
                return_counts=True
            )
        )
    )
)


# ============================================================
# CONVERT FEATURES TO TENSORS
# ============================================================

train_text = torch.tensor(
    train_text,
    dtype=torch.float32
)

train_image = torch.tensor(
    train_image,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)


val_text = torch.tensor(
    val_text,
    dtype=torch.float32
)

val_image = torch.tensor(
    val_image,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)


test_text = torch.tensor(
    test_text,
    dtype=torch.float32
)

test_image = torch.tensor(
    test_image,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


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
# TEXT-ONLY CLASSIFIER
# ============================================================

class TextClassifier(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=128
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                hidden_dim,
                2
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# IMAGE-ONLY CLASSIFIER
# ============================================================

class ImageClassifier(nn.Module):

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=128
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                hidden_dim,
                2
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# CREATE MODELS
# ============================================================

text_model = TextClassifier(
    input_dim=TEXT_DIM,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)

image_model = ImageClassifier(
    input_dim=IMAGE_DIM,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)


print("\nTEXT CLASSIFIER")
print(text_model)

print("\nIMAGE CLASSIFIER")
print(image_model)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_train.cpu().numpy()
)

class_weights = (
    len(y_train)
    / (
        len(class_counts)
        * class_counts
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:")
print(class_weights)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

criterion_text = nn.CrossEntropyLoss(
    weight=class_weights
)

criterion_image = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZERS
# ============================================================

text_optimizer = torch.optim.AdamW(
    text_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

image_optimizer = torch.optim.AdamW(
    image_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    text_model.train()
    image_model.train()

    total_loss = 0
    text_correct = 0
    image_correct = 0
    total = 0

    for (
        text_batch,
        image_batch,
        labels
    ) in train_loader:

        text_batch = text_batch.to(
            DEVICE
        )

        image_batch = image_batch.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        # ----------------------------------------------------
        # TEXT MODEL
        # ----------------------------------------------------

        text_optimizer.zero_grad()

        text_outputs = text_model(
            text_batch
        )

        text_loss = criterion_text(
            text_outputs,
            labels
        )

        text_loss.backward()

        text_optimizer.step()


        # ----------------------------------------------------
        # IMAGE MODEL
        # ----------------------------------------------------

        image_optimizer.zero_grad()

        image_outputs = image_model(
            image_batch
        )

        image_loss = criterion_image(
            image_outputs,
            labels
        )

        image_loss.backward()

        image_optimizer.step()


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        batch_loss = (
            text_loss.item()
            + image_loss.item()
        ) / 2

        total_loss += (
            batch_loss
            * labels.size(0)
        )

        text_predictions = torch.argmax(
            text_outputs,
            dim=1
        )

        image_predictions = torch.argmax(
            image_outputs,
            dim=1
        )

        text_correct += (
            text_predictions == labels
        ).sum().item()

        image_correct += (
            image_predictions == labels
        ).sum().item()

        total += labels.size(0)


    return (
        total_loss / total,
        text_correct / total,
        image_correct / total
    )


# ============================================================
# LATE FUSION EVALUATION
# ============================================================

def evaluate(loader):

    text_model.eval()
    image_model.eval()

    total = 0

    text_correct = 0
    image_correct = 0
    fused_correct = 0

    all_labels = []

    all_text_predictions = []
    all_image_predictions = []
    all_fused_predictions = []

    with torch.no_grad():

        for (
            text_batch,
            image_batch,
            labels
        ) in loader:

            text_batch = text_batch.to(
                DEVICE
            )

            image_batch = image_batch.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            # ------------------------------------------------
            # INDIVIDUAL MODALITY PREDICTIONS
            # ------------------------------------------------

            text_logits = text_model(
                text_batch
            )

            image_logits = image_model(
                image_batch
            )


            # Convert logits to probabilities

            text_probabilities = torch.softmax(
                text_logits,
                dim=1
            )

            image_probabilities = torch.softmax(
                image_logits,
                dim=1
            )


            # ------------------------------------------------
            # LATE FUSION
            # ------------------------------------------------

            fused_probabilities = (
                text_probabilities
                + image_probabilities
            ) / 2.0


            # Final prediction

            text_predictions = torch.argmax(
                text_probabilities,
                dim=1
            )

            image_predictions = torch.argmax(
                image_probabilities,
                dim=1
            )

            fused_predictions = torch.argmax(
                fused_probabilities,
                dim=1
            )


            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            text_correct += (
                text_predictions == labels
            ).sum().item()

            image_correct += (
                image_predictions == labels
            ).sum().item()

            fused_correct += (
                fused_predictions == labels
            ).sum().item()

            total += labels.size(0)


            all_labels.extend(
                labels.cpu().numpy()
            )

            all_text_predictions.extend(
                text_predictions.cpu().numpy()
            )

            all_image_predictions.extend(
                image_predictions.cpu().numpy()
            )

            all_fused_predictions.extend(
                fused_predictions.cpu().numpy()
            )


    return (
        text_correct / total,
        image_correct / total,
        fused_correct / total,
        np.array(all_labels),
        np.array(all_text_predictions),
        np.array(all_image_predictions),
        np.array(all_fused_predictions)
    )


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = -1
best_epoch = 0
patience_counter = 0

history = []

print("\n" + "=" * 70)
print("TRAINING LATE FUSION")
print("=" * 70)


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss, train_text_acc, train_image_acc = (
        train_one_epoch()
    )


    (
        val_text_acc,
        val_image_acc,
        val_fused_acc,
        _,
        _,
        _,
        _
    ) = evaluate(
        val_loader
    )


    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "train_text_accuracy":
            train_text_acc,

        "train_image_accuracy":
            train_image_acc,

        "validation_text_accuracy":
            val_text_acc,

        "validation_image_accuracy":
            val_image_acc,

        "validation_fused_accuracy":
            val_fused_acc
    })


    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Text Train: {train_text_acc:.4f} | "
        f"Image Train: {train_image_acc:.4f} | "
        f"Val Text: {val_text_acc:.4f} | "
        f"Val Image: {val_image_acc:.4f} | "
        f"Val Fusion: {val_fused_acc:.4f}"
    )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if val_fused_acc > best_val_accuracy:

        best_val_accuracy = val_fused_acc

        best_epoch = epoch

        patience_counter = 0


        torch.save(
            {
                "text_model":
                    text_model.state_dict(),

                "image_model":
                    image_model.state_dict(),

                "epoch":
                    epoch,

                "validation_accuracy":
                    val_fused_acc
            },

            os.path.join(
                OUTPUT_DIR,
                "best_late_fusion_model.pt"
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

history_df = pd.DataFrame(
    history
)

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

checkpoint = torch.load(
    os.path.join(
        OUTPUT_DIR,
        "best_late_fusion_model.pt"
    ),
    map_location=DEVICE
)

text_model.load_state_dict(
    checkpoint["text_model"]
)

image_model.load_state_dict(
    checkpoint["image_model"]
)

best_epoch = checkpoint["epoch"]

print(
    f"\nBest epoch: {best_epoch}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

(
    val_text_acc,
    val_image_acc,
    val_fused_acc,
    val_labels,
    val_text_predictions,
    val_image_predictions,
    val_fused_predictions
) = evaluate(
    val_loader
)


print("\nFINAL VALIDATION")
print(
    f"Text Accuracy   : {val_text_acc:.4f}"
)

print(
    f"Image Accuracy  : {val_image_acc:.4f}"
)

print(
    f"Fusion Accuracy : {val_fused_acc:.4f}"
)


# ============================================================
# FINAL TEST
# ============================================================

(
    test_text_acc,
    test_image_acc,
    test_fused_acc,
    test_labels,
    test_text_predictions,
    test_image_predictions,
    test_fused_predictions
) = evaluate(
    test_loader
)


print("\n" + "=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)

print(
    f"Text Accuracy   : {test_text_acc:.4f}"
)

print(
    f"Image Accuracy  : {test_image_acc:.4f}"
)

print(
    f"Fusion Accuracy : {test_fused_acc:.4f}"
)


# ============================================================
# FUSION METRICS
# ============================================================

accuracy = accuracy_score(
    test_labels,
    test_fused_predictions
)

precision = precision_score(
    test_labels,
    test_fused_predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    test_fused_predictions,
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_fused_predictions,
    zero_division=0
)

cm = confusion_matrix(
    test_labels,
    test_fused_predictions
)


print("\nLATE FUSION TEST METRICS")
print("-" * 40)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)


print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        test_labels,
        test_fused_predictions,
        target_names=[
            "REAL",
            "FAKE"
        ],
        zero_division=0
    )
)


# ============================================================
# INDIVIDUAL MODALITY METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions
):

    return {

        "accuracy":
            accuracy_score(
                labels,
                predictions
            ),

        "precision":
            precision_score(
                labels,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                labels,
                predictions,
                zero_division=0
            ),

        "f1_score":
            f1_score(
                labels,
                predictions,
                zero_division=0
            )
    }


text_metrics = calculate_metrics(
    test_labels,
    test_text_predictions
)

image_metrics = calculate_metrics(
    test_labels,
    test_image_predictions
)

fusion_metrics = calculate_metrics(
    test_labels,
    test_fused_predictions
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison = pd.DataFrame([

    {
        "model": "Text Only",
        **text_metrics
    },

    {
        "model": "Image Only",
        **image_metrics
    },

    {
        "model": "Late Fusion",
        **fusion_metrics
    }

])


comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "late_fusion_comparison.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE FUSION CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "REAL",
        "FAKE"
    ],
    columns=[
        "REAL",
        "FAKE"
    ]
)

cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    ),
    encoding="utf-8-sig"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_results = test_meta.copy()

test_results[
    "text_prediction"
] = np.where(
    test_text_predictions == 0,
    "REAL",
    "FAKE"
)

test_results[
    "image_prediction"
] = np.where(
    test_image_predictions == 0,
    "REAL",
    "FAKE"
)

test_results[
    "fused_prediction"
] = np.where(
    test_fused_predictions == 0,
    "REAL",
    "FAKE"
)

test_results[
    "true_label"
] = np.where(
    test_labels == 0,
    "REAL",
    "FAKE"
)

test_results[
    "fusion_correct"
] = (
    test_labels
    == test_fused_predictions
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
# SAVE FINAL METRICS
# ============================================================

final_metrics = pd.DataFrame([

    {
        "model": "Text Only",
        **text_metrics
    },

    {
        "model": "Image Only",
        **image_metrics
    },

    {
        "model": "Late Fusion",
        **fusion_metrics
    }

])

final_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_metrics.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("LATE FUSION COMPLETED")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)