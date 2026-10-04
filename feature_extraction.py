import os
import ast
import json
import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms


# ============================================================
# WINDOWS TERMINAL UTF-8
# ============================================================

import sys
sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\FYDP\daraz_experiment"

SPLIT_DIR = os.path.join(
    BASE_DIR,
    "output_v4",
    "dataset_split"
)

IMAGE_BASE_DIR = os.path.join(
    BASE_DIR,
    "output_v4"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output_v4",
    "features"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Input datasets
SPLITS = {
    "train": os.path.join(SPLIT_DIR, "train.csv"),
    "validation": os.path.join(SPLIT_DIR, "validation.csv"),
    "test": os.path.join(SPLIT_DIR, "test.csv")
}


# ============================================================
# MODELS
# ============================================================

BANGLABERT_MODEL = "csebuetnlp/banglabert"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MULTIMODAL FEATURE EXTRACTION")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD BANGLABERT
# ============================================================

print("\n" + "=" * 70)
print("LOADING BANGLABERT")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    BANGLABERT_MODEL
)

text_model = AutoModel.from_pretrained(
    BANGLABERT_MODEL
)

text_model = text_model.to(DEVICE)
text_model.eval()

print("BanglaBERT loaded successfully.")


# ============================================================
# LOAD RESNET-50
# ============================================================

print("\n" + "=" * 70)
print("LOADING RESNET-50")
print("=" * 70)

weights = models.ResNet50_Weights.DEFAULT

resnet = models.resnet50(
    weights=weights
)

# Remove final classification layer
image_model = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

image_model = image_model.to(DEVICE)
image_model.eval()

image_transform = weights.transforms()

print("ResNet-50 loaded successfully.")


# ============================================================
# TEXT FEATURE EXTRACTION
# ============================================================

def extract_text_feature(text):

    if pd.isna(text):
        text = ""

    text = str(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = text_model(**inputs)

        # Mean pooling over token embeddings
        hidden_states = outputs.last_hidden_state

        attention_mask = inputs["attention_mask"]

        mask = attention_mask.unsqueeze(-1).expand(
            hidden_states.size()
        ).float()

        summed = torch.sum(
            hidden_states * mask,
            dim=1
        )

        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )

        embedding = summed / counts

    return embedding.squeeze(0).cpu().numpy()


# ============================================================
# IMAGE PATH PARSER
# ============================================================

def parse_image_paths(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    # Try Python-list format
    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(path)
                for path in parsed
                if str(path).strip()
            ]

    except Exception:
        pass

    # Try JSON-list format
    try:

        parsed = json.loads(value)

        if isinstance(parsed, list):
            return [
                str(path)
                for path in parsed
                if str(path).strip()
            ]

    except Exception:
        pass

    # Single path
    return [value]


# ============================================================
# RESOLVE IMAGE PATH
# ============================================================

def resolve_image_path(path):

    path = str(path).strip()

    # Absolute path
    if os.path.isabs(path):

        if os.path.exists(path):
            return path

    # Path relative to project directory
    candidate = os.path.join(
        BASE_DIR,
        path
    )

    if os.path.exists(candidate):
        return candidate

    # Path relative to output_v4
    candidate = os.path.join(
        IMAGE_BASE_DIR,
        path
    )

    if os.path.exists(candidate):
        return candidate

    return None


# ============================================================
# IMAGE FEATURE EXTRACTION
# ============================================================

def extract_image_feature(image_paths):

    features = []

    for path in image_paths:

        resolved_path = resolve_image_path(path)

        if resolved_path is None:
            continue

        try:

            image = Image.open(
                resolved_path
            ).convert("RGB")

            image_tensor = image_transform(
                image
            ).unsqueeze(0)

            image_tensor = image_tensor.to(
                DEVICE
            )

            with torch.no_grad():

                feature = image_model(
                    image_tensor
                )

            feature = feature.squeeze().cpu().numpy()

            features.append(feature)

        except Exception as e:

            print(
                f"\nWarning: Could not process image:"
                f"\n{resolved_path}"
                f"\nError: {e}"
            )

    # No usable image
    if len(features) == 0:

        return np.zeros(2048, dtype=np.float32)

    # Multiple images → average their features
    features = np.stack(features)

    return features.mean(
        axis=0
    ).astype(np.float32)


# ============================================================
# PROCESS ONE DATASET SPLIT
# ============================================================

def process_split(split_name, input_file):

    print("\n" + "=" * 70)
    print(f"PROCESSING {split_name.upper()} SET")
    print("=" * 70)

    df = pd.read_csv(
        input_file,
        encoding="utf-8-sig"
    )

    print(
        f"Reviews: {len(df)}"
    )

    text_features = []
    image_features = []

    review_ids = []
    labels = []

    # --------------------------------------------------------
    # PROCESS REVIEWS
    # --------------------------------------------------------

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"{split_name} features"
    ):

        review_id = row["review_id"]

        # Text
        text_feature = extract_text_feature(
            row["review_text"]
        )

        # Image
        image_paths = parse_image_paths(
            row["review_image_paths"]
        )

        image_feature = extract_image_feature(
            image_paths
        )

        review_ids.append(
            review_id
        )

        text_features.append(
            text_feature
        )

        image_features.append(
            image_feature
        )

        labels.append(
            row["final_label"]
        )

    # --------------------------------------------------------
    # CONVERT TO ARRAYS
    # --------------------------------------------------------

    text_features = np.vstack(
        text_features
    )

    image_features = np.vstack(
        image_features
    )

    # --------------------------------------------------------
    # SAVE FEATURES
    # --------------------------------------------------------

    text_output = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_text_features.npy"
    )

    image_output = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_image_features.npy"
    )

    metadata_output = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_metadata.csv"
    )

    np.save(
        text_output,
        text_features
    )

    np.save(
        image_output,
        image_features
    )

    metadata = pd.DataFrame({
        "review_id": review_ids,
        "final_label": labels
    })

    metadata.to_csv(
        metadata_output,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(
        f"\nText feature shape: "
        f"{text_features.shape}"
    )

    print(
        f"Image feature shape: "
        f"{image_features.shape}"
    )

    print(
        f"Saved: {text_output}"
    )

    print(
        f"Saved: {image_output}"
    )

    print(
        f"Saved: {metadata_output}"
    )


# ============================================================
# PROCESS ALL SPLITS
# ============================================================

for split_name, input_file in SPLITS.items():

    process_split(
        split_name,
        input_file
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 70)

print("\nFeature directory:")
print(OUTPUT_DIR)

print("\nExpected files:")

for split in [
    "train",
    "validation",
    "test"
]:

    print(
        f"\n{split}:"
    )

    print(
        f"  {split}_text_features.npy"
    )

    print(
        f"  {split}_image_features.npy"
    )

    print(
        f"  {split}_metadata.csv"
    )