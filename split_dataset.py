import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"D:\FYDP\daraz_experiment\output_v4\multimodal_annotated.xlsx"

OUTPUT_DIR = r"D:\FYDP\daraz_experiment\output_v4\dataset_split"

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.csv")
VALIDATION_FILE = os.path.join(OUTPUT_DIR, "validation.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "split_report.txt")

RANDOM_STATE = 42


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_excel(INPUT_FILE)

print(f"Total reviews loaded: {len(df)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "review_id",
    "review_text",
    "review_image_paths",
    "final_label"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CHECK REVIEW ID
# ============================================================

print("\nChecking review IDs...")

if df["review_id"].isna().any():
    raise ValueError(
        "Some reviews have missing review_id values."
    )

duplicate_ids = df[
    df["review_id"].duplicated(keep=False)
]

if len(duplicate_ids) > 0:

    print("\nDuplicate review IDs found:")
    print(
        duplicate_ids[
            ["review_id"]
        ].sort_values("review_id")
    )

    raise ValueError(
        "Duplicate review_id found. "
        "Each review must have a unique review_id."
    )

print("✓ All review IDs are unique.")


# ============================================================
# NORMALIZE LABELS FOR SPLITTING
# ============================================================

print("\nChecking labels...")

df["_split_label"] = (
    df["final_label"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Convert TRUE → REAL
df["_split_label"] = df["_split_label"].replace({
    "TRUE": "REAL"
})

valid_labels = {"REAL", "FAKE"}

invalid_labels = set(
    df["_split_label"].unique()
) - valid_labels

if invalid_labels:
    raise ValueError(
        f"Invalid labels found: {invalid_labels}\n"
        f"Expected only REAL and FAKE."
    )

print("\nOverall label distribution:")
print(df["_split_label"].value_counts())

real_count = (df["_split_label"] == "REAL").sum()
fake_count = (df["_split_label"] == "FAKE").sum()

overall_ratio = real_count / fake_count

print(f"\nOverall REAL: {real_count}")
print(f"Overall FAKE: {fake_count}")
print(f"Overall REAL:FAKE = {overall_ratio:.2f}:1")


# ============================================================
# DETERMINE SPLIT SIZES
# ============================================================

total = len(df)

test_size = round(total * 0.10)
validation_size = round(total * 0.10)
train_size = total - test_size - validation_size

print("\n" + "=" * 70)
print("TARGET SPLIT SIZE")
print("=" * 70)

print(f"Total      : {total}")
print(f"Train      : {train_size}")
print(f"Validation : {validation_size}")
print(f"Test       : {test_size}")


# ============================================================
# STEP 1: CREATE TEST SET
# ============================================================

print("\n" + "=" * 70)
print("CREATING TEST SET")
print("=" * 70)

train_validation_df, test_df = train_test_split(
    df,
    test_size=test_size,
    stratify=df["_split_label"],
    random_state=RANDOM_STATE
)


# ============================================================
# STEP 2: CREATE VALIDATION SET
# ============================================================

print("\n" + "=" * 70)
print("CREATING VALIDATION SET")
print("=" * 70)

validation_fraction = validation_size / len(train_validation_df)

train_df, validation_df = train_test_split(
    train_validation_df,
    test_size=validation_fraction,
    stratify=train_validation_df["_split_label"],
    random_state=RANDOM_STATE
)


# ============================================================
# REMOVE INTERNAL SPLIT COLUMN
# ============================================================

train_df = train_df.drop(columns=["_split_label"])
validation_df = validation_df.drop(columns=["_split_label"])
test_df = test_df.drop(columns=["_split_label"])


# ============================================================
# RESET INDEX
# ============================================================

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# VERIFY TOTAL NUMBER OF REVIEWS
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING SPLIT")
print("=" * 70)

print(f"Train reviews      : {len(train_df)}")
print(f"Validation reviews : {len(validation_df)}")
print(f"Test reviews       : {len(test_df)}")

total_after_split = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

if total_after_split != total:
    raise ValueError(
        "ERROR: Number of reviews changed during splitting!"
    )

print("✓ Total number of reviews preserved.")


# ============================================================
# CHECK REVIEW ID OVERLAP
# ============================================================

train_ids = set(train_df["review_id"])
validation_ids = set(validation_df["review_id"])
test_ids = set(test_df["review_id"])

train_validation_overlap = (
    train_ids & validation_ids
)

train_test_overlap = (
    train_ids & test_ids
)

validation_test_overlap = (
    validation_ids & test_ids
)

print("\nReview ID overlap checks:")

print(
    f"Train ∩ Validation: "
    f"{len(train_validation_overlap)}"
)

print(
    f"Train ∩ Test: "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test: "
    f"{len(validation_test_overlap)}"
)

if (
    train_validation_overlap
    or train_test_overlap
    or validation_test_overlap
):
    raise ValueError(
        "ERROR: Review ID overlap detected!"
    )

print("✓ No review appears in more than one split.")


# ============================================================
# FUNCTION TO CALCULATE SPLIT STATISTICS
# ============================================================

def get_statistics(data):

    labels = (
        data["final_label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"TRUE": "REAL"})
    )

    real = (labels == "REAL").sum()
    fake = (labels == "FAKE").sum()

    total = len(data)

    if fake > 0:
        ratio = real / fake
    else:
        ratio = float("inf")

    return {
        "total": total,
        "percentage": (total / len(df)) * 100,
        "real": real,
        "fake": fake,
        "ratio": ratio
    }


train_stats = get_statistics(train_df)
validation_stats = get_statistics(validation_df)
test_stats = get_statistics(test_df)


# ============================================================
# PRINT FINAL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET DISTRIBUTION")
print("=" * 70)

print(
    f"\n{'Dataset':<15}"
    f"{'Total':<10}"
    f"{'Percent':<12}"
    f"{'REAL':<10}"
    f"{'FAKE':<10}"
    f"{'REAL:FAKE'}"
)

print("-" * 70)

for name, stats in [
    ("Train", train_stats),
    ("Validation", validation_stats),
    ("Test", test_stats)
]:

    print(
        f"{name:<15}"
        f"{stats['total']:<10}"
        f"{stats['percentage']:<12.2f}"
        f"{stats['real']:<10}"
        f"{stats['fake']:<10}"
        f"{stats['ratio']:.2f}:1"
    )


# ============================================================
# SAVE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("SAVING DATASETS")
print("=" * 70)

train_df.to_csv(
    TRAIN_FILE,
    index=False,
    encoding="utf-8-sig"
)

validation_df.to_csv(
    VALIDATION_FILE,
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    TEST_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"✓ Train saved      : {TRAIN_FILE}")
print(f"✓ Validation saved : {VALIDATION_FILE}")
print(f"✓ Test saved       : {TEST_FILE}")


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 70 + "\n")
    f.write("MULTIMODAL FAKE REVIEW DATASET SPLIT REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Total reviews: {total}\n\n")

    f.write("SPLIT DISTRIBUTION\n")
    f.write("-" * 70 + "\n")

    for name, stats in [
        ("Train", train_stats),
        ("Validation", validation_stats),
        ("Test", test_stats)
    ]:

        f.write(
            f"{name}:\n"
            f"  Total      : {stats['total']}\n"
            f"  Percentage : {stats['percentage']:.2f}%\n"
            f"  REAL       : {stats['real']}\n"
            f"  FAKE       : {stats['fake']}\n"
            f"  REAL:FAKE  : {stats['ratio']:.2f}:1\n\n"
        )

    f.write("REVIEW ID OVERLAP CHECK\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Train ∩ Validation : "
        f"{len(train_validation_overlap)}\n"
    )

    f.write(
        f"Train ∩ Test       : "
        f"{len(train_test_overlap)}\n"
    )

    f.write(
        f"Validation ∩ Test  : "
        f"{len(validation_test_overlap)}\n"
    )

    f.write("\n")

    if not (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
    ):
        f.write(
            "RESULT: No review_id appears in "
            "more than one dataset.\n"
        )

print(f"✓ Report saved      : {REPORT_FILE}")


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("SPLITTING COMPLETED SUCCESSFULLY")
print("=" * 70)