import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = r"D:\FYDP\daraz_experiment\output_v4\multimodal_annotated.csv"

OUTPUT_DIR = r"D:\FYDP\daraz_experiment\output_v4\EDA_annotated_bangla"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE,encoding="cp1252")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("ANNOTATED BANGLA-ONLY REVIEWS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nNumber of reviews:", len(df))


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("1. COLUMN INFORMATION")
print("=" * 60)

print(df.info())

print("\nColumns:")
for col in df.columns:
    print("-", col)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("2. MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

missing_percentage = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": missing_percentage
})

print(missing_df)

missing_df.to_csv(
    os.path.join(OUTPUT_DIR, "missing_values.csv"),
    encoding="utf-8-sig"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("3. FINAL LABEL DISTRIBUTION")
print("=" * 60)

label_counts = df["final_label"].value_counts()

print(label_counts)

label_percentages = df["final_label"].value_counts(normalize=True) * 100

print("\nPercentage:")
print(label_percentages.round(2))

label_summary = pd.DataFrame({
    "count": label_counts,
    "percentage": label_percentages
})

label_summary.to_csv(
    os.path.join(OUTPUT_DIR, "label_distribution.csv"),
    encoding="utf-8-sig"
)


# ============================================================
# LABEL PLOT
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="final_label"
)

plt.title("REAL vs FAKE Review Distribution")
plt.xlabel("Review Label")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "01_label_distribution.png"),
    dpi=300
)

plt.close()


# ============================================================
# RATING DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("4. RATING DISTRIBUTION")
print("=" * 60)

print(df["rating"].value_counts().sort_index())

rating_label = pd.crosstab(
    df["rating"],
    df["final_label"]
)

print("\nRating vs Label:")
print(rating_label)

rating_label.to_csv(
    os.path.join(OUTPUT_DIR, "rating_vs_label.csv"),
    encoding="utf-8-sig"
)


# ============================================================
# RATING PLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="rating",
    hue="final_label"
)

plt.title("Rating Distribution by Review Label")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "02_rating_distribution.png"),
    dpi=300
)

plt.close()


# ============================================================
# VERIFIED PURCHASE
# ============================================================

print("\n" + "=" * 60)
print("5. VERIFIED PURCHASE")
print("=" * 60)

print(df["verified_purchase"].value_counts(dropna=False))

verified_label = pd.crosstab(
    df["verified_purchase"],
    df["final_label"]
)

print("\nVerified Purchase vs Label:")
print(verified_label)

verified_label.to_csv(
    os.path.join(OUTPUT_DIR, "verified_purchase_vs_label.csv"),
    encoding="utf-8-sig"
)


# ============================================================
# REVIEW IMAGE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("6. REVIEW IMAGE ANALYSIS")
print("=" * 60)


def has_review_image(value):
    if pd.isna(value):
        return False

    value = str(value).strip()

    if value == "":
        return False

    if value.lower() in ["[]", "nan", "none"]:
        return False

    return True


df["has_review_image"] = df["review_image_paths"].apply(
    has_review_image
)

print("\nReviews with images:")
print(df["has_review_image"].value_counts())

image_label = pd.crosstab(
    df["has_review_image"],
    df["final_label"]
)

print("\nImage availability vs Label:")
print(image_label)

image_label.to_csv(
    os.path.join(OUTPUT_DIR, "image_availability_vs_label.csv"),
    encoding="utf-8-sig"
)


# ============================================================
# IMAGE COUNT
# ============================================================

def count_images(value):

    if pd.isna(value):
        return 0

    value = str(value).strip()

    if value in ["", "[]", "nan", "none"]:
        return 0

    # Try to count paths separated by comma
    if value.startswith("[") and value.endswith("]"):
        try:
            import ast

            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                return len(parsed)

        except:
            pass

    # fallback
    return len(
        [x for x in value.split(",") if x.strip()]
    )


df["image_count"] = df["review_image_paths"].apply(
    count_images
)

print("\nImage count distribution:")
print(df["image_count"].value_counts().sort_index())


# ============================================================
# TEXT STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("7. REVIEW TEXT STATISTICS")
print("=" * 60)


df["review_text"] = df["review_text"].fillna("").astype(str)

# Character count
df["character_count"] = df["review_text"].apply(len)

# Word count
df["word_count"] = df["review_text"].apply(
    lambda x: len(x.split())
)

# Bangla character count
df["bangla_character_count"] = df["review_text"].apply(
    lambda x: len(
        re.findall(r"[\u0980-\u09FF]", x)
    )
)

print("\nCharacter count:")
print(df["character_count"].describe())

print("\nWord count:")
print(df["word_count"].describe())

print("\nBangla character count:")
print(df["bangla_character_count"].describe())


# ============================================================
# TEXT LENGTH BY LABEL
# ============================================================

print("\n" + "=" * 60)
print("8. TEXT LENGTH BY LABEL")
print("=" * 60)

text_stats = df.groupby("final_label")[
    ["character_count", "word_count", "bangla_character_count"]
].agg(
    ["count", "mean", "median", "min", "max"]
)

print(text_stats)

text_stats.to_csv(
    os.path.join(OUTPUT_DIR, "text_statistics_by_label.csv"),
    encoding="utf-8-sig"
)


# ============================================================
# WORD COUNT PLOT
# ============================================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="word_count",
    hue="final_label",
    kde=True,
    bins=30
)

plt.title("Review Word Count Distribution")
plt.xlabel("Word Count")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "03_word_count_distribution.png"),
    dpi=300
)

plt.close()


# ============================================================
# BOX PLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="final_label",
    y="word_count"
)

plt.title("Review Length by Label")
plt.xlabel("Review Label")
plt.ylabel("Word Count")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "04_word_count_by_label.png"),
    dpi=300
)

plt.close()


# ============================================================
# DUPLICATE REVIEW TEXT
# ============================================================

print("\n" + "=" * 60)
print("9. DUPLICATE REVIEW TEXT")
print("=" * 60)

duplicate_count = df["review_text"].duplicated(keep=False).sum()

print("Rows involved in duplicate review text:", duplicate_count)

duplicate_reviews = df[
    df["review_text"].duplicated(keep=False)
].sort_values("review_text")

duplicate_reviews.to_csv(
    os.path.join(OUTPUT_DIR, "duplicate_review_text.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PRODUCT DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("10. PRODUCT DISTRIBUTION")
print("=" * 60)

product_counts = df["product_id"].value_counts()

print("\nNumber of unique products:")
print(df["product_id"].nunique())

print("\nTop 20 products by review count:")
print(product_counts.head(20))

product_counts.to_csv(
    os.path.join(OUTPUT_DIR, "reviews_per_product.csv"),
    encoding="utf-8-sig"
)


# ============================================================
# MARKETPLACE DISTRIBUTION
# ============================================================

if "marketplace" in df.columns:

    print("\n" + "=" * 60)
    print("11. MARKETPLACE DISTRIBUTION")
    print("=" * 60)

    print(df["marketplace"].value_counts())

    marketplace_counts = df["marketplace"].value_counts()

    marketplace_counts.to_csv(
        os.path.join(OUTPUT_DIR, "marketplace_distribution.csv"),
        encoding="utf-8-sig"
    )


# ============================================================
# FINAL DATASET SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

print("Total reviews:", len(df))

print("Unique products:", df["product_id"].nunique())

print("REAL reviews:",
      (df["final_label"].astype(str).str.upper() == "REAL").sum())

print("FAKE reviews:",
      (df["final_label"].astype(str).str.upper() == "FAKE").sum())

print("Reviews with images:",
      df["has_review_image"].sum())

print("Reviews without images:",
      (~df["has_review_image"]).sum())

print("Average word count:",
      round(df["word_count"].mean(), 2))

print("Median word count:",
      round(df["word_count"].median(), 2))

print("Average image count:",
      round(df["image_count"].mean(), 2))


# ============================================================
# SAVE DATASET WITH EDA FEATURES
# ============================================================

df.to_csv(
    os.path.join(OUTPUT_DIR, "annotated_bangla_only_with_eda_features.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nEDA files saved in:")
print(OUTPUT_DIR)