import pandas as pd
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "output_v4/reviews.csv"

OUTPUT_CSV = "output_v4/reviews_with_images.csv"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("CREATING MULTIMODAL REVIEW DATASET")
print("=" * 60)

print(f"\nReading: {INPUT_CSV}")

df = pd.read_csv(
    INPUT_CSV,
    encoding="utf-8-sig"
)

print(f"Total reviews in master dataset: {len(df)}")


# ============================================================
# CHECK REQUIRED COLUMN
# ============================================================

if "review_image_paths" not in df.columns:

    print(
        "\nERROR: 'review_image_paths' column "
        "was not found in the CSV."
    )

    print("\nAvailable columns:")

    for column in df.columns:
        print(f"  - {column}")

    raise SystemExit


# ============================================================
# SELECT REVIEWS WITH IMAGES
# ============================================================

has_image = (
    df["review_image_paths"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)

multimodal_df = df[has_image].copy()


# ============================================================
# SAVE DATASET
# ============================================================

multimodal_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

total_reviews = len(df)

reviews_with_images = len(multimodal_df)

reviews_without_images = (
    total_reviews - reviews_with_images
)


print("\n" + "=" * 60)
print("MULTIMODAL DATASET CREATED")
print("=" * 60)

print(
    f"\nTotal reviews           : "
    f"{total_reviews}"
)

print(
    f"Reviews with images     : "
    f"{reviews_with_images}"
)

print(
    f"Reviews without images  : "
    f"{reviews_without_images}"
)

if total_reviews > 0:

    percentage = (
        reviews_with_images /
        total_reviews
    ) * 100

    print(
        f"Percentage with images  : "
        f"{percentage:.2f}%"
    )


print(
    f"\nOutput file:"
)

print(
    f"{OUTPUT_CSV}"
)

print("\nOriginal master dataset was NOT modified.")

print("=" * 60)