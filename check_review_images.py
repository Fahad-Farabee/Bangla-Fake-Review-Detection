import pandas as pd

CSV_FILE = "output_v4/reviews.csv"

df = pd.read_csv(CSV_FILE)

# Count reviews with at least one image
reviews_with_images = (
    df["review_image_paths"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
    .sum()
)

# Count reviews without images
reviews_without_images = (
    df["review_image_paths"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

# Total reviews
total_reviews = len(df)

print("=" * 50)
print("REVIEW IMAGE ANALYSIS")
print("=" * 50)

print(f"Total reviews          : {total_reviews}")
print(f"Reviews with images    : {reviews_with_images}")
print(f"Reviews without images : {reviews_without_images}")

if total_reviews > 0:
    percentage = (
        reviews_with_images / total_reviews
    ) * 100

    print(
        f"Percentage with images : {percentage:.2f}%"
    )

print("=" * 50)