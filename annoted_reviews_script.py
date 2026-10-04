import pandas as pd
import os
# =========================
# FILE PATHS
# =========================
INPUT_FILE = r"D:/FYDP/daraz_experiment/output_v4/reviews_with_images.xlsx"
OUTPUT_FILE = r"D:/FYDP/daraz_experiment/output_v4/annotated_reviews.csv"

# =========================
# LOAD DATA
# =========================

df = pd.read_excel(INPUT_FILE)

print("Original dataset shape:", df.shape)

# =========================
# KEEP ONLY ANNOTATED ROWS
# =========================

# Remove rows where final_label is empty, NaN, or only whitespace
annotated_df = df[
    df["final_label"].notna() &
    (df["final_label"].astype(str).str.strip() != "")
].copy()

# =========================
# SAVE
# =========================

annotated_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# =========================
# SUMMARY
# =========================

print("\nFiltering completed.")
print("Original reviews:", len(df))
print("Annotated reviews:", len(annotated_df))
print("Unannotated reviews removed:", len(df) - len(annotated_df))

print("\nAnnotation distribution:")
print(annotated_df["final_label"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)