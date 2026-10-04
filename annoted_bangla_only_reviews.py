import pandas as pd
import re

# ==========================================
# FILE PATHS
# ==========================================

INPUT_FILE = r"D:\FYDP\daraz_experiment\output_v4\annotated_reviews.csv"
OUTPUT_FILE = r"D:\FYDP\daraz_experiment\output_v4\annotated_bangla_only.csv"

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print("Original dataset shape:", df.shape)

# ==========================================
# REMOVE REVIEWS CONTAINING ENGLISH LETTERS
# ==========================================

def contains_english(text):
    if pd.isna(text):
        return False

    return bool(re.search(r"[A-Za-z]", str(text)))


english_mask = df["review_text"].apply(contains_english)

# Keep only reviews WITHOUT English letters
bangla_only_df = df[~english_mask].copy()

# ==========================================
# SAVE
# ==========================================

bangla_only_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# RESULTS
# ==========================================

print("\n===================================")
print("BANGLA-ONLY FILTERING COMPLETED")
print("===================================")

print("Original annotated reviews:", len(df))
print("Reviews containing English:", english_mask.sum())
print("Bangla-only reviews:", len(bangla_only_df))

print("\nRemoved:", len(df) - len(bangla_only_df))

print("\nSaved to:")
print(OUTPUT_FILE)