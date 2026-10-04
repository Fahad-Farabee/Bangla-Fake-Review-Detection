import os
import re
import csv
import json
import time
import requests
import sys


# ============================================================
# WINDOWS UTF-8 SUPPORT
# ============================================================

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "output_v4"

INPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "reviews.csv"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "reviews_with_dates.csv"
)

PRODUCT_MEMORY_FILE = os.path.join(
    OUTPUT_DIR,
    "products.json"
)

REVIEW_MEMORY_FILE = os.path.join(
    OUTPUT_DIR,
    "reviews.json"
)

PAGE_SIZE = 5
REQUEST_DELAY = 1


# ============================================================
# DARAZ SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.daraz.com.bd/"
})


# ============================================================
# LOAD JSON
# ============================================================

def load_json(file_path):

    if not os.path.exists(file_path):

        print(
            f"ERROR: File not found: "
            f"{file_path}"
        )

        return None

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            f"ERROR reading {file_path}: {e}"
        )

        return None


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(file_path):

    if not os.path.exists(file_path):

        print(
            f"ERROR: CSV not found: "
            f"{file_path}"
        )

        return [], []

    try:

        with open(
            file_path,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            reader = csv.DictReader(f)

            rows = list(reader)

            columns = reader.fieldnames or []

            return rows, columns

    except Exception as e:

        print(
            f"ERROR reading CSV: {e}"
        )

        return [], []


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    file_path,
    rows,
    columns
):

    try:

        with open(
            file_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=columns
            )

            writer.writeheader()

            writer.writerows(rows)

        return True

    except Exception as e:

        print(
            f"ERROR saving CSV: {e}"
        )

        return False


# ============================================================
# GET REVIEW PAGE
# ============================================================

def get_review_page(
    item_id,
    page_no
):

    api_url = (
        "https://my.daraz.com.bd/"
        "pdp/review/getReviewList"
    )

    params = {
        "itemId": item_id,
        "pageSize": PAGE_SIZE,
        "filter": 0,
        "sort": 0,
        "pageNo": page_no
    }

    try:

        response = session.get(
            api_url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:

        print(
            f"ERROR retrieving "
            f"item {item_id}, "
            f"page {page_no}: {e}"
        )

        return None


# ============================================================
# BUILD REVIEW KEY
# ============================================================

def get_review_key(
    item_id,
    review
):

    review_rate_id = review.get(
        "reviewRateId"
    )

    if review_rate_id:

        return (
            f"{item_id}:"
            f"{review_rate_id}"
        )

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)

    print(
        "ADD DARAZ PURCHASE/REVIEW DATES"
    )

    print("=" * 65)


    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(INPUT_CSV):

        print(
            f"\nERROR: {INPUT_CSV} does not exist."
        )

        return


    if not os.path.exists(
        REVIEW_MEMORY_FILE
    ):

        print(
            f"\nERROR: {REVIEW_MEMORY_FILE} "
            f"does not exist."
        )

        return


    if not os.path.exists(
        PRODUCT_MEMORY_FILE
    ):

        print(
            f"\nERROR: {PRODUCT_MEMORY_FILE} "
            f"does not exist."
        )

        return


    # ========================================================
    # LOAD DATA
    # ========================================================

    rows, original_columns = load_csv(
        INPUT_CSV
    )

    product_memory = load_json(
        PRODUCT_MEMORY_FILE
    )

    review_memory = load_json(
        REVIEW_MEMORY_FILE
    )


    if product_memory is None:
        return

    if review_memory is None:
        return


    print(
        f"\nCSV rows: {len(rows)}"
    )

    print(
        f"Products in memory: "
        f"{len(product_memory)}"
    )

    print(
        f"Reviews in memory: "
        f"{len(review_memory)}"
    )


    # ========================================================
    # CREATE REVERSE LOOKUP
    #
    # review_memory:
    #
    # DarazItemID:ReviewRateID
    #       ↓
    #       RV01
    #
    # We need:
    #
    # RV01
    #   ↓
    # DarazItemID:ReviewRateID
    # ========================================================

    review_id_to_key = {}

    for review_key, review_info in review_memory.items():

        if not isinstance(
            review_info,
            dict
        ):

            continue

        review_id = review_info.get(
            "review_id"
        )

        if review_id:

            review_id_to_key[
                review_id
            ] = review_key


    # ========================================================
    # CREATE MAP:
    #
    # review_id → CSV row
    # ========================================================

    row_by_review_id = {}

    for row in rows:

        review_id = row.get(
            "review_id"
        )

        if review_id:

            row_by_review_id[
                review_id
            ] = row


    # ========================================================
    # INITIALIZE NEW COLUMNS
    # ========================================================

    for row in rows:

        row["purchase_date"] = ""

        row["review_date"] = ""


    # ========================================================
    # CREATE DATE MAP
    # ========================================================

    date_map = {}

    matched_reviews = 0
    unmatched_reviews = 0

    products_processed = 0


    # ========================================================
    # PROCESS EACH PRODUCT
    # ========================================================

    for item_id, product_info in product_memory.items():

        if not isinstance(
            product_info,
            dict
        ):

            continue


        product_id = product_info.get(
            "product_id",
            ""
        )


        print("\n" + "=" * 65)

        print(
            f"PRODUCT: {product_id}"
        )

        print(
            f"Daraz item ID: {item_id}"
        )

        print("=" * 65)


        # ----------------------------------------------------
        # FIND REVIEWS BELONGING TO THIS PRODUCT
        # ----------------------------------------------------

        product_review_keys = []

        for review_key, review_info in review_memory.items():

            if not isinstance(
                review_info,
                dict
            ):

                continue

            if review_info.get(
                "product_id"
            ) == product_id:

                product_review_keys.append(
                    review_key
                )


        print(
            f"Reviews remembered for product: "
            f"{len(product_review_keys)}"
        )


        # ----------------------------------------------------
        # GET FIRST PAGE
        # ----------------------------------------------------

        first_page = get_review_page(
            item_id,
            1
        )

        if not first_page:

            print(
                "Could not retrieve first page."
            )

            continue


        model = first_page.get(
            "model",
            {}
        )

        paging = model.get(
            "paging",
            {}
        )

        total_pages = paging.get(
            "totalPages",
            1
        )


        print(
            f"Total pages: {total_pages}"
        )


        products_processed += 1


        # ----------------------------------------------------
        # PROCESS ALL PAGES
        # ----------------------------------------------------

        for page_no in range(
            1,
            total_pages + 1
        ):


            if page_no == 1:

                page_data = first_page

            else:

                time.sleep(
                    REQUEST_DELAY
                )

                page_data = get_review_page(
                    item_id,
                    page_no
                )


            if not page_data:

                continue


            page_model = page_data.get(
                "model",
                {}
            )

            reviews = (
                page_model.get(
                    "items"
                )
                or []
            )


            print(
                f"Page {page_no}/{total_pages}: "
                f"{len(reviews)} reviews"
            )


            # ------------------------------------------------
            # PROCESS REVIEWS
            # ------------------------------------------------

            for review in reviews:

                review_key = get_review_key(
                    item_id,
                    review
                )


                if not review_key:

                    continue


                # --------------------------------------------
                # FIND OUR RV ID
                # --------------------------------------------

                review_info = review_memory.get(
                    review_key
                )


                if not isinstance(
                    review_info,
                    dict
                ):

                    continue


                review_id = review_info.get(
                    "review_id"
                )


                if not review_id:

                    continue


                # --------------------------------------------
                # CHECK IF THIS REVIEW IS IN CSV
                # --------------------------------------------

                if review_id not in row_by_review_id:

                    continue


                # --------------------------------------------
                # EXTRACT DATES
                # --------------------------------------------

                bought_date = (
                    review.get(
                        "boughtDate"
                    )
                    or ""
                )

                review_date = (
                    review.get(
                        "reviewTime"
                    )
                    or ""
                )


                # --------------------------------------------
                # SAVE DATE INFORMATION
                # --------------------------------------------

                date_map[
                    review_id
                ] = {

                    "purchase_date":
                        bought_date,

                    "review_date":
                        review_date
                }


                matched_reviews += 1


                print(
                    f"  {review_id}: "
                    f"purchase={bought_date} | "
                    f"review={review_date}"
                )


    # ========================================================
    # UPDATE CSV ROWS
    # ========================================================

    for row in rows:

        review_id = row.get(
            "review_id"
        )

        if review_id in date_map:

            row["purchase_date"] = (
                date_map[
                    review_id
                ][
                    "purchase_date"
                ]
            )

            row["review_date"] = (
                date_map[
                    review_id
                ][
                    "review_date"
                ]
            )

        else:

            unmatched_reviews += 1


    # ========================================================
    # COLUMN ORDER
    # ========================================================

    new_columns = original_columns.copy()

    # Add the new columns at the end.
    new_columns.append(
        "purchase_date"
    )

    new_columns.append(
        "review_date"
    )


    # ========================================================
    # SAVE NEW CSV
    # ========================================================

    print("\n" + "=" * 65)

    print(
        "SAVING NEW DATASET"
    )

    print("=" * 65)


    success = save_csv(
        OUTPUT_CSV,
        rows,
        new_columns
    )


    if not success:

        return


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    rows_with_purchase_date = sum(
        bool(
            row.get(
                "purchase_date"
            )
        )
        for row in rows
    )

    rows_with_review_date = sum(
        bool(
            row.get(
                "review_date"
            )
        )
        for row in rows
    )


    print("\n" + "=" * 65)

    print(
        "TEMPORAL DATA UPDATE COMPLETE"
    )

    print("=" * 65)


    print(
        f"\nTotal CSV rows          : "
        f"{len(rows)}"
    )

    print(
        f"Reviews matched        : "
        f"{matched_reviews}"
    )

    print(
        f"Rows with purchase date: "
        f"{rows_with_purchase_date}"
    )

    print(
        f"Rows with review date  : "
        f"{rows_with_review_date}"
    )

    print(
        f"Rows without match     : "
        f"{unmatched_reviews}"
    )

    print(
        f"\nNew file:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        "\nOriginal reviews.csv was NOT modified."
    )

    print(
        "\nNew columns:"
    )

    print(
        "  purchase_date"
    )

    print(
        "  review_date"
    )

    print("=" * 65)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()