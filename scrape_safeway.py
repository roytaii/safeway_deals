import requests
import csv
import pandas as pd
from datetime import datetime
import os
import time
import random
from google import genai
from dotenv import load_dotenv

load_dotenv()

# manually retrieved 
BASE = "https://api.flipp.com/flyerkit/v4.0/publications/safeway"

params_publication = {
    "access_token": os.getenv("FLIPP_TOKEN"),
    "locale": "en-US",
    "postal_code": os.getenv("FLIPP_POSTAL"),
    "store_code": os.getenv("FLIPP_STORE")
}

r = requests.get(BASE, params=params_publication)
publications = r.json()
current_pub = publications[0]["id"]
seen_pubs_file = "seen_publications.txt"
if os.path.exists(seen_pubs_file):
    with open(seen_pubs_file) as f:
        seen_pubs = set(f.read().splitlines())
else:
    seen_pubs = set()

if str(current_pub) in seen_pubs:
    print(f"Publication {current_pub} already processed, skipping.")
else:
    PRODUCTS_URL = f"https://dam.flippenterprise.net/flyerkit/publication/{current_pub}/products"
    params_product = {
        "display_type": "all",
        "locale": "en-US",
        "access_token": os.getenv("FLIPP_TOKEN")
    }

    r = requests.get(PRODUCTS_URL, params=params_product)
    items = r.json()
    if not items:
        print("Warning: No products found in the current publication")
    else: 
        filename = "safeway_prices.csv"
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
        else:
            existing_df = pd.DataFrame(columns=["name", "start_date", "end_date"])

        print(f"Number of Items: {len(items)}")

        rows = []
        for index, item in enumerate(items, start=1):
            pid = item.get("id")
            if not pid:
                print("Skipped 1 Item (Product ID Not Found)")
                continue
            url = f"https://dam.flippenterprise.net/flyerkit/product/{pid}"
            r = requests.get(url, params=params_product)
            if r.status_code != 200:
                print("Failed: ", pid)
                continue
            data = r.json()
            row = {
            "timestamp": datetime.now().isoformat(),
            "id": data.get("id"),
            "name": data.get("name"),
            "sale_desc": data.get("sale_story"),
            "SKU": data.get("sku"),
            "pre_price_text": data.get("pre_price_text"),
            "sale_price": data.get("price_text"),
            "post_price_text": data.get("post_price_text"),
            "regular_price": data.get("original_price"),
            "brand": data.get("brand"),
            "start_date": data.get("valid_from"),
            "end_date": data.get("valid_to"),
            "image_url": data.get("image_url")
            }
            is_duplicate = (
                (existing_df["name"] == row["name"]) &
                (existing_df["start_date"] == row["start_date"]) &
                (existing_df["end_date"] == row["end_date"])
            )
            if is_duplicate.any():
                print(f"Skipped duplicate: {row['name']}")
                continue
            rows.append(row)
            print(f"[{index}/{len(items)}] {data.get('name')}")
            time.sleep(random.uniform(0.1, 0.5))

        if rows: 
            df = pd.DataFrame(rows)
            if not os.path.exists(filename):
                df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL)
            else:
                df.to_csv(filename, mode="a", header=False, index=False, quoting=csv.QUOTE_ALL)
        with open(seen_pubs_file, "a") as f:
            f.write(str(current_pub) + "\n")


# CATEGORIZING ITEMS


df = pd.read_csv("safeway_prices.csv")

# Ensure category column exists
if "category" not in df.columns:
    df["category"] = None

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
CATEGORIES = [
    # Food Departments
    "Produce",
    "Meat & Seafood",
    "Dairy & Eggs",
    "Bakery",
    "Deli & Prepared Foods",
    "Frozen",
    "Pantry",
    "Snacks",
    "Beverages", 
    # Non-Food
    "Health & Wellness",
    "Personal Care",
    "Household",
    # Other
    "Floral",
    "Other"
]

def categorize_batch(df, uncategorized, batch_size=50):
    for i in range(0, len(uncategorized), batch_size):
        batch = uncategorized.iloc[i:i+batch_size]
        print(f'Categorizing {min(i+batch_size, len(uncategorized))} / {len(uncategorized)}')
        prompt = f"""Categorize each grocery item into one of: {CATEGORIES}
        
        Items:
        {chr(10).join(f"{j+1}. {name}" for j, name in enumerate(batch["name"]))}

        Respond with only a numbered list of category names matching the order above.
        Example format:
        1. Dairy
        2. Meat & Seafood
        """
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )

        lines = [
            l.split(". ", 1)[-1].strip()
            for l in response.text.strip().split("\n")
            if l.strip() and l.strip()[0].isdigit()
        ]

        # Validate — reject anything not in the known category list
        lines = [l if l in set(CATEGORIES) else "Other" for l in lines]

        if len(lines) != len(batch):
            print(f"  ⚠️ Warning: expected {len(batch)} results, got {len(lines)} — filling remainder with 'Other'")
            lines += ["Other"] * (len(batch) - len(lines))

        df.loc[batch.index, "category"] = lines
        df.to_csv("safeway_prices.csv", index=False)
        time.sleep(1)

# Only process uncategorized rows
uncategorized = df[df["category"].isna()]

if uncategorized.empty:
    print("All items already categorized, skipping API call.")
else:
    print(f"Categorizing {len(uncategorized)} items...")
    categorize_batch(df, uncategorized)
    print(f"Done.")
