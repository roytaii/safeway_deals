import pandas as pd
import json
import csv

with open("safeway_prices.csv", "r") as f:
    reader = csv.reader(f)
    rows = list(reader)
df = pd.DataFrame(rows[1:], columns=rows[0])
today = pd.Timestamp('today').normalize()
df['end_date'] = pd.to_datetime(df['end_date'])
df['start_date'] = pd.to_datetime(df['start_date'])

def is_friday_only(row):
    return pd.notna(row['start_date']) and pd.notna(row['end_date']) and row['start_date'] == row['end_date']

df['friday_only'] = df.apply(is_friday_only, axis=1)
active_df   = df[df['end_date'] >= today].sort_values('name')
expired_df  = df[df['end_date'] <  today].sort_values('name')

def to_records(df):
    df = df.copy()
    df['end_date'] = df['end_date'].astype(str)
    df['start_date'] = df['start_date'].astype(str)
    return df.fillna("").to_dict(orient="records")

combined = {
    "active": to_records(active_df),
    "expired": to_records(expired_df)
}

with open("deals.json", "w") as f:
    json.dump(combined, f)

print(f"Done. {len(active_df)} active, {len(expired_df)} expired.")