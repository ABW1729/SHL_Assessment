import pandas as pd
import json

# Change this to your actual Excel file name
excel_file = "shl_dataset.xlsx"

# Load Excel file
xls = pd.ExcelFile(excel_file)
print("Available sheets:", xls.sheet_names)

# =========================
# TRAIN SET (Aggregate URLs)
# =========================
def extract_slug(url):
    return url.rstrip("/").split("/")[-1]
    
train_df = pd.read_excel(excel_file, sheet_name="Train-Set")

# Group by Query and aggregate URLs into list
train_grouped = (
    train_df.groupby("Query")["Assessment_url"]
    .apply(list)
    .reset_index()
)

# Convert list to JSON string for clean CSV storage
train_grouped["Assessment_urls"] = train_grouped["Assessment_url"].apply(json.dumps)

# Drop old column
train_grouped = train_grouped.drop(columns=["Assessment_url"])

# Save grouped train.csv
train_grouped.to_csv("train.csv", index=False)

# =========================
# TEST SET (No change)
# =========================

test_df = pd.read_excel(excel_file, sheet_name="Test-Set")
test_df.to_csv("test.csv", index=False)

print("Aggregated train.csv and test.csv generated successfully.")