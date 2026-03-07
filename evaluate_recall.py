import pandas as pd
import requests
import json

API_URL = "https://web-production-298f.up.railway.app/recommend"


# ----------------------------------------
# Recall@K (Exact Matching Only)
# ----------------------------------------

def recall_at_k(true_urls, predicted_urls, k=10):

    true_slugs = {extract_slug(u) for u in true_urls}
    pred_slugs = {extract_slug(u) for u in predicted_urls[:k]}

    hits = len(true_slugs & pred_slugs)

    return hits / len(true_slugs) if len(true_slugs) > 0 else 0


# ----------------------------------------
# Main Evaluation
# ----------------------------------------
def extract_slug(url):
    return url.rstrip("/").split("/")[-1]

def mean_recall_at_k(train_csv, k=10):

    df = pd.read_csv(train_csv)

    recalls = []

    for _, row in df.iterrows():

        query = row["Query"]

        # Parse JSON list stored in CSV
        true_urls = json.loads(row["Assessment_urls"])

        # Call API endpoint
        response = requests.post(
            API_URL,
            json={"query": query}
        )

        if response.status_code != 200:
            print("API failed for query")
            continue

        data = response.json()

        predicted_urls = [
            item["url"]
            for item in data["recommended_assessments"]
        ]
        print(f"Query: {query[:100]}...")
        print(f"Expected: {len(true_urls)} assessments")
        print(f"Recommended: {len(predicted_urls)} assessments")

        r = recall_at_k(true_urls, predicted_urls, k)
        recalls.append(r)

        print(f"Recall@{k}: {r:.3f}")
        print("-" * 40)

    mean_recall = sum(recalls) / len(recalls)
    print(f"\nMean Recall@{k}: {mean_recall:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Evaluating Recall@10")
    print("=" * 60)
    mean_recall_at_k("train.csv", k=10)
