import pandas as pd
import requests
import json

API_URL = "http://localhost:8000/recommend"


def evaluate_test(test_csv, output_csv=None):
    """
    Reads queries from test.csv, makes API calls, and writes results back to CSV.
    
    Args:
        test_csv: Path to input CSV with Query column
        output_csv: Path to output CSV (defaults to test_results.csv)
    """
    if output_csv is None:
        output_csv = "test_results.csv"
    
    df = pd.read_csv(test_csv)
    
    # Each row: Query, Assessment_URL
    results = []

    for idx, row in df.iterrows():
        query = row["Query"]

        print(f"Processing query {idx + 1}/{len(df)}")
        print(f"Query: {query[:100]}...")

        try:
            response = requests.post(
                API_URL,
                json={"query": query}
            )

            if response.status_code != 200:
                print(f"API error: Status {response.status_code}")
                results.append({
                    "Query": query,
                    "Assessment_URL": None
                })
                continue

            data = response.json()
            recommended = data.get("recommended_assessments", [])

            if not recommended:
                results.append({
                    "Query": query,
                    "Assessment_URL": None
                })
            else:
                for item in recommended:
                    results.append({
                        "Query": query,
                        "Assessment_URL": item.get("url")
                    })

            print(f"Found {len(recommended)} assessments")
            print("-" * 60)

        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            results.append({
                "Query": query,
                "Assessment_URL": None
            })
            print("-" * 60)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    print(f"\nResults saved to {output_csv}")
    print(f"Total query-prediction pairs: {len(results_df)}")


if __name__ == "__main__":
    evaluate_test("test.csv", "test_results.csv")
