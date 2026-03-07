import streamlit as st
import requests

API_URL = "https://web-production-298f.up.railway.app"

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

st.title("🧪 SHL Assessment Recommendation System")

query = st.text_area(
    "Enter job description or hiring query:",
    height=150,
    placeholder="Example: Hiring Java developers who collaborate well with stakeholders..."
)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Fetching recommendations..."):
            response = requests.post(
                f"{API_URL}/recommend",
                json={"query": query}
            )

            if response.status_code == 200:
                results = response.json()["recommended_assessments"]

                st.success(f"Found {len(results)} assessments")

                # Convert to table format
                table_data = []
                for i, r in enumerate(results, 1):
                    table_data.append({
                        "Rank": i,
                        "Assessment Name": r['name'],
                        "Description": r['description'],
                        "Test Type": ', '.join(r['test_type']),
                        "Duration (min)": r['duration'],
                        "Remote Support": r['remote_support'],
                        "Adaptive": r['adaptive_support'],
                        "URL": r['url']
                    })
                
                # Display as dataframe table
                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "URL": st.column_config.LinkColumn()
                    }
                )
            else:
                st.error("API error. Make sure backend is running.")
