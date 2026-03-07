from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from pipelines.ingestion import IngestionPipeline
from pipelines.retrieval import RetrievalPipeline
from pipelines.recommendation import RecommendationPipeline


app = FastAPI()

ingestion = None
retrieval = None
recommendation = None


# =========================
# STARTUP
# =========================

@app.on_event("startup")

def startup():

    global ingestion, retrieval, recommendation

    ingestion = IngestionPipeline("data/assessments.csv")

    df = ingestion.get_dataframe()
    documents = ingestion.get_documents()
    collection = ingestion.get_collection()

    retrieval = RetrievalPipeline(df, documents, collection)

    recommendation = RecommendationPipeline(df, documents)

    print("Pipeline Ready")


# =========================
# REQUEST MODELS
# =========================

class RecommendationRequest(BaseModel):
    query: str


class Assessment(BaseModel):
    url: str
    name: str
    description: str
    duration: int
    test_type: List[str]
    remote_support: str
    adaptive_support: str


class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]


# =========================
# PIPELINE ENDPOINT
# =========================


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest):

    query = req.query

    # Extract entities first
    structured = retrieval.extract_entities(query)

    # Apply job level and language filters BEFORE retrieval (hard constraints)
    filtered_ids = retrieval.apply_filters_pre(structured)

    # Retrieve with pre-filtered pool (job level + language constrained)
    candidates, structured = retrieval.retrieve(query, filtered_ids)

    # Apply duration filter AFTER retrieval (flexible constraint)
    candidates = retrieval.apply_filters_duration(candidates, structured)

    reranked = recommendation.rerank(query, candidates, structured)

    final_indices = recommendation.diversify(reranked, structured)

    results = recommendation.format(final_indices)

    for r in results:
        r["remote_support"] = retrieval.df.loc[
            retrieval.df["url"] == r["url"], "remote_support"
        ].values[0]

        r["adaptive_support"] = retrieval.df.loc[
            retrieval.df["url"] == r["url"], "adaptive_support"
        ].values[0]

    return {"recommended_assessments": results}


# =========================
# HEALTH ENDPOINT
# =========================
@app.get("/health")
def health():
    return {"status": "healthy"}
