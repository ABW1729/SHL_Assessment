
# SHL Assessment Recommendation System

This project implements an AI-powered assessment recommendation system that suggests relevant SHL assessments based on a job description or hiring query.

The system uses LLM-based entity extraction and hybrid retrieval techniques to identify the most relevant assessments from the SHL product catalog. The architecture combines semantic search, keyword search, and skill-based retrieval to improve recommendation accuracy.

---

# Demo

Frontend (Streamlit)  
Deployment URL: **[Add URL Here]**

Backend (FastAPI)  
API URL: **[Add API URL Here]**

---

# System Architecture

The system follows a three-stage pipeline:

User Query / Job Description  
↓  
LLM Entity Extraction  
↓  
Query Expansion  
↓  
Hybrid Retrieval  
• Dense Embedding Search  
• BM25 Keyword Search  
• Centroid Skill Search  
↓  
Reciprocal Rank Fusion (RRF)  
↓  
Filtering & Diversification  
↓  
Top 10 Recommended Assessments

---

# Key Features

## LLM-based Query Understanding
A language model extracts structured hiring requirements from job descriptions including:
- Primary role
- Technical skills
- Behavioral competencies
- Alternative job titles
- Job level
- Duration constraints

## Hybrid Retrieval Pipeline

### Dense Embedding Retrieval
Captures semantic similarity between queries and assessments.

### BM25 Keyword Retrieval
Ensures exact matching for technical skills and keywords.

### Centroid Skill Embeddings
Represents multi-skill roles by averaging embeddings of extracted skills.

## Ranking using Reciprocal Rank Fusion (RRF)
Results from different retrieval strategies are combined using Reciprocal Rank Fusion to produce a robust ranking.

## Result Filtering
Recommendations are filtered based on:
- Job level
- Assessment duration
- Language requirements

## Diversified Recommendations
The system ensures the final results contain a diverse mix of assessment types such as:
- Knowledge tests
- Personality assessments
- Ability tests
- Competency evaluations

---

# Project Structure

project/
│
├── ingestion.py          # Dataset ingestion and indexing  
├── retrieval.py          # Hybrid retrieval pipeline  
├── recommendation.py     # Recommendation and ranking logic  
├── models.py             # LLM and embedding integrations  
│  
├── backend/  
│   └── main.py           # FastAPI backend  
│  
├── frontend/  
│   └── app.py            # Streamlit frontend  
│  
├── data/  
│   └── assessments.csv  
│  
├── requirements.txt  
└── README.md  

---

# Installation

## Clone Repository

git clone <repo-url>  
cd project

## Create Virtual Environment

python -m venv env  
source env/bin/activate  

Windows:

env\Scripts\activate

## Install Dependencies

pip install -r requirements.txt

---

# Running the Backend

uvicorn main:app --host 0.0.0.0 --port 8000

API docs will be available at:

http://localhost:8000/docs

---

# Running the Frontend

streamlit run app.py

---

# API Endpoint

POST /recommend

Request body:

{
  "query": "Looking for a Java developer with collaboration skills"
}

Response:

{
  "recommended_assessments": [
    {
      "name": "Java Programming Test",
      "url": "...",
      "duration": 60
    }
  ]
}

---

# Technologies Used

- Python
- FastAPI
- Streamlit
- ChromaDB
- HuggingFace Inference API
- BM25 Retrieval
- Reciprocal Rank Fusion (RRF)

---

# Deployment

The system is optimized for lightweight cloud deployment.

Key optimizations include:
- Removing heavy local ML models
- Using HuggingFace API for embeddings
- Caching API results to reduce latency

These changes enable deployment on small cloud instances with limited memory.

---

# Recall Testing  
- To generate Mean Recall for training dataset run evaluate_recall.py file.
- To test model on testing data run evaluate_test.py file
