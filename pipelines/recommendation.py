import numpy as np
import re
from pipelines.models import compute_embeddings


class RecommendationPipeline:

    def __init__(self, df, documents):

        self.df = df
        self.documents = documents

        # Pre-compute embeddings for all assessment names and descriptions
        doc_names = [str(row["name"]).lower() for _, row in df.iterrows()]
        doc_descs = [str(row["description"]).lower() for _, row in df.iterrows()]
        # Use global caching
        self.doc_name_embeddings = compute_embeddings(doc_names)
        self.doc_desc_embeddings = compute_embeddings(doc_descs)


# =========================
# RERANK
# =========================

    def rerank(self, query, candidates, structured=None):
        import os
        import requests
        HF_TOKEN = os.getenv("HF_TOKEN")
        API_URL = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-6-v2"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        if len(candidates) == 0:
            return []
        docs = [self.documents[i] for i in candidates]
        pairs = [[query, doc] for doc in docs]
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": pairs}
        )
        scores = response.json()
        # Defensive: if response is dict (error), fallback to zeros
        if isinstance(scores, dict):
            scores = [0.0] * len(docs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        # Return top 100 with their scores
        return ranked[:80]


    
    def diversify(self, scored_candidates, structured=None):
        """
        Balance top 100 results between:
        - Technical skill matches
        - Behavioral skill matches
        PRIORITY: Title matches > Description matches
        Target: 6 technical + 4 behavioral
        """

        if len(scored_candidates) == 0:
            return []

        tech_skills = []
        beh_skills = []

        if structured:
            tech_skills = [str(s).lower() for s in structured.get("technical_skills", [])]
            beh_skills = [str(s).lower() for s in structured.get("behavioral_skills", [])]

        # Precompute embeddings
        tech_embeddings = {}
        beh_embeddings = {}

        if tech_skills:
            cached_tech = compute_embeddings(tech_skills)
            for skill, emb in zip(tech_skills, cached_tech):
                tech_embeddings[skill] = emb

        if beh_skills:
            cached_beh = compute_embeddings(beh_skills)
            for skill, emb in zip(beh_skills, cached_beh):
                beh_embeddings[skill] = emb

        beh_matches = []
        tech_matches = []
        no_match = []

        for idx, score in scored_candidates:

            # TITLE matches
            has_beh_title = any(
                float(np.dot(self.doc_name_embeddings[idx], emb)) > 0.58
                for emb in beh_embeddings.values()
            )

            has_tech_title = any(
                float(np.dot(self.doc_name_embeddings[idx], emb)) > 0.58
                for emb in tech_embeddings.values()
            )

            # DESCRIPTION matches
            has_beh_desc = False
            has_tech_desc = False

            if not has_beh_title:
                has_beh_desc = any(
                    float(np.dot(self.doc_desc_embeddings[idx], emb)) > 0.58
                    for emb in beh_embeddings.values()
                )

            if not has_tech_title:
                has_tech_desc = any(
                    float(np.dot(self.doc_desc_embeddings[idx], emb)) > 0.58
                    for emb in tech_embeddings.values()
                )

            # FIX 1: allow both categories (remove elif)
            if has_tech_title or has_tech_desc:
                tech_matches.append((idx, score, has_tech_title))

            if has_beh_title or has_beh_desc:
                beh_matches.append((idx, score, has_beh_title))

            if not (has_tech_title or has_tech_desc or has_beh_title or has_beh_desc):
                no_match.append((idx, score))

        # Title matches first, then higher score
        tech_matches.sort(key=lambda x: (not x[2], -x[1]))
        beh_matches.sort(key=lambda x: (not x[2], -x[1]))
        no_match.sort(key=lambda x: -x[1])

        final = []
        added = set()

        # Add technical matches (max 6)
        for idx, score, _ in tech_matches:
            if len(final) >= 6:
                break
            if idx not in added:
                final.append(idx)
                added.add(idx)

        # Add behavioral matches (max 4)
        for idx, score, _ in beh_matches:
            if len(final) >= 10:
                break
            if idx not in added:
                final.append(idx)
                added.add(idx)

        # Fill remaining with highest scoring unmatched
        for idx, score in no_match:
            if len(final) >= 10:
                break
            if idx not in added:
                final.append(idx)
                added.add(idx)

        return final[:10]
# =========================
# FINAL FORMAT
# =========================

    def format(self, indices):

        results = []

        for idx in indices:

            row = self.df.iloc[idx]

            results.append({
                "url": row["url"],
                "name": row["name"],
                "description": row["description"],
                "duration": int(row["duration"]),
                "test_type": [
                    t.strip() for t in str(row["test_type"]).split(",")
                ]
            })

        return results
