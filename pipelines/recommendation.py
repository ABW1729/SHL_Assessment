import numpy as np
import re
from pipelines.models import compute_embeddings


class RecommendationPipeline:
    # In-memory cache for rerank results
    _rerank_cache = {}

    def log_filter_stage(self, stage, indices):
        import datetime
        with open('debug_filter_log.txt', 'a') as f:
            f.write(f"{stage},{indices},{datetime.datetime.now()}\n")

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

    def rerank(self, query, candidates, structured, top_k=80):

        self.log_filter_stage('rerank_input', candidates)
        tech_skills = structured.get("technical_skills", [])
        beh_skills = structured.get("behavioral_skills", [])

        # compute query embedding once
        q_emb = compute_embeddings([query])[0]

        scores = []

        for idx in candidates:

            doc = self.documents[idx].lower()

            # semantic similarity (name 30% + description 70%)
            combined_emb = self.doc_name_embeddings[idx] * 0.3 + self.doc_desc_embeddings[idx] * 0.7
            sim = float(np.dot(combined_emb, q_emb))

            # technical skill matches
            tech_matches = sum(
                1 for skill in tech_skills
                if skill.lower() in doc
            )

            # behavioral skill matches
            beh_matches = sum(
                1 for skill in beh_skills
                if skill.lower() in doc
            )

            # boosting
            tech_boost = 0.1 * tech_matches
            beh_boost = 0.2 * beh_matches

            score = sim + tech_boost + beh_boost

            scores.append((idx, score))

        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        reranked_indices = [idx for idx, _ in ranked[:top_k]]
        self.log_filter_stage('rerank_output', reranked_indices)
        return ranked[:top_k]


    
    def diversify(self, scored_candidates, structured=None):
        """
        Balance top 10 results between:
        - Technical skill matches
        - Behavioral skill matches

        Priority:
        Title match > Description match

        Target:
        6 technical + 4 behavioral
        """

        self.log_filter_stage('diversify_input', [idx for idx, _ in scored_candidates])
        if len(scored_candidates) == 0:
            return []

        tech_skills = []
        beh_skills = []

        if structured:
            tech_skills = [str(s).lower() for s in structured.get("technical_skills", [])]
            beh_skills = [str(s).lower() for s in structured.get("behavioral_skills", [])]

        tech_embeddings = {}
        beh_embeddings = {}

        if tech_skills:
            cached = compute_embeddings(tech_skills)
            for s, e in zip(tech_skills, cached):
                tech_embeddings[s] = e

        if beh_skills:
            cached = compute_embeddings(beh_skills)
            for s, e in zip(beh_skills, cached):
                beh_embeddings[s] = e

        beh_matches = []
        tech_matches = []
        no_match = []

        for idx, score in scored_candidates:

            # ---- TITLE SIMILARITY ----
            has_beh_title = any(
                float(np.dot(self.doc_name_embeddings[idx], beh_embeddings[s])) > 0.58
                for s in beh_skills if s in beh_embeddings
            )

            has_tech_title = any(
                float(np.dot(self.doc_name_embeddings[idx], tech_embeddings[s])) > 0.58
                for s in tech_skills if s in tech_embeddings
            )

            # ---- DESCRIPTION SIMILARITY ----
            has_beh_desc = False
            has_tech_desc = False

            if not has_beh_title:
                has_beh_desc = any(
                    float(np.dot(self.doc_desc_embeddings[idx], beh_embeddings[s])) > 0.58
                    for s in beh_skills if s in beh_embeddings
                )

            if not has_tech_title:
                has_tech_desc = any(
                    float(np.dot(self.doc_desc_embeddings[idx], tech_embeddings[s])) > 0.58
                    for s in tech_skills if s in tech_embeddings
                )

            # ---- CATEGORIZATION ----
            if has_beh_title or has_beh_desc:
                beh_matches.append((idx, score, has_beh_title))

            if has_tech_title or has_tech_desc:
                tech_matches.append((idx, score, has_tech_title))

            if not (has_beh_title or has_beh_desc or has_tech_title or has_tech_desc):
                no_match.append((idx, score))

        # ---- SORT: TITLE MATCH FIRST ----
        beh_matches = sorted(beh_matches, key=lambda x: (not x[2], -x[1]))
        tech_matches = sorted(tech_matches, key=lambda x: (not x[2], -x[1]))

        final = []
        added = set()

        # ---- ADD BEHAVIORAL (max 4) ----
        for idx, score, _ in beh_matches:
            if len(final) >= 4:
                break
            if idx not in added:
                final.append(idx)
                added.add(idx)

        # ---- ADD TECHNICAL (max 6 total) ----
        for idx, score, _ in tech_matches:
            if len(final) >= 10:
                break
            if idx not in added:
                final.append(idx)
                added.add(idx)

        # ---- FILL REMAINING ----
        if len(final) < 10:
            for idx, score in no_match:
                if len(final) >= 10:
                    break
                if idx not in added:
                    final.append(idx)
                    added.add(idx)
        self.log_filter_stage('diversify_output', final[:10])

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
