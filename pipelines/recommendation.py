import numpy as np
import re
from pipelines.models import get_cross_encoder, get_sentence_transformer, compute_embeddings


class RecommendationPipeline:

    def __init__(self, df, documents):

        self.df = df
        self.documents = documents

        # Use global models - loaded once, never reload
        self.cross_encoder = get_cross_encoder()
        self.embedder = get_sentence_transformer()
        
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

        if len(candidates) == 0:
            return []

        pairs = [(query, self.documents[i]) for i in candidates]

        scores = self.cross_encoder.predict(pairs)

        # Collect ALL extracted terms from structured JSON (skills and roles only)
        all_terms = []
        
        if structured:
            # Extract matching terms (exclude filters: test_types, job_levels, languages)
            if structured.get("primary_role"):
                all_terms.append(str(structured.get("primary_role")).lower())
            
            if structured.get("technical_skills"):
                all_terms.extend([str(s).lower() for s in structured.get("technical_skills", [])])
            
            if structured.get("behavioral_skills"):
                all_terms.extend([str(s).lower() for s in structured.get("behavioral_skills", [])])
            
            if structured.get("alternative_titles"):
                all_terms.extend([str(t).lower() for t in structured.get("alternative_titles", [])])

        # OPTIMIZATION: Pre-compute embeddings for all terms (do once, not per candidate)
        # Use global cache to avoid re-computing if same terms appear in multiple queries
        term_embeddings = {}
        if all_terms:
            cached_terms = compute_embeddings(all_terms)
            for term, emb in zip(all_terms, cached_terms):
                term_embeddings[term] = emb

        for i, idx in enumerate(candidates):

            title_matches = 0
            description_matches = 0

            # -----------------------
            # CHECK TITLE AND DESCRIPTION FOR EXTRACTED TERMS (embedding-based semantic matching)
            # -----------------------
            for term in all_terms:
                if term and len(term) > 0:
                    term_embedding = term_embeddings[term]
                    
                    # Compute cosine similarity with assessment title
                    title_sim = float(np.dot(self.doc_name_embeddings[idx], term_embedding))
                    
                    # Compute cosine similarity with description (use pre-computed embeddings)
                    desc_sim = float(np.dot(self.doc_desc_embeddings[idx], term_embedding))
                    
                    # Use 0.58 threshold: balances catching real matches (SEO, Content) while avoiding false positives
                    if title_sim > 0.58:
                        title_matches += 1
                    
                    if desc_sim > 0.58:
                        description_matches += 1

            # -----------------------
            # SCORE BOOSTING
            # -----------------------
            # Title matches (strong signal): +5.0 per match
            scores[i] += title_matches * 5.0
            
            # Description matches (weaker signal): +0.8 per match
            scores[i] += description_matches * 0.8

        ranked = np.argsort(scores)[::-1]

        # Return top 80 with their scores
        return [(candidates[i], scores[i]) for i in ranked[:80]]


    
    def diversify(self, scored_candidates, structured=None):
        """
        Balance top 10 results between:
        - Technical skill matches (assessments with tech skills in title or description)
        - Behavioral skill matches (assessments with behavioral skills in title or description)
        PRIORITY: Title matches > Description matches
        Target: 6 technical + 4 behavioral
        """
        
        if len(scored_candidates) == 0:
            return []

        # Collect behavioral and technical skills for matching
        tech_skills = []
        beh_skills = []
        
        if structured:
            tech_skills = [str(s).lower() for s in structured.get("technical_skills", [])]
            beh_skills = [str(s).lower() for s in structured.get("behavioral_skills", [])]

        # OPTIMIZATION: Pre-compute embeddings for all skills (do once, not per candidate)
        # Use global cache to avoid re-computing same skills
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

        # Categorize candidates by title + description matches
        beh_matches = []   # Has behavioral skill match (priority: title > description)
        tech_matches = []  # Has technical skill match (priority: title > description)
        no_match = []      # No skill match
        
        for idx, score in scored_candidates:
            # Check TITLE matches first (higher priority)
            has_beh_title = any(float(np.dot(self.doc_name_embeddings[idx], beh_embeddings[skill])) > 0.58 for skill in beh_skills if skill in beh_embeddings)
            has_tech_title = any(float(np.dot(self.doc_name_embeddings[idx], tech_embeddings[skill])) > 0.58 for skill in tech_skills if skill in tech_embeddings)
            
            # If no title match, check DESCRIPTION (lower priority, but still counted)
            has_beh_desc = False
            has_tech_desc = False
            
            if not has_beh_title:
                has_beh_desc = any(float(np.dot(self.doc_desc_embeddings[idx], beh_embeddings[skill])) > 0.58 for skill in beh_skills if skill in beh_embeddings)
            
            if not has_tech_title:
                has_tech_desc = any(float(np.dot(self.doc_desc_embeddings[idx], tech_embeddings[skill])) > 0.58 for skill in tech_skills if skill in tech_embeddings)
            
            # Categorize with priority: title > description
            if has_beh_title or has_beh_desc:
                beh_matches.append((idx, score, has_beh_title))  # Include flag for title match
            elif has_tech_title or has_tech_desc:
                tech_matches.append((idx, score, has_tech_title))  # Include flag for title match
            else:
                no_match.append((idx, score))
        
        # Sort by title match priority (title matches first, then by score)
        beh_matches = sorted(beh_matches, key=lambda x: (not x[2], -x[1]))  # True (title match) comes first
        tech_matches = sorted(tech_matches, key=lambda x: (not x[2], -x[1]))
        
        final = []
        
        # First: Add behavioral skill matches (up to 4, prioritizing title matches)
        for idx, score, _ in beh_matches[:4]:
            final.append(idx)
        
        # Second: Add technical skill matches (up to 6, prioritizing title matches)
        for idx, score, _ in tech_matches[:6]:
            final.append(idx)
        
        # Third: If still need more, add non-matching items
        if len(final) < 10:
            for idx, score in no_match:
                if len(final) >= 10:
                    break
                final.append(idx)
        
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