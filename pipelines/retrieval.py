import json
import numpy as np
from rank_bm25 import BM25Okapi
from utils.logger import log_debug
from utils.llm import run_llm
from pipelines.models import compute_embeddings


class RetrievalPipeline:
    def log_filter_stage(self, stage, indices):
        import datetime
        with open('debug_filter_log.txt', 'a') as f:
            f.write(f"{stage},{indices},{datetime.datetime.now()}\n")

    def __init__(self, df, documents, collection):

        self.df = df
        self.documents = documents
        self.collection = collection

        tokenized = [d.lower().split() for d in documents]
        self.bm25 = BM25Okapi(tokenized)
        # Use global caching for embeddings
        # Use global caching for embeddings (in-memory, already handled in compute_embeddings)
        self.embeddings = compute_embeddings(documents)

    # Optionally, add a static method to clear embedding cache if needed
    @staticmethod
    def clear_embedding_cache():
        from pipelines import models
        models.clear_cache()
        

# =========================
# ENTITY EXTRACTION
# =========================

    def extract_entities(self, query):
        ALLOWED_LEVELS = [
    "Graduate",
    "Professional Individual Contributor",
    "Mid-Professional",
    "Manager",
    "Supervisor",
    "Director",
    "Executive",
    "Entry-Level",
    "Front Line Manager",
    "General Population"
]

        prompt = f"""
You are an expert HR talent assessment analyst.

Your task is to extract COMPREHENSIVE hiring requirements from a job description.

IMPORTANT RULES:

1. Extract domain-relevant professional skills:
   - Technical skills (programming languages, tools, frameworks, etc.)
   - Behavioral competencies
   - Industry-specific knowledge

2. Ignore vague soft phrases (storytelling, passion, cultural fit, engagement, etc.)

3. Focus on measurable, assessable skills that correspond to job requirements

4. Extract up to 8 technical skills:
   - Specific tools, technologies, languages explicitly mentioned
   - Job titles and roles
   - Domain expertise areas
   - Prefer domain concepts over generic actions.
   - Merge related skills into one.
   - Add Full Forms for any abbreviations (e.g. "AWS" -> "Amazon Web Services")
   - DO NOT use parenthesis to represent synonyms - instead, list them directly in the skill list (e.g., "Python Programming", "Python Development" instead of "Python (Programming, Development)")
   - INCLUDE 1-2 synonyms/variations for each skill directly in the list (e.g., ["Python", "Python Programming", "Python Development"])

5. Behavioral skills:
   - Leadership, Decision Making, Collaboration, Problem Solving, etc.
   - INCLUDE 1-2 synonyms/variations for each skill directly in the list (e.g., ["Communication", "Verbal Communication", "Written Communication"])

6. Languages: Extract required languages from job description
   - Look for language requirements (e.g., "Bilingual Spanish", "Fluent in German", "English required")
   - Extract only the BASE language name (English, German, Spanish, French, etc.)
   - NOT specific variants like "English (USA)" or "English (Canada)" - use just "English"
   - If no languages specified, leave empty list
   - Examples: English, German, French, Spanish, Chinese, Portuguese, Italian, Dutch, etc.
   - If multiple languages required, list all of them

7. Suggest alternative job titles or phrasings

Allowed job levels (choose one or null):
{ALLOWED_LEVELS}

Test type definitions:
A = Ability (Cognitive reasoning)
B = Biodata (Experience/background)
C = Competency
D = Development
E = Assessment Exercise
K = Knowledge (Technical/domain tests)
P = Personality
S = Simulation (Situational judgement)

Duration and Job Level Extraction:
- Look for duration/time requirements (e.g., "15 min", "30-40 minutes", "max 1 hour")
- Extract job_levels from experience requirements (e.g., "0-2 years" → Entry-Level, "5+ years" → Mid-Professional)
- Use the ALLOWED_LEVELS list above to map experience to appropriate job level

Return RAW JSON only, no other text:

{{
"primary_role": string or null,
"technical_skills": list (max 8, includes synonyms/variations),
"behavioral_skills": list (max 3, includes synonyms/variations),
"languages": list,
"test_types": list,
"alternative_titles": list,
"min_duration": integer or null (in minutes),
"max_duration": integer or null (in minutes),
"job_levels": list or null
}}

Job Description:
{query}
"""

        response = run_llm(prompt)
        print(f"\nLLM RAW RESPONSE:\n{response}\n")
        
        structured = {}
        
        # Handle both dict and string responses
        if isinstance(response, dict):
            structured = response
        elif isinstance(response, str):
            # Try to parse JSON string
            try:
                structured = json.loads(response)
            except json.JSONDecodeError:
                # If direct parse fails, try to extract JSON from text
                import re
                match = re.search(r'\{[\s\S]*\}', response)
                if match:
                    try:
                        structured = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        structured = {}
                else:
                    structured = {}
        
        # Fix duration range: if min == max, set min to 0 to allow range
        min_duration = structured.get("min_duration")
        max_duration = structured.get("max_duration")
        
        if min_duration is not None and max_duration is not None:
            if min_duration == max_duration:
                structured["min_duration"] = 0

        log_debug("ENTITIES", structured)

        return structured




# =========================
# BUILD TERMS
# =========================

    def build_terms(self, structured):

        terms=set()

        role=structured.get("primary_role")

        tech=structured.get("technical_skills",[])
        beh=structured.get("behavioral_skills",[])
        titles=structured.get("alternative_titles",[])

        if role:
            terms.add(role)

        terms.update(tech)
        terms.update(beh)
        terms.update(titles)

        return list(terms), list(tech), list(beh)





    def embed(self, query):

        # Use global cache for query embeddings
        q_emb = compute_embeddings([query])[0]

        sims = np.dot(self.embeddings, q_emb)
        # Normalize embedding scores to [0,1]
        min_sim = np.min(sims)
        max_sim = np.max(sims)
        if max_sim - min_sim > 1e-8:
            sims_norm = (sims - min_sim) / (max_sim - min_sim)
        else:
            sims_norm = np.zeros_like(sims)

        ranked = np.argsort(sims_norm)[::-1][:50]
        return ranked.tolist()



    def bm25_search(self, query):

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        # Normalize BM25 scores by max score
        max_score = np.max(scores)
        if max_score > 1e-8:
            scores_norm = scores / max_score
        else:
            scores_norm = np.zeros_like(scores)
        ranked = np.argsort(scores_norm)[::-1][:50]
        return ranked.tolist()


# =========================
# CENTROID SEARCH
# =========================

    def centroid_search(self, terms):

        if len(terms) == 0:
            return []

       
        emb = compute_embeddings(terms)
        centroid = np.mean(emb, axis=0)
        sims = np.dot(self.embeddings, centroid)
        
        min_sim = np.min(sims)
        max_sim = np.max(sims)
        if max_sim - min_sim > 1e-8:
            sims_norm = (sims - min_sim) / (max_sim - min_sim)
        else:
            sims_norm = np.zeros_like(sims)
        ranked = np.argsort(sims_norm)[::-1][:20]
        return ranked.tolist()


# =========================
# RRF FUSION
# =========================

    def rrf(self, result_lists, weight_list=None):

        scores={}

        k=60

        if weight_list is None:
            weight_list=[1.0]*len(result_lists)

        for results,weight in zip(result_lists,weight_list):
            for rank,idx in enumerate(results):

                scores[idx]=scores.get(idx,0)+weight/(k+rank)

        ranked=sorted(scores.items(),key=lambda x:x[1],reverse=True)

        return [i for i,_ in ranked]


# =========================
# MAIN RETRIEVE
# =========================

    def retrieve(self, query, filtered_ids=None):

        self.log_filter_stage(
            'initial_pool',
            list(range(len(self.df))) if filtered_ids is None else filtered_ids
        )

        structured = self.extract_entities(query)
        terms, tech_skills , beh_skills = self.build_terms(structured)

        n_docs = len(self.df)

        # ---------- BM25 ----------
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_ranked = np.argsort(bm25_scores)[::-1][:120].tolist()

        # ---------- EMBEDDING ----------
        q_emb = compute_embeddings([query])[0]
        emb_scores = np.dot(self.embeddings, q_emb)
        emb_ranked = np.argsort(emb_scores)[::-1][:120].tolist()

        # ---------- CENTROID ----------
        centroid_terms = tech_skills + beh_skills if (tech_skills or beh_skills) else terms

        centroid_ranked = []

        if centroid_terms:
            emb = compute_embeddings(centroid_terms)
            centroid = np.mean(emb, axis=0)
            centroid_scores = np.dot(self.embeddings, centroid)
            centroid_ranked = np.argsort(centroid_scores)[::-1][:120].tolist()

        # ---------- RRF FUSION ----------
        fused = self.rrf(
            [bm25_ranked, emb_ranked, centroid_ranked],
            weight_list=[1.0, 1.2, 1.5]  
        )

        # ---------- FILTER ----------
        if filtered_ids is not None:
            filtered_set = set(filtered_ids)
            fused = [i for i in fused if i in filtered_set]

        candidates = fused[:120]

        log_debug("CANDIDATES", candidates)

        return candidates, structured

    def apply_filters_pre(self, structured):
        """
        Apply JOB LEVEL + LANGUAGE filters BEFORE retrieval (hard constraints).
        These are restrictive and should narrow the search pool.
        """
        # Log before job level filter
        self.log_filter_stage('pre_job_level', list(range(len(self.df))))
        filtered_indices = list(range(len(self.df)))

        # -----------------------
        # JOB LEVEL FILTER
        # -----------------------

        job_levels = structured.get("job_levels")

        if job_levels:
            filtered_indices = [
                i for i in filtered_indices
                if any(
                    level.lower() in str(self.df.iloc[i]["job_levels"]).lower()
                    for level in job_levels
                )
            ]
            self.log_filter_stage('post_job_level', filtered_indices)

        # -----------------------
        # LANGUAGE FILTER
        # -----------------------

        languages = structured.get("languages")

        if languages:
            # Build supported language set
            SUPPORTED_LANGUAGES = {
                'Arabic', 'Chinese Simplified', 'Chinese Traditional', 'Czech', 'Danish',
                'Dutch', 'English', 'Estonian', 'Finnish', 'Flemish', 'French',
                'German', 'Greek', 'Hungarian', 'Icelandic', 'Indonesian', 'Italian',
                'Japanese', 'Korean', 'Spanish', 'Latvian', 'Lithuanian', 'Malay',
                'Norwegian', 'Polish', 'Portuguese', 'Romanian', 'Russian', 'Serbian',
                'Slovak', 'Swedish', 'Thai', 'Turkish', 'Vietnamese'
            }
            
            def extract_languages_from_assessment(lang_str):
                """Extract all supported languages found in assessment language string"""
                found_langs = set()
                lang_str_lower = str(lang_str).lower()
                
                for supported_lang in SUPPORTED_LANGUAGES:
                    if supported_lang.lower() in lang_str_lower:
                        found_langs.add(supported_lang)
                
                return found_langs
            
            filtered_indices = [
                i for i in filtered_indices
                if any(
                    req_lang in extract_languages_from_assessment(str(self.df.iloc[i]["languages"]))
                    for req_lang in languages
                )
            ]
            self.log_filter_stage('post_language', filtered_indices)

        return filtered_indices

    def apply_filters_duration(self, candidates, structured):
        """
        Apply DURATION filter AFTER retrieval (flexible constraint).
        Allows best matches to pass through, just filtered by duration.
        """
        self.log_filter_stage('pre_duration', candidates)
        min_d = structured.get("min_duration")
        max_d = structured.get("max_duration")

        filtered_candidates = []
        for idx in candidates:
            duration = int(self.df.iloc[idx]["duration"])
            
            if min_d is not None and duration < min_d:
                continue
            if max_d is not None and duration > max_d:
                continue
            
            filtered_candidates.append(idx)

        self.log_filter_stage('post_duration', filtered_candidates)
        return filtered_candidates


     
       

    def apply_filters(self, structured):
        """
        Deprecated: Use apply_filters_pre() + apply_filters_duration() instead.
        Kept for backward compatibility.
        """
        filtered_indices = list(range(len(self.df)))

        # -----------------------
        # JOB LEVEL FILTER
        # -----------------------

        job_levels = structured.get("job_levels")

        if job_levels:
            filtered_indices = [
                i for i in filtered_indices
                if any(
                    level.lower() in str(self.df.iloc[i]["job_levels"]).lower()
                    for level in job_levels
                )
            ]

        # -----------------------
        # LANGUAGE FILTER
        # -----------------------

        languages = structured.get("languages")

        if languages:
            filtered_indices = [
                i for i in filtered_indices
                if any(
                    lang.lower() in str(self.df.iloc[i]["languages"]).lower()
                    for lang in languages
                )
            ]

        # -----------------------
        # DURATION FILTER
        # -----------------------

        min_d = structured.get("min_duration")
        max_d = structured.get("max_duration")

        if min_d is not None:
            filtered_indices = [
                i for i in filtered_indices
                if int(self.df.iloc[i]["duration"]) >= min_d
            ]

        if max_d is not None:
            filtered_indices = [
                i for i in filtered_indices
                if int(self.df.iloc[i]["duration"]) <= max_d
            ]

        return filtered_indices
        
