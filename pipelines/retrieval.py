import json
import numpy as np
from rank_bm25 import BM25Okapi
from utils.logger import log_debug
from utils.llm import run_llm
from pipelines.models import compute_embeddings


class RetrievalPipeline:

    def __init__(self, df, documents, collection):

        self.df = df
        self.documents = documents
        self.collection = collection

        tokenized = [d.lower().split() for d in documents]
        self.bm25 = BM25Okapi(tokenized)
        # Use global caching for embeddings
        self.embeddings = compute_embeddings(documents)
        

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
# QUERY EXPANSION
# =========================

    def expand_queries(self, structured):

        expanded = set()

        role = structured.get("primary_role")
        tech = structured.get("technical_skills") or []
        beh = structured.get("behavioral_skills") or []

        if role:
            expanded.add(role)

        expanded.update(tech)
        expanded.update(beh)

        for s in tech:
            if role:
                expanded.add(f"{role} {s}")

        return list(expanded)


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

        return list(terms), list(tech)


# =========================
# FORWARD SEARCH
# =========================

    def forward_search(self, query):

        res=self.collection.query(
            query_texts=[query],
            n_results=50
        )

        return [int(i) for i in res["ids"][0]]


# =========================
# REVERSE SEARCH
# =========================

    def reverse_search(self, query):

        # Use global cache for query embeddings
        q_emb = compute_embeddings([query])[0]

        sims = np.dot(self.embeddings, q_emb)

        ranked = np.argsort(sims)[::-1][:50]

        return ranked.tolist()


# =========================
# BM25 SEARCH
# =========================

    def bm25_search(self, query):

        tokens=query.lower().split()

        scores=self.bm25.get_scores(tokens)

        ranked=np.argsort(scores)[::-1][:50]

        return ranked.tolist()


# =========================
# CENTROID SEARCH
# =========================

    def centroid_search(self, terms):

        if len(terms) == 0:
            return []

        # Use global cache for term embeddings
        emb = compute_embeddings(terms)

        centroid = np.mean(emb, axis=0)

        sims = np.dot(self.embeddings, centroid)

        ranked = np.argsort(sims)[::-1][:20]

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

        structured = self.extract_entities(query)

        terms, tech_skills = self.build_terms(structured)

        expanded_terms = self.expand_queries(structured)

        log_debug("SEARCH_TERMS", terms)

        result_lists = []
        weight_list = []

        # helper - filter by pre-filtered IDs
        def filter_by_ids(results):
            if filtered_ids is None:
                return results
            allowed = set(filtered_ids)
            return [i for i in results if i in allowed]

        # ------------------------
        # FULL QUERY
        # ------------------------

        result_lists.append(filter_by_ids(self.forward_search(query)))
        weight_list.append(2.0)

        result_lists.append(filter_by_ids(self.reverse_search(query)))
        weight_list.append(2.0)

        result_lists.append(filter_by_ids(self.bm25_search(query)))
        weight_list.append(2.0)

        # ------------------------
        # ROLE
        # ------------------------

        role = structured.get("primary_role")

        if role:

            result_lists.append(filter_by_ids(self.forward_search(role)))
            weight_list.append(1.5)

            result_lists.append(filter_by_ids(self.reverse_search(role)))
            weight_list.append(1.5)

            result_lists.append(filter_by_ids(self.bm25_search(role)))
            weight_list.append(1.5)

        # ------------------------
        # TECH SKILLS
        # ------------------------

        for skill in tech_skills:

            result_lists.append(filter_by_ids(self.forward_search(skill)))
            weight_list.append(1.5)

            result_lists.append(filter_by_ids(self.reverse_search(skill)))
            weight_list.append(1.5)

            result_lists.append(filter_by_ids(self.bm25_search(skill)))
            weight_list.append(1.5)

        # ------------------------
        # BEHAVIORAL SKILLS
        # ------------------------

        beh = structured.get("behavioral_skills", [])

        for skill in beh:

            result_lists.append(filter_by_ids(self.forward_search(skill)))
            weight_list.append(1.3)

            result_lists.append(filter_by_ids(self.reverse_search(skill)))
            weight_list.append(1.3)

            result_lists.append(filter_by_ids(self.bm25_search(skill)))
            weight_list.append(1.3)

        # ------------------------
        # EXPANDED QUERIES
        # ------------------------

        for q in expanded_terms[:5]:

            result_lists.append(filter_by_ids(self.forward_search(q)))
            weight_list.append(1.2)

            result_lists.append(filter_by_ids(self.bm25_search(q)))
            weight_list.append(1.2)

        # ------------------------
        # OTHER TERMS
        # ------------------------

        other_terms = [t for t in terms if t not in tech_skills]

        for term in other_terms[:3]:

            result_lists.append(filter_by_ids(self.forward_search(term)))
            weight_list.append(1.0)

            result_lists.append(filter_by_ids(self.bm25_search(term)))
            weight_list.append(1.0)

        # ------------------------
        # CENTROID SEARCH
        # ------------------------

        centroid_terms = tech_skills if tech_skills else terms

        result_lists.append(filter_by_ids(self.centroid_search(centroid_terms)))
        weight_list.append(1.0)

        # ------------------------
        # FUSION
        # ------------------------

        fused = self.rrf(result_lists, weight_list)

        candidates = fused[:120]

        log_debug("CANDIDATES", candidates)

        return candidates, structured

    def apply_filters_pre(self, structured):
        """
        Apply JOB LEVEL + LANGUAGE filters BEFORE retrieval (hard constraints).
        These are restrictive and should narrow the search pool.
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

        return filtered_indices

    def apply_filters_duration(self, candidates, structured):
        """
        Apply DURATION filter AFTER retrieval (flexible constraint).
        Allows best matches to pass through, just filtered by duration.
        """
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

        return filtered_candidates

    def apply_filters_test_type(self, candidates, structured):
        """
        Apply TEST TYPE filter to ensure assessments match required test types.
        This is a post-retrieval constraint that eliminates irrelevant assessment types.
        """
        required_types = structured.get("test_types")
        
        if not required_types:
            return candidates
        
        # Convert required types to lowercase for case-insensitive comparison
        required_types_lower = {t.lower() for t in required_types}
        
        filtered_candidates = []
        for idx in candidates:
            assessment_types_str = str(self.df.iloc[idx]["test_type"])
            # Split by comma and strip whitespace
            assessment_types = {t.strip().lower() for t in assessment_types_str.split(",")}
            
            # Keep assessment if it has at least one matching test type
            if assessment_types & required_types_lower:  # intersection check
                filtered_candidates.append(idx)
        
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
        
