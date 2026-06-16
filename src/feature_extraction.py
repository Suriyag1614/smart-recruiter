import os
import json
import heapq
import csv
from datetime import datetime
from docx import Document
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION & PARAMETERS ---
INPUT_JSONL = "data/candidates.jsonl"
INPUT_DOCX = "data/job_description.docx"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_CSV = os.path.join(OUTPUT_DIR, "candidate_features.csv")
TOP_100_CSV = os.path.join(OUTPUT_DIR, "top_100.csv")
SUBMISSION_CSV = os.path.join(OUTPUT_DIR, "DarkHorses.csv")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # <-- Explicitly defined here
REFERENCE_YEAR = 2026

RETRIEVAL_TERMS = [
    "retrieval",
    "search",
    "semantic search",
    "information retrieval",
    "vector search",
    "ranking",
    "re-ranking",
    "recommendation",
    "recommender",
    "matching",
    "embeddings",
    "embedding",
    "similarity search"
]

VECTOR_TERMS = [
    "faiss",
    "pinecone",
    "milvus",
    "weaviate",
    "chroma",
    "vector database",
    "llm",
    "rag",
    "fine-tuning",
    "transformers",
    "lora"
]

def extract_jd_text(docx_path):
    """Extracts raw text from the provided Word document or returns a fallback."""
    if os.path.exists(docx_path):
        try:
            doc = Document(docx_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception:
            pass
    return "Senior AI Engineer. Applied ML/AI roles at product companies, building modern ML systems, embeddings, retrieval, ranking, LLMs, fine-tuning."


def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str).split('T')[0], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def calculate_metadata_scores(candidate):
    """
    Pass 1: Computes lightning-fast metadata checks.
    Allows us to screen 100k candidates instantly before doing heavy vector math.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    history = candidate.get("career_history", [])
    skills_list = candidate.get("skills", [])

    # --- Title Relevance ---
    target_titles = [
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "search engineer",
    "retrieval engineer",
    "relevance engineer",
    "recommendation engineer",
    "ranking engineer",
    "backend engineer",
    "data engineer",
    "software engineer"
    ]

    current_title = str(profile.get("current_title", "")).lower()
    title_relevance = 0.1
    if any(tgt in current_title for tgt in target_titles):
        title_relevance = 1.0 if any(ai in current_title for ai in ["ai", "ml", "machine"]) else 0.7

    # --- Experience Fit (5-9 Years target) ---
    try:
        yoe = float(profile.get("years_of_experience", 0.0))
    except (ValueError, TypeError):
        yoe = 0.0
    experience_fit = 1.0 if 5.0 <= yoe <= 9.0 else (0.7 if 4.0 <= yoe <= 11.0 else 0.3)

    # --- Product Company Score ---
    blacklisted_services = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "mindtree"]
    current_company = str(profile.get("current_company", "")).lower()
    history_companies = [str(job.get("company", "")).lower() for job in history]
    all_companies = history_companies + [current_company]
    service_overlap = sum(1 for c in all_companies if any(svc in c for svc in blacklisted_services))
    product_company_score = 1.0 if service_overlap == 0 else (0.5 if service_overlap < len(all_companies) else 0.0)

    # --- Production ML & Skill Verification ---
    history_entries = [f"Role: {j.get('title','')} at {j.get('company','')}. {j.get('description','')}" for j in history]
    career_evidence_text = " ".join(history_entries) if history_entries else profile.get("summary", "")
    
    career_lower = career_evidence_text.lower()

    retrieval_hits = sum(1 for kw in RETRIEVAL_TERMS if kw in career_lower)

    retrieval_domain_score = min(1.0, retrieval_hits / 8.0)

    vector_hits = sum(1 for kw in VECTOR_TERMS if kw in career_lower)

    vector_score = min(1.0,vector_hits / 6.0)

    prod_keywords = ["production", "pipeline", "deployed", "streaming", "scale", "kafka", "airflow", "kubernetes"]
    matches = sum(1 for kw in prod_keywords if kw in career_evidence_text.lower())
    production_ml_score = min(1.0, matches / 4.0) if matches > 0 else 0.1

    skills_text = ", ".join([s.get("name", "") for s in skills_list if isinstance(s, dict)])
    ai_skills = ["embeddings", "vector", "retrieval", "ranking", "llm", "fine-tuning"]
    skill_matches = sum(1 for s in ai_skills if s in skills_text.lower())
    skill_heuristic = min(1.0, skill_matches / 4.0) if skill_matches > 0 else 0.1

    # --- Recruitability Score ---
    open_to_work = signals.get("open_to_work_flag") in [True, "true", "True", 1]
    notice_days = signals.get("notice_period_days", 90)
    rec_base = 1.0 if open_to_work else 0.5
    notice_factor = 1.0 if notice_days <= 30 else (0.8 if notice_days <= 60 else 0.4)
    recruitability_score = (rec_base * 0.6) + (notice_factor * 0.4)

    # Base heuristic ranking score to find the top candidates for deep vector evaluation
    heuristic_score = (
    title_relevance * 0.25 +
    experience_fit * 0.20 +
    product_company_score * 0.15 +
    production_ml_score * 0.15 +
    retrieval_domain_score * 0.15 +
    vector_score * 0.10
)
    
    # Inactivity decay
    last_active = parse_date(signals.get("last_active_date"))
    if last_active and (datetime(REFERENCE_YEAR, 6, 14) - last_active).days > 180:
        heuristic_score *= 0.7

    return {
        "candidate_id": candidate.get("candidate_id", "UNKNOWN"),
        "anonymized_name": profile.get("anonymized_name", "N/A"),
        "career_text": career_evidence_text or "None",
        "profile_title": profile.get("current_title",""),
        "years_of_experience": profile.get("years_of_experience",0),
        "current_company": profile.get("current_company",""),
        "title_relevance": title_relevance,
        "experience_fit": experience_fit,
        "product_company_score": product_company_score,
        "production_ml_score": production_ml_score,
        "skill_heuristic": skill_heuristic,
        "recruitability_score": recruitability_score,
        "heuristic_score": heuristic_score,
        "retrieval_domain_score": retrieval_domain_score,
        "vector_score": vector_score
    }


def execute_pipeline():
    print("Starting Safe Two-Pass Filtering Pipeline...")
    
    # Ensure directory framework exists
    os.makedirs(os.path.dirname(FEATURE_CSV) if os.path.dirname(FEATURE_CSV) else '.', exist_ok=True)
    
    jd_text = extract_jd_text(INPUT_DOCX)
    
    # Pass 1: Stream file to instantly drop profiles with zero alignment
    print("Pass 1: Streaming structural metadata screening...")
    candidates_pool = []
    processed_count = 0

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
                meta = calculate_metadata_scores(candidate)
                processed_count += 1
                
                # Bounded Heap: Retain only the top 2,000 candidates based on structural indicators
                heap_item = (meta["heuristic_score"], processed_count, meta)
                if len(candidates_pool) < 2000:
                    heapq.heappush(candidates_pool, heap_item)
                else:
                    if meta["heuristic_score"] > candidates_pool[0][0]:
                        heapq.heapreplace(candidates_pool, heap_item)

                if processed_count % 20000 == 0:
                    print(f"  Screened {processed_count:,} profiles...")
            except Exception:
                continue

    # Extract clean target cohort from heap
    high_potential_candidates = [item[2] for item in candidates_pool]
    print(f"Pass 1 Complete. Selected {len(high_potential_candidates)} top-tier profiles for AI Vector mapping.")

    # Pass 2: Vector Search Inference Execution
    print(f"Loading embedding model: {os.path.basename(MODEL_NAME)}...")
    model = SentenceTransformer(MODEL_NAME)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)

    print("Running batched vector embedding transformations...")
    texts_to_embed = [c["career_text"] for c in high_potential_candidates]
    
    # High efficiency batch computation 
    embeddings = model.encode(texts_to_embed, batch_size=128, show_progress_bar=True, convert_to_tensor=True)
    similarities = util.cos_sim(embeddings, jd_embedding).cpu().numpy().flatten()

    # Calculate final matrix metrics
    final_output_rows = []
    for i, meta in enumerate(high_potential_candidates):
        sem_sim = max(0.0, min(1.0, float(similarities[i])))
        
        final_score = (
    sem_sim * 0.28 +
    meta["title_relevance"] * 0.15 +
    meta["experience_fit"] * 0.10 +
    meta["production_ml_score"] * 0.15 +
    meta["retrieval_domain_score"] * 0.15 +
    meta["vector_score"] * 0.10 +
    meta["skill_heuristic"] * 0.05 +
    meta["product_company_score"] * 0.01 +
    meta["recruitability_score"] * 0.01
    )

        # Generate reasoning based on evidence
        reasoning_parts = []
        if meta["title_relevance"] >= 0.7:
            reasoning_parts.append(f"Relevant title ({meta['profile_title']}) with {meta['years_of_experience']} YOE")
        else:
            reasoning_parts.append(f"Title is {meta['profile_title']} with {meta['years_of_experience']} YOE")
        
        if sem_sim > 0.6:
            reasoning_parts.append(f"strong semantic alignment ({sem_sim:.2f})")
        
        if meta["production_ml_score"] > 0.5:
            reasoning_parts.append("proven production ML experience")
            
        if meta["vector_score"] > 0.5:
            reasoning_parts.append("strong background in vector databases/search")

        if meta["recruitability_score"] < 0.8:
            reasoning_parts.append("high notice period/availability risk")

        reasoning = "; ".join(reasoning_parts) + "."

        final_output_rows.append({
            "candidate_id": meta["candidate_id"],
            "anonymized_name": meta["anonymized_name"] ,
            "score": round(final_score, 4),
            "final_score": round(final_score, 4),
            "reasoning": reasoning,
            "profile_title":meta["profile_title"],
            "years_of_experience":meta["years_of_experience"],
            "current_company":meta["current_company"],
            "semantic_similarity": round(sem_sim, 4),
            "title_relevance": round(meta["title_relevance"], 4),
            "experience_fit": round(meta["experience_fit"], 4),
            "retrieval_domain_score": round(meta["retrieval_domain_score"], 4),
            "vector_score": round(meta["vector_score"], 4),
            "production_ml_score": round(meta["production_ml_score"], 4),
            "skill_heuristic":round(meta["skill_heuristic"],4),
            "product_company_score": round(meta["product_company_score"], 4),
            "recruitability_score": round(meta["recruitability_score"], 4)
        })

    # Sort final tables
    final_output_rows.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    for rank,row in enumerate(final_output_rows, start=1):
        row["rank"] = rank

    # Save output artifacts using pandas
    print("Saving final datasets...")
    import pandas as pd
    df_all = pd.DataFrame(final_output_rows)
    cols = [
    "rank",
    "candidate_id",
    "anonymized_name",
    "final_score",
    "score",
    "reasoning",
    "profile_title",
    "years_of_experience",
    "current_company",
    "semantic_similarity",
    "title_relevance",
    "experience_fit",
    "retrieval_domain_score",
    "vector_score",
    "production_ml_score",
    "skill_heuristic",
    "product_company_score",
    "recruitability_score"
    ]

    df_all = df_all[cols]
    df_all.to_csv(FEATURE_CSV, index=False)
    df_all.head(100).to_csv(TOP_100_CSV, index=False)
    
    df_sub = df_all.head(100)[["candidate_id", "rank", "score", "reasoning"]]
    df_sub.to_csv(SUBMISSION_CSV, index=False)

    print(f"Pipeline successfully finished! Saved {FEATURE_CSV}, {TOP_100_CSV}, and {SUBMISSION_CSV}.")


if __name__ == "__main__":
    execute_pipeline()