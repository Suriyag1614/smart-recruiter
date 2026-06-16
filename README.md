# Intelligent Candidate Discovery & Ranking Engine

## Project Overview
This project is an advanced, two-pass hybrid candidate retrieval and ranking engine submitted for the **India Runs Data & AI Challenge**. It is designed to evaluate massive pools of candidate profiles (~100k) against a target job description and systematically rank the best matches.

## Challenge Objective
The goal is to build a high-performance ranking engine that can:
1. Parse complex candidate profiles from a JSONL dataset.
2. Extract rich ranking signals from unstructured career history and skills.
3. Perform deep semantic matching against the target job description (JD).
4. Generate explainable rankings for top candidates.
5. Produce final artifacts (`candidate_features.csv`, `top_100.csv`) representing the best candidate recommendations.

## Architecture
The system employs a Two-Pass Hybrid Retrieval Architecture to ensure both high speed and deep contextual understanding:

* **Pass 1: Metadata Screening (Heuristic Filtering):** 
  Evaluates candidates using deterministic metadata and keyword heuristics (e.g., Title Relevance, Experience Fit, Production ML Score, Vector Database Score). It retains the top 2,000 candidates using an efficient bounded heap.
* **Pass 2: Semantic Ranking (Vector Search):** 
  Utilizes the `sentence-transformers/all-MiniLM-L6-v2` model to compute dense vector embeddings of the job description and the top 2,000 candidates' career histories. Cosine similarity is combined with the metadata scores to produce the final ranking.

For detailed system design, data flow, and scalability, refer to `docs/ARCHITECTURE.md`.

## Folder Structure
```text
smart-recruiter/
├── src/
│   ├── feature_extraction.py   # Core production ranking pipeline
│   ├── eda.py                  # (Experimental) Exploratory Data Analysis
│   └── validate_ranking.py     # (Experimental) Validation & Audit Script
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview & instructions
├── ARCHITECTURE.md             # System design & workflows
└── SUBMISSION_NOTES.md         # Tradeoffs, assumptions & explainability
```

## Installation
Ensure you have Python 3.9+ installed.

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Execution Steps (Reproduction)
To reproduce the submission CSV end-to-end within the challenge compute budget:

1. Place the input files in their expected directory:
   * Candidates: `[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl`
   * Job Description: `[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx`
2. Run the core reproduction command:
   ```bash
   python src/feature_extraction.py
   ```
3. The results will be saved in the `outputs/` directory.

## Sandbox Testing
A sandbox environment is available at `[PLACEHOLDER_SANDBOX_LINK]` (e.g., HuggingFace Spaces). This sandbox accepts a small sample of candidates and demonstrates the end-to-end ranking functionality.

## Methodology & Ranking Logic
The engine calculates a `final_score` by blending semantic similarities with hard metadata rules. 
The final mathematical formula used in Pass 2 is:

```python
final_score = (
    semantic_similarity * 0.28 +
    title_relevance * 0.15 +
    experience_fit * 0.10 +
    production_ml_score * 0.15 +
    retrieval_domain_score * 0.15 +
    vector_score * 0.10 +
    skill_heuristic * 0.05 +
    product_company_score * 0.01 +
    recruitability_score * 0.01
)
```
This formula heavily weights Semantic Similarity (28%), while strongly rewarding structural relevance (Title, Production ML experience, and Retrieval Domain knowledge).

## Outputs
* **`outputs/candidate_features.csv`**: Contains the complete extracted feature set and scores for the top evaluated cohort (for debugging and review).
* **`outputs/top_100.csv`**: Contains the top 100 candidates with all extracted features.
* **`outputs/submission.csv`**: The official submission deliverable matching the 4-column spec (`candidate_id`, `rank`, `score`, `reasoning`). Rename this to your participant ID (e.g., `team_xxx.csv`) before uploading.

## Future Improvements
* **Advanced RAG integration**: Dynamically querying LLMs to justify individual candidate ranks.
* **Finer Embedding Models**: Moving from MiniLM to larger contextual models (e.g., `BGE-m3` or `OpenAI text-embedding-3`).
* **Automated Hyperparameter Tuning**: Using Bayesian optimization on the scoring formula weights against human-labeled datasets.
