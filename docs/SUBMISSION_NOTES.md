# Submission Notes

## Assumptions
1. **Target Candidate Profile:** Based on the JD and the evaluation metrics, we assume the target is a "Senior AI Engineer" (or similar title) with 5–9 years of experience, a background in product-centric companies (vs. services), and direct experience deploying Retrieval/Ranking and Vector Search systems to production.
2. **Missing Data:** If dates or data points are missing in a candidate profile, safe fallbacks are applied instead of throwing exceptions, ensuring maximum pipeline robustness.
3. **Blacklisted Services Companies:** A heuristic penalty is applied if a candidate's background is exclusively situated in mass-IT services (e.g., TCS, Infosys, Cognizant), under the assumption that founding AI teams seek product-oriented scaling experience.

## Design Decisions
1. **No External LLM API Calls in the Loop:** Calling OpenAI/Anthropic APIs 100k times is economically unviable and too slow. Thus, semantic extraction relies on local embeddings (`sentence-transformers`), operating significantly faster and at zero marginal cost.
2. **Streaming IO:** We never load the full 100k records into memory. The JSONL is streamed and handled record-by-record. 
3. **Bounded Heap vs Array Sort:** Rather than keeping all candidate scores in memory and sorting at the end, a bounded min-heap holds only the `Top 2000` candidates during Pass 1, reducing sorting complexity and peak memory to near-zero.

## Why Hybrid Ranking Was Chosen
Pure Vector Search (Semantic Similarity alone) often fails in HR tech. An embedding model might highly rank an academic who writes extensively about "transformers" and "ranking algorithms," even if they have 0 years of industry experience and are not "open to work."

Conversely, pure Keyword Searching fails because it cannot grasp context.
By combining both (Hybrid Ranking), we enforce strict HR boundaries (Title relevance, Years of Experience, Recruitability) while retaining the "fuzzy" contextual matching power of neural embeddings.

## Tradeoffs
1. **Speed vs. Model Size:** We use `all-MiniLM-L6-v2` which produces a 384-dimensional vector. It is exceptionally fast but slightly less nuanced than massive modern embedding models (e.g., 1536-dim OpenAI models). This tradeoff prioritizes pipeline execution speed on standard hardware over marginally higher semantic fidelity.
2. **Hard Heuristics:** Hardcoded keywords (e.g., `["faiss", "pinecone", "milvus"]`) are somewhat rigid. While effective, they risk missing new, cutting-edge tools if the dictionary is not maintained.

## Performance Considerations
* Batch processing vector embeddings drastically reduces PyTorch overhead.
* File I/O operations are optimized by utilizing Python's fast generator primitives.
* Heavy text processing, lowercase conversions, and string scans are heavily optimized, restricting CPU bounds in Python.

## Explainability Approach
Explainable AI is prioritized by tracking sub-scores. In standard deep learning matching engines, a final score of `0.94` is a black box. In this hybrid system, the `0.94` is structurally broken down:
* We can say "Candidate X scored high because they had a Semantic Similarity of `0.85`, but they won out over Candidate Y because of a `1.0` Production ML Score due to their Airflow/Kafka experience."
* All sub-scores are explicitly exported to the final CSV, giving Human Resources direct observability into *why* the AI made the recommendation.
