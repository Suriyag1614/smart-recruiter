# System Architecture

## High-Level System Design
The engine is built around a **Two-Pass Hybrid Retrieval Architecture**. Processing ~100k candidate profiles efficiently requires aggressive upfront pruning, followed by expensive deep semantic inference only on the most promising candidates.

## Data Flow Diagram

```mermaid
flowchart TD
    A[candidates.jsonl] --> B[Pass 1: JSON Streaming parser]
    C[job_description.docx] --> D[Word Text Extractor]
    B --> E{Pass 1 Metadata Filter\n(Heuristics & Keywords)}
    E -->|Drop ~98%| F[Discard Pile]
    E -->|Top 2000 Candidates| G[Bounded Min-Heap]
    G --> H[Pass 2: Vector Embedding]
    D --> I[Job Description Embedding]
    H --> J[Cosine Similarity Calculation]
    I --> J
    J --> K[Score Aggregation \n Semantic + Heuristics]
    K --> L[candidate_features.csv]
    K --> M[top_100.csv]
    K --> N[submission.csv]
```

## Pass 1: Metadata Screening Workflow
**Goal:** Process massive candidate lists instantly to surface high-potential profiles without loading the entire dataset into memory.

1. **Streaming Parsing:** Candidate JSON lines are read one by one to avoid OOM (Out-Of-Memory) issues.
2. **Metadata Checks:** Core dimensions are extracted:
   - **Title Relevance:** Boosts profiles containing "AI Engineer", "ML Engineer", etc.
   - **Experience Fit:** Targets a "sweet spot" of 5–9 years of experience.
   - **Service Company Downranking:** Identifies and down-weights profiles mostly anchored in non-product IT service companies.
   - **Recruitability:** Computes an availability score based on "open to work" flags and notice periods.
3. **Keyword Heuristics:** Scans raw career histories for keyword footprints across dimensions:
   - **Retrieval Domain:** e.g., "semantic search", "ranking"
   - **Vector Databases:** e.g., "faiss", "pinecone"
   - **Production ML:** e.g., "airflow", "kubernetes", "scale"
4. **Bounded Heap Retention:** The pipeline calculates a heuristic base score and pushes the profile to a fixed-size priority queue (Min-Heap) set to 2,000 slots. By the end of the pass, exactly the Top 2,000 strongest heuristic profiles are retained.

## Pass 2: Semantic Ranking Workflow
**Goal:** Measure deep contextual alignment between a candidate's holistic career narrative and the nuanced needs of the target Job Description.

1. **Vectorization:** 
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - The job description text is embedded into a dense vector space.
   - The combined career histories of the Top 2,000 candidates are batched and embedded via PyTorch arrays.
2. **Similarity Scoring:** Cosine Similarity is computed between the JD vector and every candidate's career vector to measure pure semantic intent.
3. **Hybrid Score Blending:** The `final_score` mathematically fuses Pass 1 logic with Pass 2 context. This prevents heavily buzzword-optimized (but contextually weak) profiles from winning, while also preventing theoretically dense but un-hirable profiles from passing.

## Explainability Layer
The solution is fundamentally explainable because every component of the `final_score` is serialized as an independent column in `candidate_features.csv`.
If a stakeholder asks "Why is Candidate A ranked #1?", the feature matrix directly exposes whether it was driven by their `semantic_similarity`, their `production_ml_score`, or their `experience_fit`. This clear mapping enables deep transparency, a requirement for HR fairness and system auditing.

## Scalability Discussion
* **Compute:** The O(N) streaming process in Pass 1 requires trivial compute overhead and scales easily to millions of JSON records. Pass 2 is bounded to O(K) where K is the heap size (2000), preventing vectorization bottlenecks.
* **Memory:** Since candidates are processed lazily and managed via a Bounded Heap, RAM consumption remains nearly flat regardless of whether the input size is 100k or 10 Million rows.
* **Batching:** Vector embedding transforms are batched on PyTorch structures allowing for rapid hardware acceleration if a GPU is available.
