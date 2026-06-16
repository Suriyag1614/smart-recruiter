import os
import json
import csv
from collections import Counter

# --- CONFIGURATION ---
TOP_100_CSV = "outputs/top_100.csv"
INPUT_JSONL = "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
DETAILED_CSV = "outputs/top_100_detailed.csv"
AUDIT_REPORT = "outputs/ranking_audit.md"


def run_validation_pipeline():
    print("Initializing Ranking Validation Pipeline...")

    if not os.path.exists(TOP_100_CSV):
        print(f"Error: {TOP_100_CSV} not found. Run your feature extraction script first.")
        return

    # 1. Load the Top 100 leaderboard scores into memory as a lookup map
    scores_lookup = {}
    ordered_ids = []  # Keep track of original final_score rank order
    
    with open(TOP_100_CSV, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["candidate_id"]
            scores_lookup[cid] = row
            ordered_ids.append(cid)

    target_ids_set = set(ordered_ids)
    print(f"Loaded {len(target_ids_set)} target candidate keys from leaderboard.")

    # 2. Stream through JSONL line-by-line to match raw profiles
    print(f"Streaming {INPUT_JSONL} to harvest comprehensive profiles...")
    matched_profiles = {}
    processed_count = 0

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                cid = record.get("candidate_id")
                
                if cid in target_ids_set:
                    matched_profiles[cid] = record
                    if len(matched_profiles) == len(target_ids_set):
                        break  # Early exit if we find all 100 profiles early
                        
                processed_count += 1
                if processed_count % 20000 == 0:
                    print(f"  Scanned {processed_count:,} source JSONL records...")
            except Exception:
                continue

    print(f"Harvested profile data for {len(matched_profiles)} matched candidates.")

    # 3. Consolidate metrics and export top_100_detailed.csv
    print(f"Generating enriched profile compilation: {DETAILED_CSV}...")
    
    detailed_headers = [
        "candidate_id", "anonymized_name", "current_title", "current_company", 
        "current_industry", "years_of_experience", "headline", "summary", 
        "top_10_skills", "final_score", "semantic_similarity", "title_relevance", 
        "experience_fit", "retrieval_domain_score", "vector_score", "production_ml_score", 
        "product_company_score", "recruitability_score"
    ]

    final_ordered_rows = []

    with open(DETAILED_CSV, mode='w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=detailed_headers)
        writer.writeheader()

        for cid in ordered_ids:
            if cid not in matched_profiles:
                continue
                
            prof_data = matched_profiles[cid]
            score_data = scores_lookup[cid]
            
            profile = prof_data.get("profile", {})
            skills_list = prof_data.get("skills", [])
            
            # Extract top 10 skills by profile order
            top_skills = [s.get("name", "") for s in skills_list if isinstance(s, dict)][:10]
            skills_str = ", ".join(filter(None, top_skills))

            row_dict = {
                "candidate_id": cid,
                "anonymized_name": score_data.get("anonymized_name", profile.get("anonymized_name", "N/A")),
                "current_title": profile.get("current_title", "N/A"),
                "current_company": profile.get("current_company", "N/A"),
                "current_industry": profile.get("current_industry", "N/A"),
                "years_of_experience": profile.get("years_of_experience", "0.0"),
                "headline": profile.get("headline", "N/A"),
                "summary": profile.get("summary", "N/A"),
                "top_10_skills": skills_str,
                "final_score": score_data.get("final_score", 0.0),
                "semantic_similarity": score_data.get("semantic_similarity", 0.0),
                "title_relevance": score_data.get("title_relevance", 0.0),
                "experience_fit": score_data.get("experience_fit", 0.0),
                "retrieval_domain_score": score_data.get("retrieval_domain_score", 0.0),
                "vector_score": score_data.get("vector_score", 0.0),
                "production_ml_score": score_data.get("production_ml_score", 0.0),
                "product_company_score": score_data.get("product_company_score", 0.0),
                "recruitability_score": score_data.get("recruitability_score", 0.0)
            }
            writer.writerow(row_dict)
            final_ordered_rows.append(row_dict)

    # 4. Generate the Comprehensive ranking_audit.md Report
    print(f"Compiling metrics distribution and generating qualitative report: {AUDIT_REPORT}...")
    generate_audit_report(final_ordered_rows)
    print("System audit processing completed successfully!")


def assign_archetype(title, summary, skills):
    """Rule engine predicting candidate archetypes for qualitative evaluation."""
    combined = f"{title} {summary} {skills}".lower()
    if "recommend" in combined or "collaborative filtering" in combined:
        return "Recommendation Engineer"
    elif "search" in combined or "retrieval" in combined or "elasticsearch" in combined:
        return "Search Engineer"
    elif "applied ml" in combined or "fine-tune" in combined:
        return "Applied ML Engineer"
    elif "ai engineer" in combined or "llm" in combined:
        return "AI Engineer"
    elif "ml engineer" in combined or "machine learning engineer" in combined or "mlops" in combined:
        return "ML Engineer"
    elif "data engineer" in combined or "pipeline" in combined or "airflow" in combined:
        return "Data Engineer"
    elif "data scientist" in combined or "notebook" in combined:
        return "Data Scientist"
    elif "backend" in combined or "django" in combined or "api" in combined:
        return "Backend Engineer"
    elif "devops" in combined or "ci/cd" in combined:
        return "DevOps Engineer"
    else:
        return "Other"


def generate_audit_report(rows):
    total_candidates = len(rows)
    if total_candidates == 0:
        return

    # Extract metrics for aggregate distributions
    titles = [r["current_title"] for r in rows]
    industries = [r["current_industry"] for r in rows]
    companies = [r["current_company"] for r in rows]
    
    all_skills = []
    for r in rows:
        if r["top_10_skills"]:
            all_skills.extend([s.strip() for s in r["top_10_skills"].split(",")])

    yoe_list = []
    for r in rows:
        try:
            yoe_list.append(float(r["years_of_experience"]))
        except ValueError:
            pass

    avg_yoe = sum(yoe_list) / len(yoe_list) if yoe_list else 0.0

    title_counts = Counter(titles)
    industry_counts = Counter(industries)
    company_counts = Counter(companies)
    skill_counts = Counter(all_skills)

    with open(AUDIT_REPORT, mode='w', encoding='utf-8') as f:
        f.write("# 📊 Ranking System Integrity Audit Report\n\n")
        f.write("This automated audit validates the performance of the candidate ranking pipeline against the target criteria for the **Senior AI Engineer (Founding Team)** role.\n\n")
        
        f.write("## 📈 Quantitative System Metrics (Top 100)\n\n")
        f.write(f"- **Total Audited Leaders:** {total_candidates}\n")
        f.write(f"- **Average Profile Work Experience:** {avg_yoe:.2f} years\n\n")

        f.write("### 🔹 Title Aggregations\n")
        f.write("| Current Professional Title | Frequency Count |\n")
        f.write("| :--- | :--- |\n")
        for t, count in title_counts.most_common(15):
            f.write(f"| {t} | {count} |\n")

        f.write("\n### 🔹 Top Target Sourcing Companies\n")
        f.write("| Sourced Company Identity | Frequency Count |\n")
        f.write("| :--- | :--- |\n")
        for c, count in company_counts.most_common(10):
            f.write(f"| {c} | {count} |\n")

        f.write("\n### 🔹 Sourced Industry Segments\n")
        f.write("| Industry Domain Group | Frequency Count |\n")
        f.write("| :--- | :--- |\n")
        for ind, count in industry_counts.most_common(10):
            f.write(f"| {ind} | {count} |\n")

        f.write("\n### 🔹 Top Common Core Component Skills\n")
        f.write("| Skill Signature Tag | Combined Frequency |\n")
        f.write("| :--- | :--- |\n")
        for sk, count in skill_counts.most_common(15):
            f.write(f"| {sk} | {count} |\n")

        f.write("\n" + "—"*30 + "\n\n")
        f.write("## 🧐 Qualitative Analysis Report: Top 20 Candidates\n\n")
        f.write("Deep inspection tracking core alignment heuristics, matching criteria profiles, and pattern outliers.\n\n")

        for idx, r in enumerate(rows[:20], start=1):
            title = r["current_title"]
            summary = r["summary"]
            skills = r["top_10_skills"]
            
            # Use deterministic heuristic function to detect archetype profiles
            archetype = assign_archetype(title, summary, skills)
            
            # Identify suspicious keyword stuffers or mismatch titles
            is_suspicious = "No"
            flag_reason = ""
            
            # Anti-pattern check: Marketing, HR, or non-technical profiles that slipped through
            if any(bad in title.lower() for bad in ["marketing", "recruiter", "sales", "accountant", "manager"]) and "engineer" not in title.lower():
                is_suspicious = "YES (Role Disconnect)"
                flag_reason = "Non-technical title context."
            elif float(r["vector_score"]) < 0.25:
                is_suspicious = "YES (Low Context Match)"
                flag_reason = "Weak textual vector correlation against required systems design parameters."

            f.write(f"### Rank {idx}: [{r['candidate_id']}] — {r['anonymized_name']}\n")
            f.write(f"- **Current Title:** {title}\n")
            f.write(f"- **Current Company:** {r['current_company']} ({r['current_industry']})\n")
            f.write(f"- **Years of Experience:** {r['years_of_experience']} Years\n")
            f.write(f"- **Predicted Sourcing Archetype:** **{archetype}**\n")
            f.write(f"- **Heuristic Baseline Indicators:** Final Score: `{r['final_score']}` | Vector Match: `{r['vector_score']}` | Dev Scale Match: `{r['production_ml_score']}`\n")
            f.write(f"- **Anomalous Candidate Flag:** {is_suspicious}\n")
            
            if is_suspicious != "No":
                f.write(f"  - *Flag Justification:* {flag_reason}\n")
                
            f.write("\n**Evaluation Notes:**\n")
            if archetype in ["AI Engineer", "Applied ML Engineer", "Search Engineer", "Recommendation Engineer"]:
                f.write(f"Strong alignment on core systems with {r['years_of_experience']} years of experience. Demonstrated production experience (`production_ml_score`={r['production_ml_score']}) with background in relevant product-focused scale configurations.\n\n")
            elif archetype in ["Data Engineer", "Backend Engineer"]:
                f.write(f"Good system-level match. The profile exhibits architecture foundations required to scale core infrastructure components. Relevant pipeline and backend skills present.\n\n")
            else:
                f.write(f"Marginal alignment profile. Selected by the pipeline primarily due to explicit overlaps in standard data libraries, but may lack core background depth in structural retrieval architectures.\n\n")
            
            f.write("*" * 40 + "\n\n")


if __name__ == "__main__":
    run_validation_pipeline()