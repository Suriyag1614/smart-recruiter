import json
from collections import Counter
import statistics
from datetime import datetime

# Define file paths
INPUT_JSONL = "data/candidates.jsonl"
OUTPUT_DIR = "outputs/eda"

# Create output directory if it doesn't exist
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_date_parse(date_str):
    """Safely parse date strings for statistics."""
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str).split('T')[0], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None

def run_eda(file_path):
    print(f"🚀 Starting EDA on {file_path} using custom schema parser...")

    # Counters for categorical data
    experience_counts = Counter()
    country_counts = Counter()
    title_counts = Counter()
    industry_counts = Counter()
    company_counts = Counter()
    skill_counts = Counter()
    notice_period_counts = Counter()

    # Metrics for flags and stats
    total_count = 0
    open_to_work_count = 0
    recruiter_responses = []
    active_dates = []

    # Stream the file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                total_count += 1

                # Extract primary nested dictionaries
                profile = record.get("profile", {})
                signals = record.get("redrob_signals", {})
                skills_list = record.get("skills", [])

                # 1. Experience (Mapped from profile -> years_of_experience)
                # Grouping into whole numbers/buckets makes distribution insights cleaner
                yoe = profile.get("years_of_experience")
                if yoe is not None:
                    try:
                        exp_bucket = f"{int(float(yoe))} years"
                        experience_counts[exp_bucket] += 1
                    except (ValueError, TypeError):
                        pass
                
                # Notice Period (Mapped from redrob_signals -> notice_period_days)
                np_days = signals.get("notice_period_days")
                if np_days is not None:
                    notice_period_counts[f"{np_days} days"] += 1

                # 2. Geography (Mapped from profile -> country)
                country = profile.get("country")
                if country:
                    country_counts[country] += 1

                # 3. Current Work Details (Mapped from profile sub-keys)
                title = profile.get("current_title")
                if title:
                    title_counts[title] += 1
                    
                industry = profile.get("current_industry")
                if industry:
                    industry_counts[industry] += 1
                    
                company = profile.get("current_company")
                if company:
                    company_counts[company] += 1

                # 4. Skills Processing (Extracting 'name' from the list of skill dicts)
                if isinstance(skills_list, list):
                    for skill_item in skills_list:
                        if isinstance(skill_item, dict):
                            skill_name = skill_item.get("name")
                            if skill_name:
                                skill_counts[skill_name] += 1

                # 5. Open to Work Flag (Mapped from redrob_signals -> open_to_work_flag)
                if signals.get("open_to_work_flag") in [True, "true", "True", 1]:
                    open_to_work_count += 1

                # 6. Recruiter Response Rate (Mapped from redrob_signals -> recruiter_response_rate)
                # Multiplying by 100 converts decimal fields (e.g., 0.34) to percentages (34.0%)
                rrr = signals.get("recruiter_response_rate")
                if rrr is not None:
                    try:
                        recruiter_responses.append(float(rrr) * 100)
                    except (ValueError, TypeError):
                        pass

                # 7. Last Active Date (Mapped from redrob_signals -> last_active_date)
                parsed_date = safe_date_parse(signals.get("last_active_date"))
                if parsed_date:
                    active_dates.append(parsed_date)

                # Progress tracker for your large file
                if total_count % 100000 == 0:
                    print(f"Processed {total_count} records...")

            except json.JSONDecodeError:
                continue

    # --- Print Terminal Summary Statistics ---
    print("\n" + "="*40)
    print("📊 EDA SUMMARY STATISTICS")
    print("="*40)
    print(f"Total Candidate Count: {total_count:,}")

    if total_count > 0:
        otw_pct = (open_to_work_count / total_count) * 100
        print(f"Open-to-Work Percentage: {otw_pct:.2f}%")
    else:
        print("Open-to-Work Percentage: N/A")

    print("\n🔹 Recruiter Response Rate Stats:")
    if recruiter_responses:
        print(f"  - Mean:   {statistics.mean(recruiter_responses):.2f}%")
        print(f"  - Median: {statistics.median(recruiter_responses):.2f}%")
        print(f"  - Min:    {min(recruiter_responses):.2f}%")
        print(f"  - Max:    {max(recruiter_responses):.2f}%")
    else:
        print("  - No data available")

    print("\n🔹 Last Active Date Stats:")
    if active_dates:
        print(f"  - Earliest Active: {min(active_dates).strftime('%Y-%m-%d')}")
        print(f"  - Latest Active:   {max(active_dates).strftime('%Y-%m-%d')}")
    else:
        print("  - No data available")

    print("\n🔹 Experience Distribution (Top 5):")
    for exp, count in experience_counts.most_common(5):
        print(f"  - {exp}: {count:,}")

    print("\n🔹 Notice Period Distribution (Top 5):")
    for np, count in notice_period_counts.most_common(5):
        print(f"  - {np}: {count:,}")

    print("-" * 40)

    # --- Deferred Pandas Import for Safe Exports ---
    print("💾 Exporting distributions to CSV files...")
    import pandas as pd

    exports = {
        OUTPUT_DIR + "/title_distribution.csv": title_counts,
        OUTPUT_DIR + "/company_distribution.csv": company_counts,
        OUTPUT_DIR + "/industry_distribution.csv": industry_counts,
        OUTPUT_DIR + "/skill_distribution.csv": skill_counts,
        OUTPUT_DIR + "/country_distribution.csv": country_counts
    }

    for filename, counter_data in exports.items():
        # Transforms the counter objects into clean tables efficiently
        df = pd.DataFrame(counter_data.most_common(), columns=["Value", "Count"])
        df.to_csv(filename, index=False)
        print(f"  ✅ Saved {filename} ({len(df):,} unique rows)")

    print("\n🎉 All processes completed successfully!")

if __name__ == "__main__":
    run_eda(INPUT_JSONL)