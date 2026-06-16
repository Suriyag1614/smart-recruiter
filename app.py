import streamlit as st
import json
import io
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.feature_extraction import rank_candidates_in_memory, extract_jd_text, MODEL_NAME

st.set_page_config(page_title="Candidate Ranking Sandbox", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

st.title("Intelligent Candidate Ranking Engine - Sandbox")
st.markdown("Upload a Job Description and a sample JSONL to see the ranking pipeline in action.")

# Section 1: Job Description
st.header("1. Job Description")
jd_option = st.radio("JD Input Method", ["Upload .docx", "Paste Text"])

jd_text = ""
if jd_option == "Upload .docx":
    jd_file = st.file_uploader("Upload JD (.docx)", type=["docx"])
    if jd_file:
        jd_text = extract_jd_text(io.BytesIO(jd_file.read()))
elif jd_option == "Paste Text":
    jd_text = st.text_area("Paste JD Text", height=200)

# Section 2: Candidate Data
st.header("2. Candidate Data Sample")
st.markdown("Maximum 100 candidates (JSONL format)")
candidates_file = st.file_uploader("Upload Candidates (.jsonl)", type=["jsonl"])

candidates_data = []
if candidates_file:
    try:
        content = candidates_file.read().decode("utf-8")
        for line in content.splitlines():
            if line.strip():
                candidates_data.append(json.loads(line))
        st.success(f"Loaded {len(candidates_data)} candidates.")
        
        if len(candidates_data) > 100:
            st.error(f"Error: Candidate limit exceeded. Sandbox only supports up to 100 candidates (Found {len(candidates_data)}).")
            candidates_data = [] # Reset to prevent processing
    except json.JSONDecodeError:
        st.error("Error: Invalid JSONL format. Please ensure each line is a valid JSON object.")
        candidates_data = []

# Section 3: Run Ranking
st.header("3. Run Ranking Pipeline")
if st.button("Run Ranking", type="primary"):
    if not jd_text.strip():
        st.error("Please provide a Job Description.")
    elif not candidates_data:
        st.error("Please provide a valid candidate JSONL file (≤ 100 candidates).")
    else:
        with st.spinner("Loading embedding model..."):
            try:
                model = load_model()
            except Exception as e:
                st.error(f"Failed to load model. Error: {str(e)}")
                model = None
                
        if model:
            with st.spinner("Running deep semantic ranking..."):
                try:
                    results = rank_candidates_in_memory(candidates_data, jd_text, model)
                    
                    if results:
                        st.success(f"Successfully processed {len(results)} candidates.")
                        
                        df_results = pd.DataFrame(results)
                        
                        st.subheader("Top 20 Candidates")
                        display_df = df_results.head(20)[["candidate_id", "rank", "score", "reasoning"]]
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Generate CSV for download
                        csv_buffer = io.StringIO()
                        df_results[["candidate_id", "rank", "score", "reasoning"]].to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="Download Ranked CSV",
                            data=csv_buffer.getvalue(),
                            file_name="ranked_candidates.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No candidates returned from ranking.")
                        
                except Exception as e:
                    st.error(f"An error occurred during ranking: {str(e)}")
