import streamlit as st
import json
import io
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.feature_extraction import rank_candidates_in_memory, extract_jd_text, MODEL_NAME

# Must be the first Streamlit command
st.set_page_config(
    page_title="Candidate Ranking Sandbox", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI enhancements
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "results" not in st.session_state:
    st.session_state.results = None
if "df_results" not in st.session_state:
    st.session_state.df_results = None

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(MODEL_NAME)

# --- Header ---
st.markdown('<div class="main-header">🧠 Intelligent Candidate Ranking Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a Job Description and a candidate JSONL sample to rank them semantically.</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("1. Job Description")
    jd_option = st.radio("JD Input Method", ["Upload .docx", "Paste Text"])
    jd_text = ""
    
    if jd_option == "Upload .docx":
        jd_file = st.file_uploader("Upload JD (.docx)", type=["docx"])
        if jd_file:
            jd_text = extract_jd_text(io.BytesIO(jd_file.read()))
    else:
        jd_text = st.text_area("Paste JD Text", height=150, placeholder="Paste job description here...")

    st.subheader("2. Candidate Data Sample")
    st.caption("Maximum 100 candidates (.jsonl)")
    candidates_file = st.file_uploader("Upload Candidates", type=["jsonl"])
    
    use_default = st.checkbox("Or use pre-loaded sample dataset")
    
    candidates_data = []
    
    if use_default:
        try:
            with open("data/sample_candidates.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        candidates_data.append(json.loads(line))
            st.success(f"Successfully loaded {len(candidates_data)} candidates from pre-loaded sample.")
        except FileNotFoundError:
            st.error("Pre-loaded sample file (data/sample_candidates.jsonl) not found.")
            candidates_data = []
    elif candidates_file:
        try:
            content = candidates_file.read().decode("utf-8")
            for line in content.splitlines():
                if line.strip():
                    candidates_data.append(json.loads(line))
            
            if len(candidates_data) > 100:
                st.error(f"Limit exceeded! Sandbox supports up to 100 candidates (Found {len(candidates_data)}).")
                candidates_data = [] # Reset
            elif len(candidates_data) > 0:
                st.success(f"Successfully loaded {len(candidates_data)} candidates.")
        except json.JSONDecodeError:
            st.error("Invalid JSONL format. Please check the file.")
            candidates_data = []

# --- Main Area Execution ---
col1, col2 = st.columns([1, 1])

# We only allow running if inputs are ready
can_run = bool(jd_text.strip()) and len(candidates_data) > 0

with col1:
    st.markdown("### Run Pipeline")
    if st.button("🚀 Run Ranking Engine", disabled=not can_run, use_container_width=True):
        st.session_state.results = None # Reset previous results
        st.session_state.df_results = None
        
        # UI Feedback during processing
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        status_text.info("Loading semantic embedding model...")
        progress_bar.progress(20)
        
        try:
            model = load_model()
            progress_bar.progress(50)
            status_text.info("Computing metadata heuristics and deep vector embeddings...")
            
            # Execute Pipeline
            results = rank_candidates_in_memory(candidates_data, jd_text, model)
            progress_bar.progress(90)
            
            if results:
                st.session_state.results = results
                st.session_state.df_results = pd.DataFrame(results)
                progress_bar.progress(100)
                status_text.success("Ranking successfully completed!")
            else:
                status_text.warning("No candidates were returned from the ranking.")
                progress_bar.empty()
                
        except Exception as e:
            status_text.error(f"An error occurred: {str(e)}")
            progress_bar.empty()

with col2:
    if not can_run:
        st.info("👈 Please provide both a Job Description and a Candidate Sample in the sidebar to begin.")
    elif st.session_state.results is None:
        st.info("Ready to rank. Click the 'Run Ranking Engine' button.")

# --- Results Presentation ---
if st.session_state.df_results is not None:
    df = st.session_state.df_results
    
    st.divider()
    st.markdown("### 🏆 Top Candidate Matches")
    
    # High-level metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Candidates Processed", len(df))
    if len(df) > 0:
        top_score = df.iloc[0]['score']
        m2.metric("Highest Alignment Score", f"{top_score:.4f}")
    
    # Display the dataframe cleanly
    display_cols = ["rank", "candidate_id", "score", "reasoning"]
    display_df = df.head(20)[display_cols]
    
    st.dataframe(
        display_df,
        use_container_width=False,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d", width=50),
            "candidate_id": st.column_config.TextColumn("Candidate ID", width=130),
            "score": st.column_config.NumberColumn("Score", format="%.4f", width=80),
            "reasoning": st.column_config.TextColumn("Matching Reasoning", width=800)
        }
    )
    
    # Download Button
    # Explicitly enforce UTF-8 encoding and UNIX line endings for strict validator compliance
    csv_bytes = df[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False, lineterminator='\n').encode('utf-8')
    
    st.download_button(
        label="📥 Download Submission CSV",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
        type="primary"
    )
