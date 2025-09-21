import streamlit as st
import streamlit as st
from streamlit import markdown
import sqlite3
import pandas as pd
import json
import os
import re
import fitz
from sentence_transformers import SentenceTransformer, util
import requests
import spacy
import hashlib

st.set_page_config(
    page_title="HireLens: AI-Powered Resume Evaluator",
    page_icon="HL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Black & White Theme ---
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background-color: #000000;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #ffffff;
}

.stSidebar {
    background-color: #111111;
    border-right: 1px solid #333333;
}

.stSidebar .stMarkdown {
    color: #ffffff !important;
}

.stSidebar .stMarkdown h1, .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3 {
    color: #ffffff !important;
    font-weight: 600;
}

.stSidebar .stMarkdown p {
    color: #cccccc !important;
}

.stSidebar label {
    color: #ffffff !important;
    font-weight: 500;
}

.stSidebar .stFileUploader label {
    color: #ffffff !important;
}

.stSidebar .stFileUploader div {
    color: #cccccc !important;
}

.stSidebar .stCheckbox label {
    color: #ffffff !important;
}

.stSidebar .stCheckbox span {
    color: #ffffff !important;
}

.stSidebar .caption {
    color: #888888 !important;
}

.stButton>button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 8px;
    font-weight: 600;
    border: 2px solid #ffffff;
    padding: 0.75rem 1.5rem;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    font-family: 'Inter', sans-serif;
}

.stButton>button:hover {
    background-color: #f5f5f5 !important;
    color: #000000 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.2);
}

.stButton>button:focus {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-color: #ffffff !important;
    outline: none;
    box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.2);
}

.stButton>button:disabled {
    background-color: #333333 !important;
    color: #666666 !important;
    border-color: #333333 !important;
}

.stDownloadButton>button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #ffffff;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
}

.stDownloadButton>button:hover {
    background-color: #f5f5f5 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.2);
}

.stMetric {
    background-color: #111111;
    padding: 1.5rem;
    border: 1px solid #333333;
    border-radius: 12px;
    color: #ffffff;
    text-align: center;
}

.stMetric label {
    color: #cccccc !important;
    font-size: 0.9rem;
    font-weight: 500;
}

.stMetric div {
    color: #ffffff !important;
    font-size: 2rem;
    font-weight: 700;
}

.stDataFrame {
    background-color: #111111;
    border: 1px solid #333333;
    border-radius: 12px;
    color: #ffffff;
}

.stDataFrame table {
    background-color: #111111 !important;
    color: #ffffff !important;
}

.stDataFrame thead th {
    background-color: #222222 !important;
    color: #ffffff !important;
    font-weight: 600;
    border-bottom: 2px solid #333333;
}

.stDataFrame tbody td {
    background-color: #111111 !important;
    color: #ffffff !important;
    border-bottom: 1px solid #333333;
}

.stDataFrame tbody tr:hover {
    background-color: #1a1a1a !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #ffffff !important;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
}

.stMarkdown h1 {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.stMarkdown h2 {
    font-size: 1.8rem;
    margin-bottom: 0.8rem;
}

.stMarkdown h3 {
    font-size: 1.4rem;
    margin-bottom: 0.6rem;
}

.stMarkdown p {
    color: #cccccc !important;
    line-height: 1.6;
    font-size: 1rem;
}

.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    background-color: #111111 !important;
    border: 2px solid #333333 !important;
    border-radius: 8px;
    color: #ffffff !important;
    font-size: 0.95rem;
    padding: 0.75rem;
}

.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
    border-color: #ffffff !important;
    outline: none;
    box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.1);
}

.stTextInput label, .stTextArea label {
    color: #ffffff !important;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

.stSelectbox>div>div>select {
    background-color: #111111 !important;
    border: 2px solid #333333 !important;
    border-radius: 8px;
    color: #ffffff !important;
    padding: 0.75rem;
}

.stSelectbox label {
    color: #ffffff !important;
    font-weight: 500;
}

.stFileUploader {
    color: #ffffff !important;
}

.stFileUploader label {
    color: #ffffff !important;
    font-weight: 500;
}

.stFileUploader div {
    color: #cccccc !important;
}

.stFileUploader section {
    border: 2px dashed #333333 !important;
    background-color: #111111 !important;
    color: #ffffff !important;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.stFileUploader section:hover {
    border-color: #ffffff !important;
    background-color: #1a1a1a !important;
}

.stFileUploader section div {
    color: #cccccc !important;
}

.stCheckbox label {
    color: #ffffff !important;
    font-weight: 500;
}

.stCheckbox span {
    color: #ffffff !important;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #111111;
    border-radius: 12px;
    padding: 0.25rem;
    border: 1px solid #333333;
}

.stTabs [data-baseweb="tab"] {
    color: #cccccc !important;
    background-color: transparent;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-weight: 600;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #222222 !important;
    color: #ffffff !important;
}

div[data-testid="stSidebarNav"] {
    color: #ffffff !important;
}

div[data-testid="stSidebarNav"] span {
    color: #ffffff !important;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #111111;
}

::-webkit-scrollbar-thumb {
    background: #333333;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #555555;
}

/* Loading spinner */
.stSpinner {
    color: #ffffff !important;
}

/* Success/Error/Warning messages */
.stSuccess {
    background-color: #0f4c28 !important;
    color: #ffffff !important;
    border: 1px solid #28a745;
}

.stError {
    background-color: #4c0f0f !important;
    color: #ffffff !important;
    border: 1px solid #dc3545;
}

.stWarning {
    background-color: #4c3d0f !important;
    color: #ffffff !important;
    border: 1px solid #ffc107;
}

.stInfo {
    background-color: #0f2a4c !important;
    color: #ffffff !important;
    border: 1px solid #17a2b8;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Modern Header ---
st.markdown("""
<div style='background: linear-gradient(135deg, #111111 0%, #222222 100%); padding: 3rem 2rem; border: 1px solid #333333; border-radius: 16px; margin-bottom: 2rem; text-align: center;'>
    <h1 style='color: #ffffff; font-size: 3.5rem; margin-bottom: 1rem; font-weight: 700; letter-spacing: -0.02em;'>
        HireLens
    </h1>
    <p style='color: #cccccc; font-size: 1.3rem; margin-bottom: 0.5rem; font-weight: 400;'>
        AI-Powered Resume Evaluation Platform
    </p>
    <p style='color: #888888; font-size: 1.1rem; margin: 0; font-weight: 300;'>
        Advanced candidate matching using semantic analysis and machine learning
    </p>
</div>
""", unsafe_allow_html=True)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="
API_KEY = "AIzaSyDItolbJRFfyrf-WAo3dnM-_8dxQcNMFP4"

# --- Database Configuration ---
DATABASE_PATH = 'data/results.db'

# --- Backend Functions ---
@st.cache_resource
def load_llm_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading LLM model: {e}")
        return None

@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.warning("Downloading spaCy model 'en_core_web_sm'...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        return nlp

def create_table():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY,
            resume_name TEXT NOT NULL,
            relevance_score REAL NOT NULL,
            analysis_summary TEXT,
            extracted_skills TEXT,
            match_level TEXT,
            skill_gaps TEXT,
            file_hash TEXT UNIQUE
        )
    ''')
    # Migration: add columns if missing (for existing DBs)
    cursor.execute("PRAGMA table_info(analysis_results)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'match_level' not in columns:
        try:
            cursor.execute('ALTER TABLE analysis_results ADD COLUMN match_level TEXT')
        except Exception:
            pass
    if 'skill_gaps' not in columns:
        try:
            cursor.execute('ALTER TABLE analysis_results ADD COLUMN skill_gaps TEXT')
        except Exception:
            pass
    conn.commit()
    conn.close()

def clear_results():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM analysis_results')
    conn.commit()
    conn.close()

def insert_result(resume_name, relevance_score, analysis_summary, extracted_skills, match_level, skill_gaps, file_hash):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO analysis_results (resume_name, relevance_score, analysis_summary, extracted_skills, match_level, skill_gaps, file_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (resume_name, relevance_score, analysis_summary, extracted_skills, match_level, skill_gaps, file_hash))
    conn.commit()
    conn.close()

def extract_skills_from_text(text):
    """Extract skills using keyword matching"""
    # Updated skills list matching your parser.py
    all_skills = [
        'Python', 'SQL', 'Power BI', 'Tableau', 'Pandas', 'NumPy',
        'Matplotlib', 'Seaborn', 'Excel', 'Web Scraping', 'EDA',
        'Data Cleaning', 'Statistics', 'MySQL', 'PostgreSQL',
        'Machine Learning', 'Scikit-learn', 'Git', 'Jupyter',
        'Data Visualization', 'DAX'
    ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in all_skills:
        if skill.lower() in text_lower:
            if skill not in found_skills:
                found_skills.append(skill)
    return found_skills

def parse_resume_advanced(file_content):
    nlp = load_spacy_model()
    if not nlp:
        return None
    try:
        # Extract text from PDF content
        doc = fitz.open(stream=file_content, filetype="pdf")
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()

        # Use spaCy for processing
        doc_nlp = nlp(raw_text)

        # Extract skills using the same logic as parser.py
        skills_text = ""
        text_lower = raw_text.lower()

        # Try to find skills section
        skills_match = re.search(r'skills(.*?)projects|technical skills(.*?)projects|skills(.*?)education', text_lower, re.DOTALL)
        if skills_match:
            skills_text = skills_match.group(1) or skills_match.group(2) or skills_match.group(3)

        # Extract skills using the dedicated function
        found_skills = extract_skills_from_text(skills_text or raw_text)

        # Improved name extraction
        name = ""
        # Only consider the first 10 lines for name extraction
        lines = raw_text.splitlines()[:10]
        name_candidates = []
        for line in lines:
            # Remove extra spaces and check for two capitalized words
            line = line.strip()
            match = re.match(r'^([A-Z][a-z]+ [A-Z][a-z]+)$', line)
            if match:
                candidate = match.group(1)
                # Filter out common non-name phrases
                if candidate.lower() not in [
                    'professional summary', 'core skills', 'phone no', 'aspiring data', 'summary', 'resume', 'curriculum vitae', 'contact info', 'address', 'email', 'profile', 'objective', 'career objective', 'personal details', 'education', 'skills', 'projects', 'work experience', 'experience', 'certifications', 'achievements', 'interests', 'hobbies', 'languages', 'references', 'declaration', 'date', 'place', 'signature'
                ]:
                    name_candidates.append(candidate)
        if name_candidates:
            name = name_candidates[0]
        else:
            # Fallback: use spaCy NER for PERSON entities in first 10 lines
            doc_first = nlp('\n'.join(lines))
            for ent in doc_first.ents:
                if ent.label_ == 'PERSON' and len(ent.text.split()) == 2:
                    name = ent.text
                    break
        # Extract projects
        projects_text = ""
        projects_match = re.search(r'projects(.*?)education|projects(.*?)experience|projects(.*?)certifications', text_lower, re.DOTALL)
        if projects_match:
            projects_text = projects_match.group(1) or projects_match.group(2) or projects_match.group(3)
        parsed_data = {
            "name": name,
            "raw_text": raw_text,
            "skills": found_skills,
            "projects_summary": projects_text.strip()
        }
        return parsed_data
    except Exception as e:
        st.error(f"Error parsing resume: {e}")
        return None

def analyze_resume_semantic(resume_text, job_description, model):
    if not model:
        return 0, "Model not loaded."
    try:
        resume_embedding = model.encode(resume_text.lower(), convert_to_tensor=True)
        jd_embedding = model.encode(job_description.lower(), convert_to_tensor=True)
        cosine_score = util.cos_sim(resume_embedding, jd_embedding).item()
        relevance_score = (cosine_score + 1) / 2 * 100
        analysis_summary = f"Semantic score based on AI embeddings."
        return relevance_score, analysis_summary
    except Exception as e:
        st.error(f"Error during semantic analysis: {e}")
        return 0, "Error during analysis."

def generate_skill_match_feedback(job_description, extracted_skills):
    required_skills = [
        'python', 'sql', 'power bi', 'tableau', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'excel',
        'web scraping', 'eda', 'data cleaning', 'statistics', 'mysql', 'postgresql', 'machine learning',
        'scikit-learn', 'git', 'jupyter', 'data visualization', 'dax', 'beautifulsoup'
    ]
    candidate_skills = set([s.lower() for s in extracted_skills])
    matched = sum(1 for s in required_skills if s in candidate_skills)
    ratio = matched / len(required_skills)
    if ratio > 0.7:
        match_level = "Excellent match"
    elif ratio > 0.4:
        match_level = "Partial match"
    else:
        match_level = "Low match"
    missing = [s for s in required_skills if s not in candidate_skills]
    if missing:
        skill_gaps = ', '.join(missing)
    else:
        skill_gaps = "No significant skill gaps."
    return match_level, skill_gaps

def get_results():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        query = "SELECT resume_name, relevance_score, analysis_summary, extracted_skills, match_level, skill_gaps FROM analysis_results"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching data from database: {e}")
        return pd.DataFrame()
    

def get_file_hash(file_content):
    hasher = hashlib.sha256()
    hasher.update(file_content)
    return hasher.hexdigest()

def get_existing_hashes():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        query = "SELECT file_hash FROM analysis_results"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return set(df['file_hash'])
    except:
        return set()

def process_resume(file_content, job_description, model, resume_name):
    parsed_data = parse_resume_advanced(file_content)
    if not parsed_data or not parsed_data['raw_text']:
        return
    relevance_score, analysis_summary = analyze_resume_semantic(parsed_data['raw_text'], job_description, model)
    match_level, skill_gaps = generate_skill_match_feedback(job_description, parsed_data['skills'])
    candidate_name = parsed_data['name'] if parsed_data['name'] else resume_name
    try:
        insert_result(candidate_name, relevance_score, analysis_summary, json.dumps(parsed_data['skills']), match_level, skill_gaps, get_file_hash(file_content))
        st.success(f"✅ Analysis for **{candidate_name}** completed and saved to database.")
    except sqlite3.IntegrityError:
        st.info(f"ℹ️ Analysis for **{candidate_name}** is already in the database. Skipping.")

def run_full_analysis(uploaded_files, job_description, model, save_resumes=False):
    create_table()
    if not st.session_state.job_description:
        st.warning("⚠️ Please load a Job Description first.")
        return

    existing_hashes = get_existing_hashes()
    # Process files from the data/resumes folder
    data_resumes_path = os.path.join(os.getcwd(), 'data', 'resumes')
    if os.path.exists(data_resumes_path):
        st.markdown("<h4 style='color: #ffffff; font-weight: 600;'>🔄 Processing from local folder...</h4>", unsafe_allow_html=True)
        for filename in os.listdir(data_resumes_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(data_resumes_path, filename)
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                file_hash = get_file_hash(file_content)
                if file_hash not in existing_hashes:
                    process_resume(file_content, st.session_state.job_description, model, filename)

    # Process newly uploaded files
    if uploaded_files:
        st.markdown("<h4 style='color: #ffffff; font-weight: 600;'>📁 Processing newly uploaded files...</h4>", unsafe_allow_html=True)
        if save_resumes:
            # Save to disk permanently
            resumes_dir = os.path.join(os.getcwd(), 'data', 'resumes')
            os.makedirs(resumes_dir, exist_ok=True)
            st.info("💾 Resumes will be saved permanently and visible to all users.")
        else:
            st.info("🔄 Resumes are being processed in memory only - not saved to disk.")
        
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.read()
            uploaded_file.seek(0)
            file_hash = get_file_hash(file_content)
            
            # Save to disk only if the option is enabled
            if save_resumes:
                local_resume_path = os.path.join(resumes_dir, uploaded_file.name)
                if not os.path.exists(local_resume_path):
                    with open(local_resume_path, 'wb') as f:
                        f.write(file_content)
                    st.success(f"✅ Resume **{uploaded_file.name}** saved permanently.")
            
            if file_hash not in existing_hashes:
                process_resume(file_content, st.session_state.job_description, model, uploaded_file.name)
            else:
                st.info(f"ℹ️ **{uploaded_file.name}** is already in the database. Skipping.")

# --- Streamlit UI and Workflow ---
def main():
    # --- Export Section ---
    st.markdown("""
    <div style='background: linear-gradient(135deg, #111111 0%, #1a1a1a 100%); padding: 2rem; border: 1px solid #333333; border-radius: 16px; margin-bottom: 2rem;'>
        <h2 style='color: #ffffff; margin-bottom: 1rem; font-size: 1.8rem; font-weight: 700;'>
            📊 Export Results
        </h2>
        <p style='color: #cccccc; margin-bottom: 1.5rem; font-size: 1.1rem; font-weight: 400;'>Download your analysis results for offline use or sharing</p>
    """, unsafe_allow_html=True)
    
    results_df = get_results()
    if not results_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="hirelens_results.csv",
                mime="text/csv"
            )
        with col2:
            try:
                import io
                from fpdf import FPDF
                class PDF(FPDF):
                    def header(self):
                        self.set_font('Arial', 'B', 12)
                        self.cell(0, 10, 'HireLens Resume Analysis Results', 0, 1, 'C')
                pdf = PDF()
                pdf.add_page()
                pdf.set_font('Arial', '', 10)
                for i, row in results_df.iterrows():
                    pdf.cell(0, 10, f"{row['resume_name']} | Score: {row['relevance_score']} | Skills: {row['extracted_skills']}", ln=1)
                pdf_bytes = pdf.output(dest='S').encode('latin1')
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_bytes,
                    file_name="hirelens_results.pdf",
                    mime="application/pdf"
                )
            except ImportError:
                st.info("💡 Install the 'fpdf' package to enable PDF export: pip install fpdf")
    else:
        st.markdown('<p style="color: #888888; font-size: 1rem; text-align: center; padding: 2rem;">No results to export. Please run an analysis first.</p>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- AI Assistant Section ---
    st.markdown("""
    <div style='background: linear-gradient(135deg, #111111 0%, #1a1a1a 100%); padding: 2rem; border: 1px solid #333333; border-radius: 16px; margin-bottom: 2rem;'>
        <h2 style='color: #ffffff; margin-bottom: 1rem; font-size: 1.8rem; font-weight: 700;'>
            🤖 AI Assistant
        </h2>
        <p style='color: #cccccc; margin-bottom: 1.5rem; font-size: 1.1rem; font-weight: 400;'>Get instant answers about hiring, recruitment strategies, and candidate evaluation</p>
    """, unsafe_allow_html=True)
    
    user_query = st.text_input("Ask anything about hiring and recruitment:", placeholder="e.g., How to evaluate technical candidates?")
    if user_query:
        if not API_KEY:
            st.warning("⚠️ AI Assistant is currently unavailable. Please check configuration.")
        else:
            headers = {'Content-Type': 'application/json'}
            payload = {"contents": [{"parts": [{"text": user_query}]}]}
            try:
                response = requests.post(API_URL + API_KEY, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                answer = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', None)
                if answer:
                    st.markdown(f"""
                    <div style='background: #222222; border-left: 4px solid #ffffff; padding: 1.5rem; border-radius: 8px; margin-top: 1.5rem;'>
                        <p style='color: #ffffff; margin: 0; line-height: 1.6; font-size: 1rem;'>{answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ No response from AI Assistant.")
            except Exception as e:
                st.error(f"❌ AI Assistant error: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)
                
    model = load_llm_model()
    load_spacy_model()
    create_table()

    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        if st.button("🗑️ Clear All Results", type="primary"):
            clear_results()
            st.success("✅ All results cleared!")
        
        st.markdown