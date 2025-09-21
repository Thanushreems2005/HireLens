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
    page_icon="<HL>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Theme and Branding ---
custom_css = """
<style>
.stApp {
    background-color: #f8fafc;
}
.stSidebar {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}
.stButton>button {
    background-color: #3b82f6;
    color: white;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    padding: 0.75rem 1.5rem;
    font-size: 0.95rem;
    transition: all 0.2s;
}
.stButton>button:hover {
    background-color: #2563eb;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}
.stMetric {
    background-color: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.stDataFrame {
    background-color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #1e293b;
    font-weight: 700;
}
.stTextInput>div>div>input, .stTextArea>div>div>textarea {
    background-color: #ffffff;
    border: 2px solid #e2e8f0;
    border-radius: 10px;
    color: #1e293b;
    font-size: 0.95rem;
}
.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}
.stSelectbox>div>div>select {
    background-color: #ffffff;
    border: 2px solid #e2e8f0;
    border-radius: 10px;
    color: #1e293b;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Clean Modern Header ---
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 3rem 2rem; 
            border-radius: 20px; 
            margin-bottom: 2rem;
            text-align: center;'>
    <h1 style='color: white; font-size: 3rem; margin-bottom: 1rem; font-weight: 800;'>
        HireLens
    </h1>
    <p style='color: rgba(255,255,255,0.9); font-size: 1.3rem; margin-bottom: 0.5rem; font-weight: 400;'>
        Upload resumes and job descriptions to find the best candidates using AI-powered matching.
    </p>
    <p style='color: rgba(255,255,255,0.7); font-size: 1.1rem; margin: 0;'>
        Our system analyzes skills, experience, and relevance to rank applicants automatically.
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
        st.success(f"Analysis for {candidate_name} completed and saved to database.")
    except sqlite3.IntegrityError:
        st.info(f"Analysis for {candidate_name} is already in the database. Skipping.")

def run_full_analysis(uploaded_files, job_description, model, save_resumes=False):
    create_table()
    if not st.session_state.job_description:
        st.warning("Please load a Job Description first.")
        return

    existing_hashes = get_existing_hashes()
    # Process files from the data/resumes folder
    data_resumes_path = os.path.join(os.getcwd(), 'data', 'resumes')
    if os.path.exists(data_resumes_path):
        st.markdown("<h4 style='color: #1e293b;'>Processing from local folder...</h4>", unsafe_allow_html=True)
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
        st.markdown("<h4 style='color: #1e293b;'>Processing newly uploaded files...</h4>", unsafe_allow_html=True)
        if save_resumes:
            # Save to disk permanently
            resumes_dir = os.path.join(os.getcwd(), 'data', 'resumes')
            os.makedirs(resumes_dir, exist_ok=True)
            st.info("Resumes will be saved permanently and visible to all users.")
        else:
            st.info("Resumes are being processed in memory only - not saved to disk.")
        
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
                    st.success(f"Resume {uploaded_file.name} saved permanently.")
            
            if file_hash not in existing_hashes:
                process_resume(file_content, st.session_state.job_description, model, uploaded_file.name)
            else:
                st.info(f"{uploaded_file.name} is already in the database. Skipping.")

# --- Streamlit UI and Workflow ---
def main():
    # --- Clean Export Section ---
    st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 2rem;'>
        <h2 style='color: #1e293b; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;'>
            📊 Export Results
        </h2>
        <p style='color: #64748b; margin-bottom: 1.5rem;'>Download your analysis results for offline use or sharing</p>
    """, unsafe_allow_html=True)
    
    results_df = get_results()
    if not results_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=" Download CSV",
                data=csv,
                file_name="hirelens_results.csv",
                mime="text/csv",
                help="Download all results as a CSV file."
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
                    label="📄Download PDF",
                    data=pdf_bytes,
                    file_name="hirelens_results.pdf",
                    mime="application/pdf",
                    help="Download all results as a PDF file."
                )
            except ImportError:
                st.info("Install the 'fpdf' package to enable PDF export: pip install fpdf")
    else:
        st.info("No results to export. Please run an analysis first.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- Clean AI Assistant Section ---
    st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 2rem;'>
        <h2 style='color: #1e293b; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;'>
            🤖 AI Assistant
        </h2>
        <p style='color: #64748b; margin-bottom: 1.5rem;'>Get instant answers about hiring, recruitment strategies, and candidate evaluation</p>
    """, unsafe_allow_html=True)
    
    user_query = st.text_input("Ask anything about hiring and recruitment:", placeholder="e.g., How to evaluate technical candidates?")
    if user_query:
        if not API_KEY:
            st.warning("AI Assistant is currently unavailable. Please check configuration.")
        else:
            headers = {'Content-Type': 'application/json'}
            payload = {"contents": [{"parts": [{"text": user_query}]}]}
            try:
                response = requests.post(API_URL + API_KEY, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                # Try to extract AI response
                answer = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', None)
                if answer:
                    st.markdown(f"""
                    <div style='background: #f8fafc; border-left: 4px solid #3b82f6; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;'>
                        <p style='color: #1e293b; margin: 0; line-height: 1.6;'>{answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No response from AI Assistant.")
            except Exception as e:
                st.error(f"AI Assistant error: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)
                
    model = load_llm_model()
    load_spacy_model()
    create_table()

    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    
    # --- Clean Sidebar ---
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        if st.button("🗑️ Clear All Results", type="primary"):
            clear_results()
            st.success("All results cleared!")
        
        st.markdown("---")

        st.markdown("### 📝 Job Description")
        jd_file = st.file_uploader(
            "Upload JD File",
            type=["pdf", "txt"],
            key="jd_uploader",
            help="Upload job description as PDF or TXT"
        )

        jd_text_input = st.text_area(
            "Or paste text directly:",
            value="",
            height=150,
            key="jd_textarea",
            help="Paste the job description here"
        )

        st.markdown("### 📄 Resume Upload")
        
        save_resumes = st.checkbox(
            "💾 Save resumes permanently",
            value=False,
            help="Save uploaded files to disk for future use"
        )
        
        uploaded_resume_files = st.file_uploader(
            "Upload Resume Files",
            type=["pdf"],
            accept_multiple_files=True,
            key="resume_uploader",
            help="Upload multiple PDF resumes"
        )

        st.markdown("---")

        load_jd_clicked = st.button("✅ Load Job Description", use_container_width=True)
        if load_jd_clicked:
            try:
                if jd_text_input.strip():
                    jd_text = jd_text_input.strip()
                elif jd_file and jd_file.type == "application/pdf":
                    pdf_bytes = jd_file.read()
                    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                        jd_text = ""
                        for page in doc:
                            jd_text += page.get_text()
                elif jd_file:
                    jd_text = jd_file.read().decode("utf-8")
                else:
                    jd_text = ""
                st.session_state.job_description = jd_text
                st.success("Job Description loaded!")
            except Exception as e:
                st.error(f"Error loading job description: {e}")

        # Allow Run Analysis if JD is loaded
        jd_loaded = 'job_description' in st.session_state and st.session_state.job_description.strip()
        if st.button("🚀 Start Analysis", disabled=not jd_loaded, use_container_width=True):
            run_full_analysis(uploaded_resume_files, st.session_state.job_description, model, save_resumes)
            st.rerun()

        if save_resumes:
            st.caption("📌 Files saved to 'data/resumes' folder")
        else:
            st.caption("📌 Processing in memory only")
    
    # --- Top Performer Highlight ---
    results_df = get_results()
    if not results_df.empty and 'job_description' in st.session_state and st.session_state.job_description.strip():
        filtered_df = results_df[~results_df['resume_name'].str.lower().str.contains('professional summary|tarun kumar')].copy()
        import numpy as np
        filtered_df['resume_name'] = filtered_df['resume_name'].apply(lambda x: x if (isinstance(x, str) and x.strip()) else 'No Name')
        if not filtered_df.empty:
            top_performer = filtered_df.sort_values(by='relevance_score', ascending=False).iloc[0]
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        padding: 2.5rem; border-radius: 20px; margin-bottom: 2rem; 
                        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);'>
                <div style='text-align: center;'>
                    <div style='font-size: 4rem; margin-bottom: 1rem;'>🏆</div>
                    <h2 style='color: white; margin-bottom: 0.5rem; font-size: 2.5rem; font-weight: 800;'>
                        {top_performer['resume_name']}
                    </h2>
                    <h3 style='color: #d1fae5; margin-bottom: 1rem; font-size: 1.8rem;'>
                        Match Score: {top_performer['relevance_score']:.1f}%
                    </h3>
                    <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 12px; display: inline-block;'>
                        <p style='color: white; margin: 0; font-size: 1.1rem;'>
                            <strong>Key Skills:</strong> {top_performer['extracted_skills']}
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- Main Tabs ---
    tab1, tab2 = st.tabs(["📊 All Candidates", "🎯 Top 3 Matches"])

    with tab1:
        st.markdown("### 📋 Complete Candidate Analysis")
        results_df = get_results()
        import numpy as np
        results_df = results_df.copy()
        results_df['resume_name'] = results_df['resume_name'].apply(lambda x: x if (isinstance(x, str) and x.strip()) else 'No Name')
        
        # Clean metrics row
        if not results_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("👥 Total Candidates", len(results_df))
            with col2:
                avg_score = results_df['relevance_score'].astype(str).str.replace('%','').astype(float).mean()
                st.metric("📈 Average Score", f"{avg_score:.1f}%")
            with col3:
                excellent_count = len(results_df[results_df['match_level'] == 'Excellent match'])
                st.metric("⭐ Excellent Matches", excellent_count)
            with col4:
                partial_count = len(results_df[results_df['match_level'] == 'Partial match'])
                st.metric("💡 Partial Matches", partial_count)
        
        search_name = st.text_input("🔍 Search candidates:", placeholder="Enter name to filter...")
        display_df = results_df
        if search_name.strip():
            display_df = results_df[results_df['resume_name'].str.lower().str.contains(search_name.strip().lower())]
        
        if not display_df.empty:
            display_df['relevance_score'] = display_df['relevance_score'].astype(str).str.replace('%','').astype(float)
            display_df = display_df.sort_values(by='relevance_score', ascending=False)
            display_df['relevance_score'] = display_df['relevance_score'].apply(lambda x: f"{x:.1f}%")
            display_df['extracted_skills'] = display_df['extracted_skills'].apply(lambda x: ', '.join(json.loads(x)))
            display_df = display_df[['resume_name', 'relevance_score', 'match_level', 'extracted_skills', 'skill_gaps']]
            display_df.index = np.arange(1, len(display_df) + 1)
            st.dataframe(display_df, use_container_width=True)

    with tab2:
        st.markdown("### 🎯 Top 3 Best Matches")
        results_df = get_results()
        filtered_df = results_df[~results_df['resume_name'].str.lower().str.contains('professional summary|tarun kumar')].copy()
        filtered_df['relevance_score'] = filtered_df['relevance_score'].astype(str).str.replace('%','').astype(float)
        filtered_df = filtered_df.sort_values(by='relevance_score', ascending=False)
        
        if not filtered_df.empty:
            top_candidates = filtered_df.head(3)
            for i, (index, row) in enumerate(top_candidates.iterrows()):
                # Position-based styling
                if i == 0:  # 1st place
                    bg_color = "linear-gradient(135deg, #ffd700 0%, #ffed4e 100%)"
                    icon = "🥇"
                    text_color = "#92400e"
                elif i == 1:  # 2nd place
                    bg_color = "linear-gradient(135deg, #c0c0c0 0%, #e5e5e5 100%)"
                    icon = "🥈"
                    text_color = "#374151"
                else:  # 3rd place
                    bg_color = "linear-gradient(135deg, #cd7f32 0%, #d4915c 100%)"
                    icon = "🥉"
                    text_color = "#92400e"
                
                st.markdown(f"""
                <div style='background: {bg_color}; 
                            padding: 2rem; border-radius: 16px; margin-bottom: 1.5rem; 
                            box-shadow: 0 6px 20px rgba(0,0,0,0.15);'>
                    <div style='display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1rem;'>
                        <div style='font-size: 3rem;'>{icon}</div>
                        <div>
                            <h3 style='color: {text_color}; margin: 0; font-size: 1.8rem; font-weight: 700;'>
                                {row['resume_name']}
                            </h3>
                            <p style='color: {text_color}aa; margin: 0; font-size: 1.3rem; font-weight: 600;'>
                                Match Score: {row['relevance_score']:.1f}%
                            </p>
                        </div>
                    </div>
                    <div style='background: rgba(255,255,255,0.3); padding: 1.5rem; border-radius: 12px;'>
                        <div style='display: grid; grid-template-columns: 1fr; gap: 0.8rem;'>
                            <p style='color: {text_color}; margin: 0; font-weight: 600;'>
                                <strong>Match Level:</strong> <span style='color: {text_color}cc;'>{row['match_level']}</span>
                            </p>
                            <p style='color: {text_color}; margin: 0; font-weight: 600;'>
                                <strong>Key Skills:</strong> <span style='color: {text_color}cc;'>{row['extracted_skills']}</span>
                            </p>
                            <p style='color: {text_color}; margin: 0; font-weight: 600;'>
                                <strong>Areas to Develop:</strong> <span style='color: {text_color}cc;'>{row['skill_gaps']}</span>
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: white; padding: 3rem; border-radius: 16px; text-align: center; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 2px dashed #e2e8f0;'>
                <div style='font-size: 4rem; margin-bottom: 1rem;'>🎯</div>
                <h3 style='color: #64748b; margin-bottom: 0.5rem;'>No Candidates Yet</h3>
                <p style='color: #94a3b8; margin: 0;'>Upload resumes and run analysis to see top matches</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    create_table()
    main()