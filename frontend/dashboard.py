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
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Theme and Branding ---
custom_css = """
<style>
.stApp {
    background-color: #ffffff;
}
.stSidebar {
    background-color: #f8f9fa;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    font-weight: bold;
    border: none;
}
.stButton>button:hover {
    background-color: #1d4ed8;
}
.stMetric {
    color: #2563eb;
}
.stDataFrame {
    background-color: #ffffff;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #1e293b;
}
.stTextInput>div>div>input {
    background-color: #f8f9fa;
    color: #1e293b;
}
.stTextArea>div>div>textarea {
    background-color: #f8f9fa;
    color: #1e293b;
}
.stSelectbox>div>div>select {
    background-color: #f8f9fa;
    color: #1e293b;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Branding Header ---
st.markdown("""
<div style='display: flex; align-items: center; gap: 1rem; background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
    <img src='https://img.icons8.com/ios-filled/100/ffffff/resume.png' width='48' height='48'/>
    <div>
        <h1 style='margin-bottom: 0; color: white;'>HireLens</h1>
        <p style='margin: 0; color: #e2e8f0; font-size: 1.1rem;'>Upload resumes and job descriptions to find the best candidates using AI-powered matching.</p>
        <p style='margin: 0; color: #cbd5e1; font-size: 0.9rem;'>Our system analyzes skills, experience, and relevance to rank applicants automatically.</p>
    </div>
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
    # --- Export Options ---
    st.markdown("""
    <div style='background-color: #f1f5f9; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2563eb; margin-bottom: 2rem;'>
        <h3 style='color:#2563eb; margin-bottom: 0.5rem;'>📊 Export Results</h3>
        <p style='color:#64748b; margin-bottom: 0;'>Download the analysis results for offline use or sharing.</p>
    </div>
    """, unsafe_allow_html=True)
    results_df = get_results()
    if not results_df.empty:
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="hirelens_results.csv",
            mime="text/csv",
            help="Download all results as a CSV file."
        )
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
                label="📄 Download as PDF",
                data=pdf_bytes,
                file_name="hirelens_results.pdf",
                mime="application/pdf",
                help="Download all results as a PDF file."
            )
        except ImportError:
            st.info("Install the 'fpdf' package to enable PDF export: pip install fpdf")
    else:
        st.info("No results to export. Please run an analysis first.")
    
    # --- AI Assistant Section ---
    st.markdown("""
    <div style='background-color: #f1f5f9; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2563eb; margin-bottom: 2rem;'>
        <h3 style='color:#2563eb; margin-bottom: 0.5rem;'>🤖 AI Assistant</h3>
        <p style='color:#64748b; margin-bottom: 0;'>Ask any question about resumes, recruitment, or career guidance.</p>
    </div>
    """, unsafe_allow_html=True)
    user_query = st.text_input("Ask anything about hiring and recruitment:", "")
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
                    <div style='background-color: #e0f2fe; padding: 1rem; border-radius: 8px; border-left: 4px solid #0288d1;'>
                        <p style='color: #01579b; margin: 0;'>{answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No response from AI Assistant.")
            except Exception as e:
                st.error(f"AI Assistant error: {e}")
                
    model = load_llm_model()
    load_spacy_model()
    create_table()

    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    
    # --- Sidebar for Inputs ---
    with st.sidebar:
        if st.button("🗑️ Clear Results", type="primary"):
            clear_results()
            st.success("All results cleared!")

        st.markdown("### 📝 Step 1: Upload Job Description")
        jd_file = st.file_uploader(
            "Upload JD (PDF or TXT)",
            type=["pdf", "txt"],
            key="jd_uploader",
            help="Upload the job description as a PDF or TXT file."
        )

        jd_text_input = st.text_area(
            "Or paste/type the Job Description here:",
            value="",
            height=200,
            key="jd_textarea",
            help="You can type or paste the job description directly. This will override any uploaded file."
        )

        st.markdown("### 📄 Step 2: Upload Resumes")
        
        # Add storage option toggle
        save_resumes = st.checkbox(
            "💾 Save uploaded resumes permanently",
            value=False,
            help="If checked, uploaded resumes will be saved to disk and visible to all users. If unchecked, resumes are processed in memory only."
        )
        
        uploaded_resume_files = st.file_uploader(
            "Upload Resumes (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="resume_uploader",
            help="You can upload multiple PDF resumes at once."
        )

        load_jd_clicked = st.button("✅ Load Job Description")
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
                st.success("Job Description loaded successfully!")
            except Exception as e:
                st.error(f"Unexpected error loading job description: {e}")

        # Allow Run Analysis if JD is loaded (resumes optional)
        jd_loaded = 'job_description' in st.session_state and st.session_state.job_description.strip()
        if st.button("🚀 Run Analysis", disabled=not jd_loaded, help="Click to analyze resumes in the local folder and any uploaded resumes against the loaded job description."):
            run_full_analysis(uploaded_resume_files, st.session_state.job_description, model, save_resumes)
            st.rerun()

        # Update caption based on storage option
        if save_resumes:
            st.caption("📌 Note: Uploaded resumes will be saved to the 'data/resumes' folder and visible to all users.")
        else:
            st.caption("📌 Note: Uploaded resumes are processed in memory only and not saved to disk.")
    
    # --- Highlight Top Performer (only after analysis) ---
    results_df = get_results()
    # Only show results if a JD is loaded and resumes have been analyzed
    if not results_df.empty and 'job_description' in st.session_state and st.session_state.job_description.strip():
        # Filter out placeholder/empty names
        filtered_df = results_df[~results_df['resume_name'].str.lower().str.contains('professional summary|tarun kumar')].copy()
        # Show 'No Name' for missing names (fix for None/empty/NaN)
        import numpy as np
        filtered_df['resume_name'] = filtered_df['resume_name'].apply(lambda x: x if (isinstance(x, str) and x.strip()) else 'No Name')
        if not filtered_df.empty:
            top_performer = filtered_df.sort_values(by='relevance_score', ascending=False).iloc[0]
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 2rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(16, 185, 129, 0.1);'>
                <div style='display: flex; align-items: center; gap: 1rem;'>
                    <div style='font-size: 3rem;'>🏆</div>
                    <div>
                        <h2 style='color:white; margin-bottom: 0.5rem;'>Top Performer: {top_performer['resume_name']}</h2>
                        <h3 style='color:#d1fae5; margin-bottom: 0.5rem;'>Match Score: {top_performer['relevance_score']:.1f}%</h3>
                        <p style='color:#a7f3d0; margin: 0;'><strong>Key Skills:</strong> {top_performer['extracted_skills']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        if not ('job_description' in st.session_state and st.session_state.job_description.strip()):
            st.markdown("""
            <div style='background-color: #fef3c7; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #f59e0b; margin-bottom: 2rem;'>
                <p style='color: #92400e; margin: 0;'><strong>⚠️ Getting Started:</strong> Please load a Job Description and click 'Run Analysis' to see results.</p>
            </div>
            """, unsafe_allow_html=True)
        elif results_df.empty:
            st.markdown("""
            <div style='background-color: #e0f2fe; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #0288d1; margin-bottom: 2rem;'>
                <p style='color: #01579b; margin: 0;'><strong>ℹ️ No Results:</strong> No analysis results found. Please run analysis after uploading resumes or placing them in the data/resumes folder.</p>
            </div>
            """, unsafe_allow_html=True)

    # --- Main Content Area ---
    tab1, tab2 = st.tabs(["📊 Dashboard", "🎯 Top Candidates"])

    # Dashboard Tab: Show all applicants with count, filter out placeholder/empty names
    with tab1:
        st.markdown("### 📋 All Applicants")
        results_df = get_results()
        # Show all resumes, including those with missing/placeholder names
        import numpy as np
        results_df = results_df.copy()
        results_df['resume_name'] = results_df['resume_name'].apply(lambda x: x if (isinstance(x, str) and x.strip()) else 'No Name')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="👥 Total Applicants", value=len(results_df))
        with col2:
            if not results_df.empty:
                avg_score = results_df['relevance_score'].astype(str).str.replace('%','').astype(float).mean()
                st.metric(label="📈 Average Score", value=f"{avg_score:.1f}%")
        with col3:
            if not results_df.empty:
                high_match = len(results_df[results_df['match_level'] == 'Excellent match'])
                st.metric(label="⭐ Excellent Matches", value=high_match)
        
        # Add search for resume name
        search_name = st.text_input("🔍 Search by Resume Name (optional)", "")
        display_df = results_df
        if search_name.strip():
            display_df = results_df[results_df['resume_name'].str.lower().str.contains(search_name.strip().lower())]
        if not display_df.empty:
            # Sort by numeric relevance_score descending
            display_df['relevance_score'] = display_df['relevance_score'].astype(str).str.replace('%','').astype(float)
            display_df = display_df.sort_values(by='relevance_score', ascending=False)
            display_df['relevance_score'] = display_df['relevance_score'].apply(lambda x: f"{x:.1f}%")
            display_df['extracted_skills'] = display_df['extracted_skills'].apply(lambda x: ', '.join(json.loads(x)))
            display_df = display_df[['resume_name', 'relevance_score', 'match_level', 'extracted_skills', 'skill_gaps']]
            display_df.index = np.arange(1, len(display_df) + 1)  # Start index from 1
            st.dataframe(display_df, use_container_width=True)
            # Download top applicant only (ensure not repeated/placeholder)
            top_applicant = display_df.iloc[0]
            top_applicant_df = pd.DataFrame([top_applicant])
            csv = top_applicant_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Top Applicant's Result (CSV)",
                data=csv,
                file_name=f"top_applicant_{top_applicant['resume_name'].replace(' ', '_')}.csv",
                mime='text/csv',
                help="Download the result for the top applicant only."
            )
        else:
            st.markdown("""
            <div style='background-color: #e0f2fe; padding: 2rem; border-radius: 8px; text-align: center; border-left: 4px solid #0288d1;'>
                <h4 style='color: #01579b; margin-bottom: 0.5rem;'>🔍 No Applicants Found</h4>
                <p style='color: #0277bd; margin: 0;'>Please load a Job Description and run analysis to see results.</p>
            </div>
            """, unsafe_allow_html=True)

    # Top Candidates Tab: Show only top 3 by relevance score, filter out placeholder/empty names
    with tab2:
        st.markdown("### 🎯 Top Candidates for the Role")
        results_df = get_results()
        filtered_df = results_df[~results_df['resume_name'].str.lower().str.contains('professional summary|tarun kumar')].copy()
        # Sort by numeric relevance_score descending
        filtered_df['relevance_score'] = filtered_df['relevance_score'].astype(str).str.replace('%','').astype(float)
        filtered_df = filtered_df.sort_values(by='relevance_score', ascending=False)
        if not filtered_df.empty:
            top_candidates = filtered_df.head(3)
            for index, row in top_candidates.iterrows():
                # Determine card color based on match level
                if row['match_level'] == "Excellent match":
                    card_color = "#10b981"
                    icon = "🏆"
                    text_color = "#ffffff"
                elif row['match_level'] == "Partial match":
                    card_color = "#f59e0b"
                    icon = "💡"
                    text_color = "#ffffff"
                else:
                    card_color = "#64748b"
                    icon = "📊"
                    text_color = "#ffffff"
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {card_color} 0%, {card_color}dd 100%); 
                            padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
                        <div style='font-size: 2rem;'>{icon}</div>
                        <div>
                            <h3 style='color: {text_color}; margin: 0; font-size: 1.5rem;'>{row['resume_name']}</h3>
                            <p style='color: {text_color}cc; margin: 0; font-size: 1.2rem;'>Match Score: {row['relevance_score']:.1f}%</p>
                        </div>
                    </div>
                    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px;'>
                        <p style='color: {text_color}; margin-bottom: 0.5rem;'><strong>Match Level:</strong> {row['match_level']}</p>
                        <p style='color: {text_color}; margin-bottom: 0.5rem;'><strong>Key Skills:</strong> {row['extracted_skills']}</p>
                        <p style='color: {text_color}; margin: 0;'><strong>Skill Gaps:</strong> {row['skill_gaps']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #e0f2fe; padding: 2rem; border-radius: 8px; text-align: center; border-left: 4px solid #0288d1;'>
                <h4 style='color: #01579b; margin-bottom: 0.5rem;'>🎯 No Top Candidates</h4>
                <p style='color: #0277bd; margin: 0;'>No candidates to display. Please run the analysis first.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    create_table()
    main()