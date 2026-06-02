import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


def calculate_match_score(resume_text, job_description):
    documents = [resume_text, job_description]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity[0][0] * 100, 2)


def extract_keywords(text):
    common_words = {
        "the", "and", "for", "with", "you", "your", "are", "this", "that",
        "from", "have", "will", "can", "our", "job", "role", "work",
        "team", "company", "experience", "skills", "ability", "required",
        "responsibilities", "qualifications", "candidate", "position"
    }

    words = clean_text(text).split()
    keywords = []

    for word in words:
        if len(word) > 3 and word not in common_words:
            keywords.append(word)

    return set(keywords)


def keyword_analysis(resume_text, job_description):
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)

    matched = resume_keywords.intersection(job_keywords)
    missing = job_keywords.difference(resume_keywords)

    return sorted(matched), sorted(missing)


st.set_page_config(
    page_title="Resume Job Analyzer",
    page_icon="📄",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1e293b;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #475569;
        margin-bottom: 35px;
    }

    .card {
        background-color: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .score-card {
        background: linear-gradient(135deg, #2563eb, #14b8a6);
        color: white;
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }

    .score-text {
        font-size: 48px;
        font-weight: 800;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 12px;
    }

    .keyword-box {
        background-color: #e0f2fe;
        color: #075985;
        padding: 8px 12px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px;
        font-size: 14px;
    }

    .missing-box {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 8px 12px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 40px;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.title("📌 About Project")
    st.write(
        "This app analyzes a resume against a job description using NLP techniques."
    )

    st.markdown("### Features")
    st.write("✅ Resume PDF upload")
    st.write("✅ Job description analysis")
    st.write("✅ Match score")
    st.write("✅ Matched keywords")
    st.write("✅ Missing keywords")
    st.write("✅ Improvement suggestions")

    st.markdown("### Tech Stack")
    st.write("Python")
    st.write("Streamlit")
    st.write("PyPDF2")
    st.write("Scikit-learn")
    st.write("TF-IDF")
    st.write("Cosine Similarity")


st.markdown("<div class='main-title'>📄 Resume Job Analyzer</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Analyze how well your resume matches a job description using NLP.</div>",
    unsafe_allow_html=True
)

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Upload Resume</div>", unsafe_allow_html=True)
    resume_file = st.file_uploader("Choose your resume PDF", type=["pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Job Description</div>", unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste the job description here",
        height=220
    )
    st.markdown("</div>", unsafe_allow_html=True)


analyze_button = st.button("Analyze Resume", use_container_width=True)

if analyze_button:

    if resume_file is not None and job_description.strip() != "":

        resume_text = extract_text_from_pdf(resume_file)

        if resume_text.strip() == "":
            st.error("Could not extract text from the PDF.")
        else:
            score = calculate_match_score(resume_text, job_description)
            matched_keywords, missing_keywords = keyword_analysis(
                resume_text,
                job_description
            )

            st.markdown("---")

            result_col1, result_col2 = st.columns([1, 2])

            with result_col1:
                st.markdown(
                    f"""
                    <div class='score-card'>
                        <h3>Match Score</h3>
                        <div class='score-text'>{score}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(score / 100)

                if score >= 70:
                    st.success("Excellent match.")
                elif score >= 40:
                    st.warning("Moderate match.")
                else:
                    st.error("Low match.")

            with result_col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Resume Feedback</div>", unsafe_allow_html=True)

                if score >= 70:
                    st.write("Your resume is strongly aligned with this job description.")
                elif score >= 40:
                    st.write("Your resume has some relevant keywords, but it can be improved.")
                else:
                    st.write("Your resume does not strongly match this job description yet.")

                if missing_keywords:
                    st.write("Try adding relevant missing keywords naturally into your resume.")
                else:
                    st.write("Your resume already covers most of the important keywords.")

                st.markdown("</div>", unsafe_allow_html=True)

            keyword_col1, keyword_col2 = st.columns(2)

            with keyword_col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Matched Keywords</div>", unsafe_allow_html=True)

                if matched_keywords:
                    for keyword in matched_keywords[:30]:
                        st.markdown(
                            f"<span class='keyword-box'>{keyword}</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.write("No strong keyword matches found.")

                st.markdown("</div>", unsafe_allow_html=True)

            with keyword_col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Missing Keywords</div>", unsafe_allow_html=True)

                if missing_keywords:
                    for keyword in missing_keywords[:30]:
                        st.markdown(
                            f"<span class='missing-box'>{keyword}</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.write("No major missing keywords found.")

                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Improvement Suggestions</div>", unsafe_allow_html=True)

            if missing_keywords:
                top_missing = ", ".join(missing_keywords[:10])
                st.write(f"Add these relevant terms if they match your real experience: {top_missing}")
                st.write("Use these keywords in your skills, experience, or project sections.")
            else:
                st.write("Your resume already includes most of the important job-related keywords.")

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.error("Please upload a resume and paste a job description.")


st.markdown(
    "<div class='footer'>Built by Abdullah Zafar | Resume Job Analyzer</div>",
    unsafe_allow_html=True
)