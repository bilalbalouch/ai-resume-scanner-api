from flask import Flask, request, jsonify, send_from_directory
import os
import re

# -------- Resume Text Extraction --------
import pdfplumber
import pytesseract
from PIL import Image

# -------- AI Models --------
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- FLASK SETUP ----------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD AI MODELS ----------------
print("Loading AI models...")

# NER model for skill extraction
ner_tokenizer = AutoTokenizer.from_pretrained("yashpwr/resume-ner-bert-v2")
ner_model = AutoModelForTokenClassification.from_pretrained("yashpwr/resume-ner-bert-v2")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)

# Skill whitelist
VALID_SKILLS = {
    "java", "python", "sql", "flask", "django","flutter","shopify",
    "react", "angular", "node", "javascript","meta ads","android",
    "html", "css", "git", "aws", "docker","dart","full stack developer",
    "machine learning", "nlp", "rest api","firebase",
    "backend", "frontend", "web development",
    "software engineering", "spring", "hibernate"
}

# SBERT for JD-resume similarity
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Models loaded successfully")

# ---------------- HELPER FUNCTIONS ----------------

def extract_text(pdf_path):
    text = ""
    # 1️⃣ Normal text extraction
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    # 2️⃣ If empty → use OCR
    if len(text.strip()) < 30:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                image = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(image)
                text += ocr_text + "\n"
    return text.lower()

def extract_skills(text):
    found = set()
    for skill in VALID_SKILLS:
        if skill in text:
            found.add(skill.title())
    return list(found)

def extract_email(text):
    """Extract first email found in text"""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else None

def calculate_similarity(resume_text, job_description):
    embeddings = sbert_model.encode([resume_text, job_description])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(max(0, float(score) * 100), 2)

def extract_candidate_name(filename):
    name = os.path.splitext(filename)[0]
    name = name.replace("_", " ").replace("-", " ")
    return name.split("(")[0].strip().title()

# ---------------- API ENDPOINTS ----------------

@app.route("/analyze-resumes", methods=["POST"])
def analyze_resumes():
    """
    Input:
      - job_description (form field)
      - resumes (files, multiple)
    Output:
      - ranked resumes with score, candidate name, skills, email, and URL
    """
    job_description = request.form.get("job_description")
    files = request.files.getlist("resumes")

    if not job_description or not files:
        return jsonify({"error": "Job description or resumes missing"}), 400

    results = []

    for file in files:
        # Save resume
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Extract text, skills, email, similarity
        resume_text = extract_text(file_path)
        skills = extract_skills(resume_text)
        score = calculate_similarity(resume_text, job_description)
        email = extract_email(resume_text)

        results.append({
            "resume_name": file.filename,
            "candidate_name": extract_candidate_name(file.filename),
            "match_score": score,
            "skills": skills,
            "email": email
        })

    # Rank resumes
    ranked_resumes = sorted(results, key=lambda x: x["match_score"], reverse=True)

    # Build final JSON
    final_output = []
    base_url = request.host_url.rstrip("/")

    for index, resume in enumerate(ranked_resumes):
        final_output.append({
            "rank": index + 1,
            "candidate_name": resume["candidate_name"],
            "score": resume["match_score"],
            "resume_file": resume["resume_name"],
            "resume_url": f"{base_url}/resumes/{resume['resume_name']}",
            "skills": resume["skills"],
            "email": resume.get("email")
        })

    return jsonify({
        "job_description": job_description,
        "total_resumes": len(final_output),
        "ranked_resumes": final_output
    })

@app.route("/resumes/<filename>")
def serve_resume(filename):
    """Serve uploaded resumes so they can be opened in external PDF viewer"""
    return send_from_directory(
        UPLOAD_FOLDER,
        filename,
        as_attachment=False
    )

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
