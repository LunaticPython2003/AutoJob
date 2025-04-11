from fastapi import FastAPI, Header, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional, Dict
from pathlib import Path
import zipfile
import os
import shutil
import uuid
import csv
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import fitz  # PyMuPDF for PDF parsing
import requests

# Initialize FastAPI app
app = FastAPI(title="AutoJob - Intelligent Resume Ranking System")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create necessary directories
Path("uploads").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("static/css").mkdir(exist_ok=True)
Path("static/js").mkdir(exist_ok=True)

# Email configuration
SMTP_SERVER = "smtp.office365.com"
SMTP_PORT = 587
SENDER_EMAIL = "shashwat_202100185@smit.smu.edu.in"
SENDER_PASSWORD = "Sanjay@115531"  # In production, use environment variables

# LLM API configuration
LLM_ENDPOINT = "http://localhost:11434/api/generate"
LLM_MODEL = "mistral:latest"

# --- Helper Classes ---

class FileHandler:
    def extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract ZIP files containing resumes."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

class JDParser:
    def parse(self, jd_text: str) -> str:
        """Parse job description text."""
        return jd_text.strip()

class ResumeParser:
    def extract_text_from_pdf(self, pdf_path) -> str:
        """Extract text content from PDF files."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def parse(self, resume_text: str) -> str:
        """Parse and clean resume text."""
        return resume_text.strip()
    
    def extract_email(self, text: str) -> str:
        """Extract email address from text."""
        # Simple extraction - in production, use a more robust regex pattern
        lines = text.split('\n')
        for line in lines:
            if '@' in line and '.' in line.split('@')[1]:
                words = line.split()
                for word in words:
                    if '@' in word and '.' in word.split('@')[1]:
                        # Clean up the email by removing surrounding punctuation
                        return word.strip('.,;:()<>[]{}')
        return ""

class RAGPipeline:
    def __init__(self, model_name: str):
        self.model = model_name
        self.endpoint = LLM_ENDPOINT

    def _call_model(self, prompt: str) -> str:
        """Call the LLM API with a prompt."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.endpoint, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"[Error calling model]: {e}"

    def score_candidate(self, resume: str, jd: str, notes: str = None) -> tuple:
        """Score a candidate resume against a job description."""
        prompt = f"""
Given the following job description:

{jd}

And the following candidate resume:

{resume}

{f'Additional requirements: {notes}' if notes else ''}

Evaluate how suitable this candidate is for the job on a scale of 0 to 100.
Also, provide a brief explanation of the reasoning behind the score.

Respond in this format:
Score: <number>
Analysis: <text>
"""
        result = self._call_model(prompt)

        try:
            score_line = next(line for line in result.splitlines() if line.lower().startswith("score"))
            score = float(score_line.split(":")[1].strip())
        except Exception:
            score = 0.0

        try:
            analysis_line = result.split("Analysis:", 1)[1].strip()
        except Exception:
            analysis_line = "No analysis provided."

        return score, analysis_line

# --- Email Functions ---

def send_interview_email(recipient_email, recipient_name, job_role, interview_date, interview_time):
    """Send interview invitation email to a candidate."""
    try:
        # Create message
        message = MIMEMultipart()
        message["From"] = SENDER_EMAIL
        message["To"] = recipient_email
        message["Subject"] = f"Interview Invitation for {job_role} Position"
        
        # Email body
        email_body = f"""
Dear {recipient_name},

We are pleased to invite you for an interview for the {job_role} position.

Interview Details:
- Date: {interview_date}
- Time: {interview_time}

Please confirm your availability by replying to this email.

Best regards,
Recruitment Team
        """
        
        message.attach(MIMEText(email_body, "plain"))
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
            
        return True
    except Exception as e:
        print(f"Failed to send email to {recipient_email}. Error: {e}")
        return False

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main application page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/results/{session_id}", response_class=HTMLResponse)
async def results(request: Request, session_id: str):
    """Serve the results page for a specific session."""
    results_file = Path(f"uploads/{session_id}/results.json")
    if not results_file.exists():
        return templates.TemplateResponse("error.html", {
            "request": request, 
            "error": "Results not found"
        })
    
    results_data = json.loads(results_file.read_text())
    
    # Group candidates by job role
    job_roles = {}
    for candidate in results_data.get("results", []):
        role = candidate.get("job_role", "Unspecified Role")
        if role not in job_roles:
            job_roles[role] = []
        job_roles[role].append(candidate)
    
    # Add job roles data
    results_data["job_roles"] = list(job_roles.keys())
    
    return templates.TemplateResponse("results.html", {
        "request": request,
        "results": results_data
    })

@app.post("/upload-job-data/")
async def upload_job_data(
    request: Request,
    jd_file: Optional[UploadFile] = File(None),
    jd_text: Optional[str] = Form(None),
    cvs: List[UploadFile] = File(...),
    additional_notes: Optional[str] = Form(None),
    threshold: float = Form(80.0),
    content_type: str = Header(default="text/csv", alias="Content-Type", convert_underscores=False),
):
    """Process uploaded job descriptions and resumes."""
    # Create a unique session ID and directories
    session_id = str(uuid.uuid4())
    session_path = Path("uploads") / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    cv_dir = session_path / "cvs"
    cv_dir.mkdir(exist_ok=True)

    # Initialize parsers
    jd_parser = JDParser()
    jd_path = session_path / "job_description.txt"

    # Get content type from request
    content_type = request.headers.get("content-type", "")
    job_description = ""
    job_roles = []

    # Process job description based on content type
    if "application/json" in content_type:
        payload = await request.json()
        if isinstance(payload, list):
            jd_entries = []
            for entry in payload:
                role = entry.get('role', '')
                job_roles.append(role)
                jd_entries.append(f"Role: {role}\nDescription: {entry.get('description', '')}")
            job_description = "\n\n".join(jd_entries)
        else:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON format. Must be a list of role-description objects."})

    elif "multipart/form-data" in content_type:
        if jd_file:
            csv_file = await jd_file.read()
            try:
                lines = csv_file.decode("utf-8", errors="ignore").splitlines()
                reader = csv.DictReader(lines)
                jd_entries = []
                for row in reader:
                    role = row.get('role', '')
                    job_roles.append(role)
                    jd_entries.append(f"Role: {role}\nDescription: {row.get('description', '')}")
                job_description = "\n\n".join(jd_entries)
            except:
                # Assume it's plain text
                job_description = csv_file.decode("utf-8", errors="ignore")
                job_roles.append("Default Role")
        elif jd_text:
            job_description = jd_text
            job_roles.append("Default Role")
        else:
            return JSONResponse(status_code=400, content={"error": "Job description is required"})

    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported job description input. Expecting CSV, JSON or text."})

    # Save job description to file
    jd_path.write_text(job_description, encoding="utf-8")

    # Process resume files
    file_handler = FileHandler()
    for file in cvs:
        filename = file.filename
        file_path = cv_dir / filename
        if filename.endswith(".zip"):
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            file_handler.extract_zip(file_path, cv_dir)
            file_path.unlink()
        elif filename.endswith(".pdf"):
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
        else:
            return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {filename}"})

    # Parse job description
    parsed_jd = jd_parser.parse(job_description)
    
    # Split job descriptions by role
    job_descriptions = {}
    if job_roles:
        current_role = None
        current_desc = []
        
        for line in parsed_jd.split('\n'):
            if line.startswith('Role:'):
                if current_role and current_desc:
                    job_descriptions[current_role] = '\n'.join(current_desc)
                current_role = line.replace('Role:', '').strip()
                current_desc = []
            elif current_role and line.strip():
                current_desc.append(line)
        
        # Add the last role
        if current_role and current_desc:
            job_descriptions[current_role] = '\n'.join(current_desc)
    else:
        job_descriptions["Default Role"] = parsed_jd

    # Process resumes and score candidates
    resume_parser = ResumeParser()
    candidate_reports = []
    rag = RAGPipeline(model_name=LLM_MODEL)

    for pdf_file in cv_dir.glob("*.pdf"):
        resume_text = resume_parser.extract_text_from_pdf(pdf_file)
        parsed_resume = resume_parser.parse(resume_text)
        
        # Evaluate candidate against each job role
        best_score = 0
        best_analysis = ""
        best_role = ""
        
        for role, role_description in job_descriptions.items():
            score, analysis = rag.score_candidate(parsed_resume, role_description, additional_notes)
            
            if score > best_score:
                best_score = score
                best_analysis = analysis
                best_role = role
        
        # Extract candidate email from resume
        email = resume_parser.extract_email(parsed_resume)
        
        # Create candidate report
        candidate_reports.append({
            "candidate_id": pdf_file.stem,
            "score": best_score,
            "analysis": best_analysis,
            "shortlisted": best_score >= threshold,
            "job_role": best_role,
            "email": email,
            "interview_scheduled": False,
            "interview_datetime": None
        })

    # Prepare results data
    result_data = {
        "results": candidate_reports, 
        "count": len(candidate_reports), 
        "session_id": session_id,
        "job_roles": job_roles
    }
    
    # Save results to file
    result_file = session_path / "results.json"
    result_file.write_text(json.dumps(result_data))

    # Return response with redirect
    return {"results": candidate_reports, "count": len(candidate_reports), "session_id": session_id, "redirect": f"/results/{session_id}"}

@app.post("/schedule-interview/{session_id}")
async def schedule_interview(
    session_id: str,
    candidate_id: str = Form(...),
    interview_date: str = Form(...),
    interview_time: str = Form(...),
    job_role: str = Form(...),
    candidate_email: str = Form(...),
    candidate_name: str = Form(...)
):
    """Schedule an interview for a candidate and send email notification."""
    results_file = Path(f"uploads/{session_id}/results.json")
    if not results_file.exists():
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    
    # Load current results
    results_data = json.loads(results_file.read_text())
    
    # Update candidate interview status
    for candidate in results_data.get("results", []):
        if candidate.get("candidate_id") == candidate_id:
            candidate["interview_scheduled"] = True
            candidate["interview_datetime"] = f"{interview_date} {interview_time}"
            
            # Send email notification
            email_sent = send_interview_email(
                candidate_email,
                candidate_name,
                job_role,
                interview_date,
                interview_time
            )
            
            # Save updated results
            results_file.write_text(json.dumps(results_data))
            
            return {
                "success": True, 
                "message": "Interview scheduled successfully", 
                "email_sent": email_sent
            }
    
    return JSONResponse(status_code=404, content={"error": "Candidate not found"})

# --- For development only ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# from dataclasses import Field
# from fastapi import FastAPI, Header, UploadFile, File, Form, Request
# from fastapi.responses import JSONResponse
# from typing import List, Optional
# from pathlib import Path
# import zipfile
# import os
# import shutil
# import uuid
# import csv
# import json
# from parsers.jd_parser import JDParser
# from parsers.resume_parser import ResumeParser
# from rag.rag_pipeline import RAGPipeline
# from utils.file_handler import FileHandler

# app = FastAPI()

# @app.post("/upload-job-data/")
# async def upload_job_data(
#     request: Request,
#     jd_file: Optional[UploadFile] = File(None),
#     jd_text: Optional[str] = Form(None),
#     cvs: List[UploadFile] = File(...),
#     additional_notes: Optional[str] = Form(None),
#     threshold: float = Form(80.0),
#     content_type: str = Header(default="text/csv", alias="Content-Type", convert_underscores=False),
# ):
#     session_id = str(uuid.uuid4())
#     session_path = Path("uploads") / session_id
#     session_path.mkdir(parents=True, exist_ok=True)
#     cv_dir = session_path / "cvs"
#     cv_dir.mkdir(exist_ok=True)

#     jd_parser = JDParser()
#     jd_path = session_path / "job_description.txt"

#     content_type = request.headers.get("content-type", "")
#     job_description = ""

#     if "application/json" in content_type:
#         payload = await request.json()
#         if isinstance(payload, list):
#             jd_entries = [f"Role: {entry.get('role', '')}\nDescription: {entry.get('description', '')}" for entry in payload]
#             job_description = "\n\n".join(jd_entries)
#         else:
#             return JSONResponse(status_code=400, content={"error": "Invalid JSON format. Must be a list of role-description objects."})

#     elif "text/csv" in content_type:
#         if not jd_file:
#             return JSONResponse(status_code=400, content={"error": "CSV file is required for Content-Type: text/csv"})
#         csv_file = await jd_file.read()
#         lines = csv_file.decode("utf-8", errors="ignore").splitlines()
#         reader = csv.DictReader(lines)
#         jd_entries = [f"Role: {row.get('role', '')}\nDescription: {row.get('description', '')}" for row in reader]
#         job_description = "\n\n".join(jd_entries)

#     elif jd_text:
#         job_description = jd_text

#     else:
#         return JSONResponse(status_code=400, content={"error": "Unsupported job description input. Expecting CSV, JSON or text."})

#     jd_path.write_text(job_description, encoding="utf-8")

#     file_handler = FileHandler()
#     for file in cvs:
#         filename = file.filename
#         file_path = cv_dir / filename
#         if filename.endswith(".zip"):
#             with open(file_path, "wb") as buffer:
#                 print("Hello World")
#                 buffer.write(await file.read())
#             file_handler.extract_zip(file_path, cv_dir)
#             file_path.unlink()
#         elif filename.endswith(".pdf"):
#             with open(file_path, "wb") as buffer:
#                 buffer.write(await file.read())
#         else:
#             return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {filename}"})

#     parsed_jd = jd_parser.parse(job_description)

#     resume_parser = ResumeParser()
#     candidate_reports = []
#     rag = RAGPipeline(model_name="mistral:latest")

#     for pdf_file in cv_dir.glob("*.pdf"):
#         resume_text = resume_parser.extract_text_from_pdf(pdf_file)
#         parsed_resume = resume_parser.parse(resume_text)
#         score, analysis = rag.score_candidate(parsed_resume, parsed_jd, additional_notes)
#         candidate_reports.append({
#             "candidate_id": pdf_file.stem,
#             "score": score,
#             "analysis": analysis,
#             "shortlisted": score >= threshold
#         })

#     return {"results": candidate_reports, "count": len(candidate_reports), "session_id": session_id}



