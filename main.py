from dataclasses import Field
from fastapi import FastAPI, Header, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path
import zipfile
import os
import shutil
import uuid
import csv
import json
from parsers.jd_parser import JDParser
from parsers.resume_parser import ResumeParser
from rag.rag_pipeline import RAGPipeline
from utils.file_handler import FileHandler

app = FastAPI()

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
    session_id = str(uuid.uuid4())
    session_path = Path("uploads") / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    cv_dir = session_path / "cvs"
    cv_dir.mkdir(exist_ok=True)

    jd_parser = JDParser()
    jd_path = session_path / "job_description.txt"

    content_type = request.headers.get("content-type", "")
    job_description = ""

    if "application/json" in content_type:
        payload = await request.json()
        if isinstance(payload, list):
            jd_entries = [f"Role: {entry.get('role', '')}\nDescription: {entry.get('description', '')}" for entry in payload]
            job_description = "\n\n".join(jd_entries)
        else:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON format. Must be a list of role-description objects."})

    elif "text/csv" in content_type:
        if not jd_file:
            return JSONResponse(status_code=400, content={"error": "CSV file is required for Content-Type: text/csv"})
        csv_file = await jd_file.read()
        lines = csv_file.decode("utf-8", errors="ignore").splitlines()
        reader = csv.DictReader(lines)
        jd_entries = [f"Role: {row.get('role', '')}\nDescription: {row.get('description', '')}" for row in reader]
        job_description = "\n\n".join(jd_entries)

    elif jd_text:
        job_description = jd_text

    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported job description input. Expecting CSV, JSON or text."})

    jd_path.write_text(job_description, encoding="utf-8")

    file_handler = FileHandler()
    for file in cvs:
        filename = file.filename
        file_path = cv_dir / filename
        if filename.endswith(".zip"):
            with open(file_path, "wb") as buffer:
                print("Hello World")
                buffer.write(await file.read())
            file_handler.extract_zip(file_path, cv_dir)
            file_path.unlink()
        elif filename.endswith(".pdf"):
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
        else:
            return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {filename}"})

    parsed_jd = jd_parser.parse(job_description)

    resume_parser = ResumeParser()
    candidate_reports = []
    rag = RAGPipeline(model_name="mistral:latest")

    for pdf_file in cv_dir.glob("*.pdf"):
        resume_text = resume_parser.extract_text_from_pdf(pdf_file)
        parsed_resume = resume_parser.parse(resume_text)
        score, analysis = rag.score_candidate(parsed_resume, parsed_jd, additional_notes)
        candidate_reports.append({
            "candidate_id": pdf_file.stem,
            "score": score,
            "analysis": analysis,
            "shortlisted": score >= threshold
        })

    return {"results": candidate_reports, "count": len(candidate_reports), "session_id": session_id}
