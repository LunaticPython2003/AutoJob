import fitz

class ResumeParser:
    def extract_text_from_pdf(self, pdf_path) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def parse(self, resume_text: str) -> str:
        return resume_text.strip()
