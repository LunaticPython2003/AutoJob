import requests

class RAGPipeline:
    def __init__(self, model_name: str):
        self.model = model_name
        self.endpoint = "http://localhost:11434/api/generate"

    def _call_model(self, prompt: str) -> str:
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
