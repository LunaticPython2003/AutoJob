import zipfile
from pathlib import Path

class FileHandler:
    def extract_zip(self, zip_path: Path, extract_to: Path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print("WORLD")
            zip_ref.extractall(extract_to)
