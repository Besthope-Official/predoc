import json
import requests
import pathlib


BACKEND_URL = "http://127.0.0.1:8080/api/documents/upload"
PDF_DIR = pathlib.Path("AI预实验文献/英文")

metadata_file = PDF_DIR / "metadata.json"
with open(metadata_file, "r", encoding="utf-8") as f:
    data = json.load(f)
metadata_mapping = {entry["document"]["title"]
    : entry for entry in data["metadata"]}

valid_titles = set(metadata_mapping.keys())

for pdf_file in PDF_DIR.glob("*.pdf"):
    if pdf_file.stem in valid_titles:
        specific_payload = {
            "type": "journal",
            "metadata": json.dumps(metadata_mapping[pdf_file.stem])
        }
        with open(pdf_file, "rb") as file_obj:
            response = requests.post(
                BACKEND_URL, data=specific_payload, files={"file": file_obj})
            print(f"Uploaded {pdf_file.name}: {response.status_code}")
