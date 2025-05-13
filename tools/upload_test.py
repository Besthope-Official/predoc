import json
import requests
import pathlib
from loguru import logger

BASE_URL = "http://127.0.0.1:8080"
BACKEND_URL = f"{BASE_URL}/api/documents/upload"
PDF_DIR = pathlib.Path("AI预实验文献")
NUM_TO_PROCESS = 100


def send_tasks():
    metadata_file = PDF_DIR / "metadata.json"
    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata_mapping = {
        entry["document"]["title"]: entry for entry in data["metadata"]
    }

    language_dir_mapping = {
        "chinese": PDF_DIR / "中文",
        "english": PDF_DIR / "英文"
    }

    valid_titles = set(metadata_mapping.keys())
    cnt = 0
    for language, subdir in language_dir_mapping.items():
        for pdf_file in subdir.glob("*.pdf"):
            if pdf_file.stem not in valid_titles:
                logger.warning(
                    f"File {pdf_file.name} not in metadata, skipping...")
                continue
            if cnt >= NUM_TO_PROCESS or pdf_file.stem not in valid_titles:
                continue
            metadata = metadata_mapping[pdf_file.stem]
            if metadata["document"]["language"] != language:
                continue
            payload = {
                "type": "journal",
                "metadata": json.dumps(metadata)
            }
            with open(pdf_file, "rb") as file_obj:
                response = requests.post(
                    BACKEND_URL, data=payload, files={"file": file_obj})
                logger.info(
                    f"Uploaded {pdf_file.name}: {response.status_code}")
                cnt += 1
            if cnt >= NUM_TO_PROCESS:
                break


if __name__ == "__main__":
    send_tasks()
