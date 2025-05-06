import os
import re
import json
from typing import Tuple, Dict, Optional
import numpy as np
import fitz
from loguru import logger
from PIL import Image
import pytesseract
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
import cv2
from pathlib import Path

from config.backend import OSSConfig
from task.oss import upload_file

class Parser:
    def __init__(self):
        self.model = self._load_yolo_model()

    def _load_yolo_model(self):
        filepath = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="./doclayout_yolo_docstructbench_imgsz1024.pt"
        )
        return YOLOv10(filepath)

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'[\x00-\x1F\x7F-\x9F\u200b-\u200d\uFEFF]', '', text)
        return text.strip()

    @staticmethod
    def check_file_access(file_path: str) -> None:
        if not os.path.exists(file_path) or not os.access(file_path, os.R_OK):
            raise FileNotFoundError(f"文件 {file_path} 不存在或无权限")

    @staticmethod
    def ensure_output_dirs(paper_output_dir: Path) -> Tuple[Path, Path, Path, Path, Path]:
        dirs = {
            "text": paper_output_dir / "text_contents",
            "formulas": paper_output_dir / "formulas",
            "figures": paper_output_dir / "figures",
            "tables": paper_output_dir / "tables",
            "temp": paper_output_dir / "temp"
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs["text"], dirs["formulas"], dirs["figures"], dirs["tables"], dirs["temp"]
    
    @staticmethod
    def _upload_to_oss(save_path: str, object_name: str):
        upload_result = upload_file(
            file_path=Path(save_path),
            object_name=object_name,
            bucket_name=OSSConfig.minio_bucket
        )
        logger.debug(f"upload to {upload_result}")
    
    def process_pdf(self, pdf_path: str, output_dir: str, upload_to_oss = False) -> str:
        paper_title = os.path.splitext(os.path.basename(pdf_path))[0]
        paper_output_dir = Path(output_dir) / paper_title
        text_dir, formulas_dir, figures_dir, tables_dir, temp_dir = self.ensure_output_dirs(paper_output_dir)

        doc = fitz.open(pdf_path)
        all_text = []
        content_index = []
        counters = {"formula": 1, "figure": 1, "table": 1}
        found_references = False

        for page_num in range(len(doc)):
            if found_references:
                logger.info(f"跳过页面 {page_num + 1}，因已检测到参考文献")
                break

            page = doc.load_page(page_num)
            zoom = 4
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            temp_img_path = temp_dir / f"temp_page_{page_num}.png"
            img.save(temp_img_path, "PNG", compress_level=0)

            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            det_res = self.model.predict(
                str(temp_img_path), imgsz=1024, conf=0.25, device="cuda:0")
            results = det_res[0]

            text_blocks = []
            sorted_boxes = sorted(
                results.boxes.data.tolist(), key=lambda x: x[1])
            for box in sorted_boxes:
                x1, y1, x2, y2, conf, cls = map(int, box[:6])
                crop_img = cv_img[y1:y2, x1:x2]

                element_type = self._get_element_type(cls)
                if not element_type:
                    continue

                if element_type != "text":
                    save_path, marker = self._save_non_text_element(
                        element_type, counters, crop_img, locals()[
                            f"{element_type}s_dir"]
                    )
                    content_index.append({
                        "type": element_type,
                        "id": counters[element_type],
                        "page": page_num + 1,
                        "bbox": (x1, y1, x2, y2),
                        "image_path": save_path,
                        "context_marker": marker
                    })
                    counters[element_type] += 1
                    text_blocks.append(marker)
                    if upload_to_oss:
                        self._upload_to_oss(
                            save_path=save_path,
                            object_name=f"{paper_title}/{element_type}/{Path(save_path).name}")
                else:
                    text = self._process_text_block(crop_img)
                    if re.fullmatch(
                        r'^\s*(参考文献|参考书目|引用文献|References?|Bibliography)[\s\.:：]*$',
                        text,
                        flags=re.IGNORECASE
                    ):
                        found_references = True
                    elif len(text) < 20 and re.search(
                        r'\b(refs?|biblio)\b',
                        text,
                        flags=re.IGNORECASE
                    ):
                        found_references = True
                    text_blocks.append(text.strip())

            if text_blocks and not found_references:
                page_text = "\n\n".join(text_blocks)
                all_text.append(f"[PAGE][{page_num + 1}][PAGE]\n{page_text}")

            os.remove(temp_img_path)

        index_path = paper_output_dir / "content_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(content_index, f, ensure_ascii=False, indent=2)

        return "\n\n".join(all_text)

    def _get_element_type(self, cls: int) -> Optional[str]:
        if cls == 2:
            return None
        elif cls == 5:
            return "table"
        elif cls == 8:
            return "formula"
        elif cls == 3:
            return "figure"
        elif cls in (0, 1):
            return "text"
        return None

    def _save_non_text_element(self, element_type: str, counters: Dict, crop_img: np.ndarray, save_dir: Path) -> Tuple[str, str]:
        filename = f"{element_type}_{counters[element_type]}.png"
        save_path = save_dir / filename
        cv2.imwrite(str(save_path), crop_img)
        logger.info(f"保存 {element_type} 到 {save_path}")
        return str(save_path), f"[/{element_type}][{counters[element_type]}][/{element_type}]"

    def _process_text_block(self, crop_img: np.ndarray) -> str:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray, lang='chi_sim+eng', config='--psm 6 --oem 3')
