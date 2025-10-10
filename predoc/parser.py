import os
import re
import json
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from abc import ABC, abstractmethod
import numpy as np
import fitz
from loguru import logger
from PIL import Image
import pytesseract
import cv2

from config.backend import OSSConfig
from task.oss import upload_file
from .utils import clean_text
from .model import init_model
from config.model import CONFIG

_oss_config = OSSConfig.from_yaml()


class Parser(ABC):
    """文档解析器基类"""

    def __init__(self):
        pass

    @abstractmethod
    def parse(
        self, file_path: str, output_dir: str, upload_to_oss: bool = False
    ) -> str:
        """
        解析文档，提取文本内容

        Args:
            file_path: 文档文件路径
            output_dir: 输出目录
            upload_to_oss: 是否上传到OSS

        Returns:
            提取的文本内容
        """
        raise NotImplementedError("parse method not implemented")

    @staticmethod
    def check_file_access(file_path: str) -> None:
        """检查文件访问权限"""
        if not os.path.exists(file_path) or not os.access(file_path, os.R_OK):
            raise FileNotFoundError(f"文件 {file_path} 不存在或无权限")

    @staticmethod
    def ensure_output_dirs(
        paper_output_dir: Path,
    ) -> Tuple[Path, Path, Path, Path, Path]:
        """确保输出目录存在"""
        dirs = {
            "text": paper_output_dir / "text_contents",
            "formulas": paper_output_dir / "formulas",
            "figures": paper_output_dir / "figures",
            "tables": paper_output_dir / "tables",
            "temp": paper_output_dir / "temp",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return (
            dirs["text"],
            dirs["formulas"],
            dirs["figures"],
            dirs["tables"],
            dirs["temp"],
        )

    @staticmethod
    def _upload_to_oss(save_path: str, object_name: str):
        """上传文件到OSS"""
        upload_result = upload_file(
            file_path=Path(save_path),
            object_name=object_name,
            bucket_name=_oss_config.preprocessed_files_bucket,
        )
        logger.debug(f"upload to {upload_result}")

    def _save_and_upload_file(
        self,
        content,
        save_path: Path,
        paper_title: str = None,
        upload_to_oss: bool = False,
    ):
        """保存文件并可选上传到OSS"""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, np.ndarray):
            cv2.imwrite(str(save_path), content)
        elif isinstance(content, str):
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
        elif isinstance(content, dict) or isinstance(content, list):
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)

        logger.info(f"已保存: {save_path}")

        if upload_to_oss:
            if paper_title:
                object_name = f"{paper_title}/{save_path.name}"
            else:
                object_name = save_path.name

            self._upload_to_oss(str(save_path), object_name)

        return str(save_path)

    def _detect_references(self, text: str) -> bool:
        """检测是否为参考文献部分"""
        if re.fullmatch(
            r"^\s*(参考文献|参考书目|引用文献|References?|Bibliography)[\s\.:：]*$",
            text,
            flags=re.IGNORECASE,
        ):
            return True
        elif len(text) < 20 and re.search(
            r"\b(refs?|biblio)\b", text, flags=re.IGNORECASE
        ):
            return True
        return False


class YoloParser(Parser):
    """基于YOLO模型的PDF解析器"""

    def __init__(self):
        super().__init__()
        self.model = init_model("yolo")

    def _process_page(
        self,
        page,
        page_num: int,
        temp_dir: Path,
        counters: Dict,
        content_index: List,
        paper_title: str,
        upload_to_oss: bool,
        formulas_dir: Path,
        figures_dir: Path,
        tables_dir: Path,
    ) -> Tuple[List[str], bool]:
        """处理单个页面"""
        zoom = 4
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        temp_img_path = temp_dir / f"temp_page_{page_num}.png"
        img.save(temp_img_path, "PNG", compress_level=0)

        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        det_res = self.model.predict(
            str(temp_img_path), imgsz=1024, conf=0.25, device=CONFIG.device
        )
        results = det_res[0]

        text_blocks = []
        found_references = False
        sorted_boxes = sorted(results.boxes.data.tolist(), key=lambda x: x[1])

        for box in sorted_boxes:
            x1, y1, x2, y2, _conf, cls = map(int, box[:6])
            crop_img = cv_img[y1:y2, x1:x2]

            element_type = self._get_element_type(cls)
            if not element_type:
                continue

            if element_type != "text":
                element_dir = {
                    "formula": formulas_dir,
                    "figure": figures_dir,
                    "table": tables_dir,
                }[element_type]

                filename = f"{element_type}_{counters[element_type]}.png"
                save_path = element_dir / filename
                saved_path = self._save_and_upload_file(
                    crop_img,
                    save_path,
                    paper_title=paper_title if upload_to_oss else None,
                    upload_to_oss=upload_to_oss,
                )

                marker = f"[/{element_type}][{counters[element_type]}][/{element_type}]"
                content_index.append(
                    {
                        "type": element_type,
                        "id": counters[element_type],
                        "page": page_num + 1,
                        "bbox": (x1, y1, x2, y2),
                        "image_path": saved_path,
                        "context_marker": marker,
                    }
                )

                counters[element_type] += 1
                text_blocks.append(marker)
            else:
                text = self._process_text_block(crop_img)
                if self._detect_references(text):
                    found_references = True

                text_blocks.append(text.strip())

        os.remove(temp_img_path)
        return text_blocks, found_references

    def parse(
        self, file_path: str, output_dir: str, upload_to_oss: bool = False
    ) -> str:
        """处理PDF文件，提取结构化内容"""
        self.check_file_access(file_path)

        paper_title = os.path.splitext(os.path.basename(file_path))[0]
        paper_output_dir = Path(output_dir) / paper_title
        (
            _text_dir,
            formulas_dir,
            figures_dir,
            tables_dir,
            temp_dir,
        ) = self.ensure_output_dirs(paper_output_dir)

        doc = fitz.open(file_path)
        all_text = []
        content_index = []
        counters = {"formula": 1, "figure": 1, "table": 1}
        found_references = False

        for page_num in range(len(doc)):
            if found_references:
                logger.info(f"跳过页面 {page_num + 1}，因已检测到参考文献")
                break

            page = doc.load_page(page_num)
            text_blocks, page_has_references = self._process_page(
                page,
                page_num,
                temp_dir,
                counters,
                content_index,
                paper_title,
                upload_to_oss,
                formulas_dir,
                figures_dir,
                tables_dir,
            )

            found_references = found_references or page_has_references

            if text_blocks and not found_references:
                page_text = "\n\n".join(text_blocks)
                all_text.append(f"[PAGE][{page_num + 1}][PAGE]\n{page_text}")

        self._save_and_upload_file(
            content_index,
            paper_output_dir / Path("content_index.json"),
            paper_title=paper_title,
            upload_to_oss=upload_to_oss,
        )

        parsed_text = "\n\n".join(all_text)
        self._save_and_upload_file(
            parsed_text,
            paper_output_dir / Path("text.txt"),
            paper_title=paper_title,
            upload_to_oss=upload_to_oss,
        )

        return clean_text(parsed_text)

    def _get_element_type(self, cls: int) -> Optional[str]:
        """根据类别ID获取元素类型"""
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

    def _process_text_block(self, crop_img: np.ndarray) -> str:
        """处理文本块，进行OCR识别"""
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(
            gray, lang="chi_sim+eng", config="--psm 6 --oem 3"
        )
