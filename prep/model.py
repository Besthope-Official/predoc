"""模型初始化和嵌入生成模块。

支持固定预训练模型的初始化和嵌入生成，优化内存使用并与 CONFIG 保持一致。
"""

import os
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import Dict, Tuple
from functools import lru_cache
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from config.model import CONFIG
from loguru import logger


def _model_or_path(model_name: str, local_dir: str, hf_repo_id: str) -> str:
    '''
       Returns model repo on HuggingFace or local path.

       Util for model load. If model exists (and accessible) on local path, use it, else download from HuggingFace.
    '''
    local_model_path = os.path.join(local_dir, model_name)
    return local_model_path \
        if os.path.exists(local_model_path) \
        and os.access(local_model_path, os.R_OK) \
        else hf_repo_id


@lru_cache(maxsize=8)
def _load_tokenizer_and_model(model_name: str, local_dir: str, hf_repo_id: str, device: str = CONFIG.DEVICE) -> Tuple[AutoTokenizer, AutoModel]:
    '''
        Load AutoTokenizer and AutoModel.
    '''
    model_name_or_path = _model_or_path(model_name, local_dir, hf_repo_id)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)

    return tokenizer, model


@lru_cache(maxsize=8)
def _load_sentence_transformer(model_name: str, local_dir: str, hf_repo_id: str, device: str = CONFIG.DEVICE) -> SentenceTransformer:
    '''
        Load SentenceTransformer model.
    '''
    model_name_or_path = _model_or_path(model_name, local_dir, hf_repo_id)
    model = SentenceTransformer(
        model_name_or_path, device=device)

    return model


@lru_cache(maxsize=8)
def _load_yolo_model(model_name: str, local_dir: str, hf_repo_id: str, device: str = CONFIG.DEVICE):
    model_name_or_path = _model_or_path(model_name, local_dir, hf_repo_id)
    if model_name_or_path == hf_repo_id:
        filepath = hf_hub_download(
            repo_id=CONFIG.YOLO_HF_REPO_ID,
            filename=CONFIG.YOLO_MODEL_FILENAME
        )
        logger.info(f"从 Hugging Face 下载 YOLOv10 模型: {filepath}")
        model_name_or_path = filepath
    model = YOLOv10(model_name_or_path)

    return model


def init_model(model_type='st'):
    '''Initialize the model based on the type specified in the config.'''
    try:
        if model_type == "hf":
            # This type of embedding model is not used in current project.
            auto_tokenizer, auto_model = _load_tokenizer_and_model(
                CONFIG.EMBEDDING_MODEL_NAME, CONFIG.EMBEDDING_MODEL_DIR, CONFIG.EMBEDDING_HF_REPO_ID
            )
            model = (auto_tokenizer, auto_model)
        elif model_type == "st":
            model = _load_sentence_transformer(
                CONFIG.EMBEDDING_MODEL_NAME, CONFIG.EMBEDDING_MODEL_DIR, CONFIG.EMBEDDING_HF_REPO_ID
            )
        elif model_type == "yolo":
            model = _load_yolo_model(
                CONFIG.YOLO_MODEL_FILENAME, CONFIG.YOLO_MODEL_DIR, CONFIG.YOLO_HF_REPO_ID
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    except Exception as e:
        raise ValueError(f"Model loading failed: {str(e)}")

    return model
