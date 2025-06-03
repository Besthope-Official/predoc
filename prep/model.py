"""模型初始化和嵌入生成模块。

支持固定预训练模型的初始化和嵌入生成，优化内存使用并与 CONFIG 保持一致。
"""

import os
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import Dict, Tuple
from functools import lru_cache
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from config.model import CONFIG
from loguru import logger
import pathlib


def _model_or_path(model_name: str, local_dir: str, hf_repo_id: str) -> str:
    '''
    Returns local path for model. If model doesn't exist locally, downloads it from HuggingFace
    and saves to specified local directory.
    '''
    local_model_path = os.path.join(local_dir, model_name)

    if os.path.exists(local_model_path) and os.access(local_model_path, os.R_OK):
        logger.info(f"Using local model from: {local_model_path}")
        return local_model_path

    os.makedirs(local_dir, exist_ok=True)

    logger.info(f"Downloading model from {hf_repo_id} to {local_model_path}")
    try:
        # Specific for YOLO model
        if isinstance(model_name, str) and model_name.endswith('.pt'):
            filepath = hf_hub_download(
                repo_id=hf_repo_id,
                filename=model_name,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            return filepath

        return hf_repo_id
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model from {hf_repo_id}: {str(e)}")


@lru_cache(maxsize=8)
def _load_tokenizer_and_model(model_name: str, local_dir: str, hf_repo_id: str, device: str = CONFIG.DEVICE) -> Tuple[AutoTokenizer, AutoModel]:
    '''
    Load AutoTokenizer and AutoModel. Downloads and saves to local_dir if not present.
    '''
    model_name_or_path = _model_or_path(model_name, local_dir, hf_repo_id)
    local_model_path = os.path.join(local_dir, model_name)

    if model_name_or_path == hf_repo_id:
        logger.info(f"Downloading model to {local_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)
        model = AutoModel.from_pretrained(hf_repo_id)

        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)
        logger.info(f"Model saved to {local_model_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModel.from_pretrained(local_model_path)

    return tokenizer, model.to(device)


@lru_cache(maxsize=8)
def _load_sentence_transformer(model_name: str, local_dir: str, hf_repo_id: str, device: str = CONFIG.DEVICE) -> SentenceTransformer:
    '''
    Load SentenceTransformer model. Downloads and saves to local_dir if not present.
    '''
    model_name_or_path = _model_or_path(model_name, local_dir, hf_repo_id)
    local_model_path = os.path.join(local_dir, model_name)

    if model_name_or_path == hf_repo_id:
        logger.info(f"Downloading model to {local_model_path}")
        model = SentenceTransformer(hf_repo_id, device=device)
        model.save(local_model_path)
        logger.info(f"Model saved to {local_model_path}")
    else:
        model = SentenceTransformer(local_model_path, device=device)

    return model


@lru_cache(maxsize=8)
def _load_yolo_model(model_name: str, local_dir: str, hf_repo_id: str, device: str = CONFIG.DEVICE):
    '''
    Load YOLOv10 model. Downloads and saves to local_dir if not present.
    '''
    model_path = _model_or_path(model_name, local_dir, hf_repo_id)
    logger.info(f"Loading YOLO model from: {model_path}")
    model = YOLOv10(model_path)
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
