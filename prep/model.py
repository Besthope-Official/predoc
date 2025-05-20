"""模型初始化和嵌入生成模块。

支持固定预训练模型的初始化和嵌入生成，优化内存使用并与 CONFIG 保持一致。
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import Dict
from functools import lru_cache

from config.model import CONFIG
from loguru import logger


@lru_cache(maxsize=8)
def load_model_from_local_or_hf(model_name: str, device: str, local_dir: str, hf_repo_id: str) -> tuple:
    """从本地路径或Hugging Face加载模型和tokenizer，支持缓存。"""
    local_model_path = os.path.join(local_dir, model_name)

    try:
        if os.path.exists(local_model_path) and os.access(local_model_path, os.R_OK):
            logger.info(f"从本地路径加载嵌入模型: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModel.from_pretrained(local_model_path).to(device)
            return tokenizer, model
        else:
            logger.warning(
                f"本地模型目录 {local_model_path} 不存在或无读权限，尝试从 Hugging Face 下载")
    except Exception as e:
        logger.error(f"加载本地嵌入模型失败: {e}")
    try:
        logger.info(f"从 Hugging Face 下载嵌入模型: {hf_repo_id}")
        tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)
        model = AutoModel.from_pretrained(hf_repo_id).to(device)
        return tokenizer, model
    except Exception as e:
        logger.error(f"从 Hugging Face 下载模型失败: {e}")
        raise RuntimeError(f"无法加载嵌入模型: 本地和 Hugging Face 均失败")


def init_model() -> Dict:
    """初始化固定模型 paraphrase-multilingual-mpnet-base-v2。

    Returns:
        Dict: 包含模型配置、tokenizer、模型、设备和维度信息的字典。

    Raises:
        ValueError: 如果模型加载失败。
    """
    config = {
        "name": CONFIG.EMBEDDING_MODEL_NAME,
        "hf_name": CONFIG.EMBEDDING_HF_REPO_ID,
        "type": "st",
        "normalize": True,
        "metric": "INNER_PRODUCT",
        "dimension": 768
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if config["type"] == "hf":
            tokenizer, model = load_model_from_local_or_hf(
                config["name"], device, CONFIG.EMBEDDING_MODEL_DIR, CONFIG.EMBEDDING_HF_REPO_ID
            )
        elif config["type"] == "st":
            local_model_path = os.path.join(
                CONFIG.EMBEDDING_MODEL_DIR, CONFIG.EMBEDDING_MODEL_NAME)
            try:
                if os.path.exists(local_model_path) and os.access(local_model_path, os.R_OK):
                    logger.info(
                        f"从本地路径加载 SentenceTransformer 模型: {local_model_path}")
                    model = SentenceTransformer(
                        local_model_path, device=device.type)
                else:
                    logger.warning(
                        f"本地模型目录 {local_model_path} 不存在或无读权限，尝试从 Hugging Face 下载")
                    model = SentenceTransformer(
                        CONFIG.EMBEDDING_HF_REPO_ID, device=device.type)
                tokenizer = None
            except Exception as e:
                logger.error(f"加载 SentenceTransformer 模型失败: {e}")
                raise RuntimeError(
                    f"无法加载 SentenceTransformer 模型: 本地和 Hugging Face 均失败")
        else:
            raise ValueError(f"未知模型类型: {config['type']}")
        logger.info(f"成功加载模型: {config['name']} 到 {device}")
        return {"config": config, "tokenizer": tokenizer, "model": model, "device": device, "dimension": config["dimension"]}
    except Exception as e:
        logger.error(f"模型 {config['name']} 加载失败: {e}")
        raise ValueError("模型加载失败")


def generate_embeddings(model_info: Dict, texts: list) -> np.ndarray:
    """生成嵌入向量，优化内存使用并记录失败文本。

    Args:
        model_info (Dict): 模型信息。
        texts (list): 文本列表。

    Returns:
        np.ndarray: 嵌入向量数组。
    """
    if not texts or not any(t.strip() for t in texts):
        logger.warning("文本列表为空或全为空白")
        return np.array([])

    model, device, config = model_info["model"], model_info["device"], model_info["config"]
    embeddings = []
    batch_size = min(CONFIG.BATCH_SIZE, 32 if device.type == "cuda" else 8)

    try:
        if config["type"] == "hf":
            tokenizer = model_info["tokenizer"]
            model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast():
                for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入"):
                    batch = texts[i:i + batch_size]
                    try:
                        inputs = tokenizer(batch, padding=True, truncation=True,
                                           max_length=CONFIG.MAX_LENGTH, return_tensors="pt").to(device)
                        outputs = model(**inputs)
                        batch_emb = outputs.last_hidden_state[:, 0, :].cpu(
                        ).numpy()
                        if config["normalize"]:
                            batch_emb = batch_emb / \
                                np.linalg.norm(
                                    batch_emb, axis=1, keepdims=True)
                        embeddings.append(batch_emb)
                    except Exception as e:
                        logger.warning(
                            f"批次 {i} 处理失败: {e}, 跳过文本: {batch[:50]}...")
                    finally:
                        del inputs, outputs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            result = np.vstack(embeddings) if embeddings else np.array([])
        else:
            result = model.encode(
                texts, batch_size=batch_size, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=config["normalize"], device=device.type
            )
            if not isinstance(result, np.ndarray) or result.size == 0:
                logger.warning("嵌入生成结果无效，返回空数组")
                result = np.array([])

        logger.info(f"生成嵌入完成，向量数: {result.shape[0]}, 维度: {result.shape[1]}")
        return result
    except Exception as e:
        logger.error(f"嵌入生成失败: {e}", exc_info=True)
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU 缓存清理")
