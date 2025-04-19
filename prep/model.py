"""模型初始化和嵌入生成模块。

支持固定预训练模型的初始化和嵌入生成，优化内存使用并与 CONFIG 保持一致。
"""

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
def load_model_from_hf(hf_name: str, device: str) -> tuple:
    """缓存加载 Hugging Face 模型和 tokenizer。"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModel.from_pretrained(hf_name).to(device)
        return tokenizer, model
    except Exception as e:
        logger.error(f"加载 HF 模型 {hf_name} 失败: {e}")
        raise


def init_model() -> Dict:
    """初始化固定模型 paraphrase-multilingual-mpnet-base-v2。

    Returns:
        Dict: 包含模型配置、tokenizer、模型、设备和维度信息的字典。

    Raises:
        ValueError: 如果模型加载失败。
    """
    config = {
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "hf_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "type": "st",
        "normalize": True,
        "metric": "INNER_PRODUCT",
        "dimension": 768
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["hf_name"] != CONFIG.EMBEDDING_MODEL:
        logger.warning(
            f"模型 {config['hf_name']} 与 CONFIG.EMBEDDING_MODEL {CONFIG.EMBEDDING_MODEL} 不一致，使用固定模型")

    try:
        if config["type"] == "hf":
            tokenizer, model = load_model_from_hf(config["hf_name"], device)
        elif config["type"] == "st":
            model = SentenceTransformer(config["hf_name"], device=device.type)
            tokenizer = None
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
