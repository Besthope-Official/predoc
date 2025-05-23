from typing import Dict, List
from config.model import CONFIG
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from .model import init_model


class EmbeddingModel:
    def __init__(self, model_type='st'):
        self.model_type = model_type
        self.device = CONFIG.DEVICE
        self.normalize = True
        self.tokenizer = None
        model = init_model(model_type)
        if model_type == "hf":
            self.tokenizer, self.model = model
        else:
            self.model = model

    def _hf_generate_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        tokenizer = self.tokenizer
        model = self.model
        model.eval()
        embeddings = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入"):
                batch = texts[i:i + batch_size]
                try:
                    inputs = tokenizer(batch, padding=True, truncation=True,
                                       max_length=CONFIG.MAX_LENGTH, return_tensors="pt").to(self.device)
                    outputs = model(**inputs)
                    batch_emb = outputs.last_hidden_state[:, 0, :].cpu(
                    ).numpy()
                    if self.normalize:
                        batch_emb = batch_emb / \
                            np.linalg.norm(batch_emb, axis=1, keepdims=True)
                    embeddings.append(batch_emb)
                except Exception as e:
                    logger.warning(f"批次 {i} 处理失败: {e}, 跳过文本: {batch[:50]}...")
                finally:
                    del inputs, outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        result = np.vstack(embeddings) if embeddings else np.array([])
        return result

    def generate_embedding(self, text: str) -> np.ndarray:
        return self.generate_embeddings([text])[0]

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts or not any(t.strip() for t in texts):
            logger.warning("文本列表为空或全为空白")
            return np.array([])

        batch_size = min(CONFIG.BATCH_SIZE, 32 if self.device == "cuda" else 8)
        try:
            if self.model_type == "hf":
                result = self._hf_generate_embeddings(texts, batch_size)
            else:
                result = self.model.encode(
                    texts, batch_size=batch_size, show_progress_bar=True,
                    convert_to_numpy=True, normalize_embeddings=self.normalize, device=self.device
                )
            if not isinstance(result, np.ndarray) or result.size == 0:
                logger.warning("嵌入生成结果无效，返回空数组")
                result = np.array([])
            logger.info(
                f"生成嵌入完成，向量数: {result.shape[0]}, 维度: {result.shape[1]}")
            return result
        except Exception as e:
            logger.error(f"嵌入生成失败: {e}", exc_info=True)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU 缓存清理")
