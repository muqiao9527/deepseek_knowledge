# core/vectorization/text_embedder.py
import logging
import os
from typing import List

import numpy as np
# 导入 transformers
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class TextEmbedder:
    """文本嵌入器，将文本转换为向量表示"""

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", device: str = "cpu", model_path: str = None):
        """
        初始化文本嵌入器

        Args:
            model_name: 使用的模型名称
            device: 运行设备 ("cpu" 或 "cuda")
            model_path: 本地模型路径，如果设置了这个参数，将优先使用本地模型
        """
        self.model_name = model_name
        self.device = device
        self.model_path = model_path or model_name  # 如果未提供model_path，使用model_name
        self.model = None
        self.tokenizer = None
        self.embedding_size = 0

        logger.info(f"初始化文本嵌入器: model={model_name}, device={device}, model_path={self.model_path}")
        self._load_model()

    def _load_model(self):
        """加载嵌入模型"""
        try:

            # 检查模型路径是否存在
            is_local_path = os.path.exists(self.model_path)
            if is_local_path:
                logger.info(f"使用本地模型: {self.model_path}")
            else:
                logger.info(f"使用在线模型: {self.model_name}")

            # 加载模型和分词器
            model_path_or_name = self.model_path if is_local_path else self.model_name
            logger.info(f"正在加载嵌入模型: {model_path_or_name}")

            # 加载分词器
            tokenizer_path = self.model_path if is_local_path else self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

            # 加载模型
            self.model = AutoModel.from_pretrained(model_path_or_name, trust_remote_code=True)

            # 移动模型到指定设备
            self.model.to(self.device)

            # 设置模型为评估模式
            self.model.eval()

            # 获取嵌入维度
            self.embedding_size = self.model.config.hidden_size
            logger.info(f"嵌入模型加载完成: 向量维度={self.embedding_size}")

        except Exception as e:
            logger.error(f"加载嵌入模型时出错: {str(e)}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        将单个文本转换为嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        if not text:
            logger.warning("嵌入空文本，返回零向量")
            return [0.0] * self.embedding_size

        try:
            # 使用 transformers 进行编码
            import torch

            with torch.no_grad():
                # 编码文本
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 获取模型输出
                outputs = self.model(**inputs)

                # 使用最后一层隐藏状态的 [CLS] 令牌作为嵌入
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                # 归一化向量
                embedding = embeddings[0]
                embedding_norm = np.linalg.norm(embedding)
                if embedding_norm > 0:
                    embedding = embedding / embedding_norm

                return embedding.tolist()

        except Exception as e:
            logger.error(f"生成嵌入向量时出错: {str(e)}")
            # 出错时返回零向量
            return [0.0] * self.embedding_size

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        将多个文本批量转换为嵌入向量

        Args:
            texts: 输入文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            logger.warning("嵌入空文本列表，返回空列表")
            return []

        try:
            # 批量处理
            import torch

            embeddings = []
            batch_size = 32  # 批处理大小

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # 跳过空文本
                valid_indices = [j for j, text in enumerate(batch_texts) if text]
                valid_texts = [batch_texts[j] for j in valid_indices]

                if not valid_texts:
                    # 如果批次中没有有效文本，为每个项生成零向量
                    batch_embeddings = [[0.0] * self.embedding_size for _ in batch_texts]
                else:
                    with torch.no_grad():
                        # 编码有效文本
                        inputs = self.tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True,
                                                max_length=512)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        # 获取模型输出
                        outputs = self.model(**inputs)

                        # 获取嵌入
                        batch_valid_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                        # 归一化
                        for j in range(len(batch_valid_embeddings)):
                            embedding = batch_valid_embeddings[j]
                            embedding_norm = np.linalg.norm(embedding)
                            if embedding_norm > 0:
                                batch_valid_embeddings[j] = embedding / embedding_norm

                        # 构建包含零向量的完整批次
                        batch_embeddings = [[0.0] * self.embedding_size for _ in batch_texts]
                        for idx, valid_idx in enumerate(valid_indices):
                            batch_embeddings[valid_idx] = batch_valid_embeddings[idx].tolist()

                embeddings.extend(batch_embeddings)

            return embeddings

        except Exception as e:
            logger.error(f"批量生成嵌入向量时出错: {str(e)}")
            # 返回零向量列表

    def get_embedding_size(self) -> int:
        """
        获取嵌入向量的维度

        Returns:
            嵌入向量维度
        """
        return self.embedding_size