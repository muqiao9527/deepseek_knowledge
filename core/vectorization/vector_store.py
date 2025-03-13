# core/vectorization/vector_store.py
import os
import logging
import time
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from util.numpy_utils import convert_numpy_types

logger = logging.getLogger(__name__)


class VectorStore:
    """向量存储，负责管理和检索文本嵌入向量"""

    def __init__(self, store_type: str = "faiss", index_path: str = "./data/vector_store"):
        """
        初始化向量存储

        Args:
            store_type: 存储类型 ("faiss" 或 "simple")
            index_path: 索引文件路径
        """
        self.store_type = store_type.lower()
        self.index_path = index_path
        self.index = None
        self.texts = []  # 存储原始文本或ID
        self.metadatas = []  # 存储文档元数据
        self.embeddings = []  # 仅用于简单存储方式

        # 确保索引目录存在
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        logger.info(f"初始化向量存储: type={store_type}, path={index_path}")

        # 尝试加载现有索引
        self._load_or_create_index()

    def _load_or_create_index(self):
        """加载现有索引或创建新索引"""
        try:
            if self.store_type == "faiss":
                self._init_faiss()
            else:
                self._init_simple_store()

            # 尝试加载现有数据
            self._load_data()

        except Exception as e:
            logger.error(f"初始化向量存储时出错: {str(e)}")
            # 失败时创建空索引
            self._create_empty_index()

    def _init_faiss(self):
        """初始化FAISS索引"""
        try:
            import faiss
            self.faiss = faiss

            # 检查是否存在索引文件
            index_file = os.path.join(self.index_path, "faiss.index")
            if os.path.exists(index_file):
                logger.info(f"加载现有FAISS索引: {index_file}")
                self.index = faiss.read_index(index_file)
            else:
                logger.info("创建新的FAISS索引")
                self._create_empty_index()

        except ImportError:
            logger.error("未安装FAISS库，回退到简单存储")
            self.store_type = "simple"
            self._init_simple_store()

    def _init_simple_store(self):
        """初始化简单向量存储"""
        store_file = os.path.join(self.index_path, "vectors.pkl")
        if os.path.exists(store_file):
            logger.info(f"加载简单向量存储: {store_file}")
            try:
                with open(store_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data.get('embeddings', [])
                    self.metadatas = data.get('metadatas', [])
                    self.texts = data.get('texts', [])
            except Exception as e:
                logger.error(f"加载简单向量存储时出错: {str(e)}")
                self._create_empty_index()
        else:
            logger.info("创建新的简单向量存储")
            self._create_empty_index()

    def _create_empty_index(self):
        """创建空索引"""
        if self.store_type == "faiss":
            try:
                # 创建空的FAISS索引
                # 注意：此处需要知道向量维度
                # 默认使用1536维，可以在添加第一个向量时重新初始化
                dimension = 1536
                self.index = self.faiss.IndexFlatL2(dimension)
                logger.info(f"创建了空的FAISS索引，维度: {dimension}")
            except Exception as e:
                logger.error(f"创建FAISS索引时出错: {str(e)}")
                self.store_type = "simple"
                self.embeddings = []
        else:
            # 简单存储
            self.embeddings = []
            self.metadatas = []
            self.texts = []
            logger.info("创建了空的简单向量存储")

    def _load_data(self):
        """加载辅助数据"""
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.metadatas = data.get('metadatas', [])
                    self.texts = data.get('texts', [])
                logger.info(f"加载了 {len(self.metadatas)} 条元数据")
            except Exception as e:
                logger.error(f"加载元数据时出错: {str(e)}")
                self.metadatas = []
                self.texts = []
        else:
            logger.info("未找到元数据文件")
            self.metadatas = []
            self.texts = []

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]],
                  embedder) -> List[str]:
        """
        添加文本及其元数据到向量存储

        Args:
            texts: 文本列表
            metadatas: 元数据列表
            embedder: 文本嵌入器

        Returns:
            添加文本的ID列表
        """
        if not texts:
            logger.warning("尝试添加空文本列表")
            return []

        try:
            # 生成嵌入向量
            embeddings = embedder.embed_texts(texts)

            # 生成ID
            ids = [f"doc_{len(self.texts) + i}" for i in range(len(texts))]

            # 添加到存储
            self._add_embeddings(embeddings, metadatas, ids, texts)

            logger.info(f"添加了 {len(texts)} 个文本到向量存储")
            return ids

        except Exception as e:
            logger.error(f"添加文本到向量存储时出错: {str(e)}")
            return []

    def _add_embeddings(self, embeddings: List[List[float]],
                        metadatas: List[Dict[str, Any]],
                        ids: List[str],
                        texts: List[str]):
        """
        添加嵌入向量及其元数据

        Args:
            embeddings: 嵌入向量列表
            metadatas: 元数据列表
            ids: ID列表
            texts: 原始文本列表
        """
        if not embeddings:
            return

        if self.store_type == "faiss":
            try:
                # 转换为NumPy数组
                embeddings_array = np.array(embeddings).astype('float32')

                # 检查索引是否需要重新初始化
                if self.index.ntotal == 0:
                    dimension = embeddings_array.shape[1]
                    logger.info(f"重新初始化FAISS索引，维度: {dimension}")
                    self.index = self.faiss.IndexFlatL2(dimension)

                # 添加向量到FAISS
                self.index.add(embeddings_array)

                # 保存元数据和文本
                self.metadatas.extend(metadatas)
                self.texts.extend(texts)

            except Exception as e:
                logger.error(f"添加向量到FAISS时出错: {str(e)}")
                raise
        else:
            # 简单存储
            self.embeddings.extend(embeddings)
            self.metadatas.extend(metadatas)
            self.texts.extend(texts)

    def search(self, query_vector: List[float], k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        搜索最相似的向量

        Args:
            query_vector: 查询向量
            k: 返回的最大结果数
            filters: 筛选条件

        Returns:
            搜索结果列表
        """
        if not query_vector or (self.store_type == "faiss" and self.index.ntotal == 0) or \
                (self.store_type != "faiss" and not self.embeddings):
            logger.warning("向量存储为空或查询向量无效")
            return []

        try:
            if self.store_type == "faiss":
                return self._search_faiss(query_vector, k, filters)
            else:
                return self._search_simple(query_vector, k, filters)

        except Exception as e:
            logger.error(f"搜索向量时出错: {str(e)}")
            return []

    def _search_faiss(self, query_vector: List[float], k: int,
                      filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用FAISS搜索"""
        # 转换查询向量
        query_array = np.array([query_vector]).astype('float32')

        # 获取足够多的结果以应用过滤
        fetch_k = k
        if filters:
            fetch_k = min(k * 10, self.index.ntotal)  # 获取更多结果以便过滤

        # 执行搜索
        distances, indices = self.index.search(query_array, fetch_k)

        # 处理结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS返回-1表示没有足够的结果
                continue

            metadata = self.metadatas[idx] if idx < len(self.metadatas) else {}

            # 应用过滤
            if filters and not self._match_filters(metadata, filters):
                continue

            # 计算相似度分数 (将距离转换为相似度)
            distance = distances[0][i]
            score = 1.0 / (1.0 + float(distance))  # 确保使用Python float

            # 构建结果
            result = {
                "text": self.texts[idx] if idx < len(self.texts) else "",
                "score": score,
                "vector_id": f"doc_{idx}"
            }

            # 添加元数据
            result.update(metadata)

            # 转换所有numpy类型
            result = convert_numpy_types(result)
            results.append(result)

            # 达到所需数量就停止
            if len(results) >= k:
                break

        return results

    def _search_simple(self, query_vector: List[float], k: int,
                       filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用简单向量存储搜索"""
        if not self.embeddings:
            return []

        # 转换为NumPy数组
        query_array = np.array(query_vector)
        embeddings_array = np.array(self.embeddings)

        # 计算距离 (使用欧几里得距离)
        distances = np.linalg.norm(embeddings_array - query_array, axis=1)

        # 构建结果并应用过滤
        results = []
        for idx in np.argsort(distances):
            metadata = self.metadatas[idx] if idx < len(self.metadatas) else {}

            # 应用过滤
            if filters and not self._match_filters(metadata, filters):
                continue

            # 计算相似度分数
            distance = float(distances[idx])  # 确保使用Python float
            score = 1.0 / (1.0 + distance)

            # 构建结果
            result = {
                "text": self.texts[idx] if idx < len(self.texts) else "",
                "score": score,
                "vector_id": f"doc_{idx}"
            }

            # 添加元数据
            result.update(metadata)

            # 转换所有numpy类型
            result = convert_numpy_types(result)
            results.append(result)

            # 达到所需数量就停止
            if len(results) >= k:
                break

        return results

    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """检查元数据是否匹配过滤条件"""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def save(self):
        """保存向量存储到文件"""
        try:
            # 创建目录（如果不存在）
            os.makedirs(self.index_path, exist_ok=True)

            # 保存索引
            if self.store_type == "faiss":
                index_file = os.path.join(self.index_path, "faiss.index")
                self.faiss.write_index(self.index, index_file)
                logger.info(f"FAISS索引已保存到 {index_file}")
            else:
                # 保存简单存储
                store_file = os.path.join(self.index_path, "vectors.pkl")
                with open(store_file, 'wb') as f:
                    pickle.dump({
                        'embeddings': self.embeddings,
                        'metadatas': self.metadatas,
                        'texts': self.texts
                    }, f)
                logger.info(f"简单向量存储已保存到 {store_file}")

            # 保存元数据
            metadata_file = os.path.join(self.index_path, "metadata.pkl")
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'metadatas': self.metadatas,
                    'texts': self.texts
                }, f)
            logger.info(f"元数据已保存到 {metadata_file}")

        except Exception as e:
            logger.error(f"保存向量存储时出错: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        获取向量存储的统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "store_type": self.store_type,
            "index_path": self.index_path,
            "document_count": len(self.texts)
        }

        if self.store_type == "faiss":
            stats.update({
                "vector_count": self.index.ntotal,
                "dimension": self.index.d if hasattr(self.index, 'd') else None
            })
        else:
            stats.update({
                "vector_count": len(self.embeddings),
                "dimension": len(self.embeddings[0]) if self.embeddings else None
            })

        return stats