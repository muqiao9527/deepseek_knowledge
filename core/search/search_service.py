# services/search_service.py
import logging
from typing import List, Dict, Any, Optional

from core.vectorization.text_embedder import TextEmbedder
from core.vectorization.vector_store import VectorStore
from core.search.vector_search import VectorSearchEngine
from core.search.keyword_search import KeywordSearchEngine
from core.search.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)


class SearchService:
    """搜索服务，提供知识库搜索功能"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化搜索服务

        Args:
            config: 搜索服务配置
        """
        self.config = config or {}

        # 初始化向量嵌入器
        model_name = self.config.get("model_name", "BAAI/bge-large-zh-v1.5")
        device = self.config.get("device", "cpu")
        self.embedder = TextEmbedder(model_name=model_name, device=device)

        # 初始化向量存储
        store_type = self.config.get("store_type", "faiss")
        index_path = self.config.get("index_path", "./data/vector_store")
        self.vector_store = VectorStore(store_type=store_type, index_path=index_path)

        # 初始化搜索引擎
        self.vector_engine = VectorSearchEngine(self.vector_store, self.embedder)
        self.keyword_engine = KeywordSearchEngine(use_jieba=self.config.get("use_jieba", True))

        vector_weight = self.config.get("vector_weight", 0.7)
        self.hybrid_engine = HybridSearchEngine(
            self.vector_engine,
            self.keyword_engine,
            vector_weight=vector_weight
        )

        logger.info(f"搜索服务初始化完成: model={model_name}, store={store_type}")

    def search(self, query: str, top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None,
               search_type: str = "hybrid") -> Dict[str, Any]:
        """
        执行搜索

        Args:
            query: 查询文本
            top_k: 返回的最大结果数
            filters: 筛选条件
            search_type: 搜索类型 ('hybrid', 'vector', 'keyword')

        Returns:
            搜索结果
        """
        try:
            # 检查查询是否为空
            if not query or not query.strip():
                return {
                    "query": query,
                    "results": [],
                    "total": 0,
                    "search_type": search_type,
                    "error": "查询不能为空"
                }

            # 执行搜索
            results = self.hybrid_engine.search(
                query,
                top_k=top_k,
                filters=filters,
                mode=search_type
            )

            # 格式化结果
            formatted_results = []
            for result in results:
                # 提取核心字段
                formatted_result = {
                    "text": result.get("text", ""),
                    "score": result.get("score", 0),
                    "rank": result.get("rank", 0)
                }

                # 添加元数据
                metadata = {}
                for key, value in result.items():
                    if key not in ["text", "score", "rank"]:
                        metadata[key] = value

                formatted_result["metadata"] = metadata
                formatted_results.append(formatted_result)

            return {
                "query": query,
                "results": formatted_results,
                "total": len(formatted_results),
                "search_type": search_type
            }

        except Exception as e:
            logger.error(f"搜索时出错: {str(e)}")
            return {
                "query": query,
                "results": [],
                "total": 0,
                "search_type": search_type,
                "error": f"搜索失败: {str(e)}"
            }

    def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        索引文档

        Args:
            documents: 文档列表，每个文档是一个字典，包含text和metadata

        Returns:
            索引结果
        """
        try:
            # 提取文本和元数据
            texts = []
            metadatas = []

            for doc in documents:
                text = doc.get("text", "")
                if not text:
                    continue

                metadata = {k: v for k, v in doc.items() if k != "text"}

                texts.append(text)
                metadatas.append(metadata)

            # 向量索引
            vector_ids = self.vector_store.add_texts(texts, metadatas, self.embedder)

            # 关键词索引
            self.keyword_engine.add_documents([{"text": t, **m} for t, m in zip(texts, metadatas)])

            # 保存向量存储
            self.vector_store.save()

            return {
                "indexed_count": len(vector_ids),
                "vector_ids": vector_ids,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"索引文档时出错: {str(e)}")
            return {
                "indexed_count": 0,
                "vector_ids": [],
                "status": "error",
                "error": f"索引失败: {str(e)}"
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        获取搜索服务统计信息

        Returns:
            统计信息
        """
        vector_stats = self.vector_store.get_stats()

        return {
            "vector_store": vector_stats,
            "document_count": len(self.keyword_engine.documents),
            "embedding_model": self.config.get("model_name", "BAAI/bge-large-zh-v1.5"),
            "device": self.config.get("device", "cpu"),
            "vector_weight": self.hybrid_engine.vector_weight,
            "keyword_weight": self.hybrid_engine.keyword_weight
        }