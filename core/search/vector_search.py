# core/search/vector_search.py
import logging
from typing import List, Dict, Any, Optional
import time

from sympy.core.sympify import _convert_numpy_types

from ..vectorization.text_embedder import TextEmbedder
from ..vectorization.vector_store import VectorStore

logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """基于向量的语义搜索引擎"""

    def __init__(self, vector_store: VectorStore, embedder: TextEmbedder):
        """
        初始化向量搜索引擎

        Args:
            vector_store: 向量存储实例
            embedder: 文本嵌入器实例
        """
        self.vector_store = vector_store
        self.embedder = embedder
        logger.info("初始化向量搜索引擎")

    # 修改VectorSearchEngine类的search方法
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行语义搜索

        Args:
            query: 查询文本
            top_k: 返回的最大结果数
            filters: 筛选条件

        Returns:
            搜索结果列表
        """
        start_time = time.time()

        try:
            # 生成查询向量
            query_vector = self.embedder.embed_text(query)

            # 在向量存储中搜索
            results = self.vector_store.search(query_vector, k=top_k)

            # 应用筛选条件
            if filters:
                filtered_results = []
                for result in results:
                    match = True
                    for key, value in filters.items():
                        if key in result and result[key] != value:
                            match = False
                            break
                    if match:
                        filtered_results.append(result)
                results = filtered_results

            # 转换numpy类型为Python原生类型
            # results = [_convert_numpy_types(result) for result in results]

            # 记录搜索时间
            search_time = time.time() - start_time
            logger.info(f"向量搜索完成，查询: '{query}'，找到 {len(results)} 个结果，用时 {search_time:.3f}秒")

            return results

        except Exception as e:
            logger.error(f"向量搜索时出错: {str(e)}")
            raise


