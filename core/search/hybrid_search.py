# core/search/hybrid_search.py
import logging
from typing import List, Dict, Any, Optional
import time
from collections import defaultdict

from sympy.core.sympify import _convert_numpy_types

from .vector_search import VectorSearchEngine
from .keyword_search import KeywordSearchEngine

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """混合搜索引擎，结合向量搜索和关键词搜索"""

    def __init__(self, vector_engine: VectorSearchEngine, keyword_engine: KeywordSearchEngine,
                 vector_weight: float = 0.7):
        """
        初始化混合搜索引擎

        Args:
            vector_engine: 向量搜索引擎
            keyword_engine: 关键词搜索引擎
            vector_weight: 向量搜索结果的权重（0-1之间）
        """
        self.vector_engine = vector_engine
        self.keyword_engine = keyword_engine
        self.vector_weight = max(0.0, min(1.0, vector_weight))
        self.keyword_weight = 1.0 - self.vector_weight

        logger.info(
            f"初始化混合搜索引擎: vector_weight={self.vector_weight:.2f}, keyword_weight={self.keyword_weight:.2f}")

    def search(self, query: str, top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None,
               mode: str = "hybrid") -> List[Dict[str, Any]]:
        """
        执行混合搜索

        Args:
            query: 查询文本
            top_k: 返回的最大结果数
            filters: 筛选条件
            mode: 搜索模式 ('hybrid', 'vector', 'keyword')

        Returns:
            搜索结果列表
        """
        start_time = time.time()

        try:
            results = []

            # 根据模式选择搜索方法
            if mode == "vector":
                # 仅使用向量搜索
                results = self.vector_engine.search(query, top_k=top_k, filters=filters)

            elif mode == "keyword":
                # 仅使用关键词搜索
                results = self.keyword_engine.search(query, top_k=top_k)

                # 应用筛选条件
                if filters:
                    results = [r for r in results if self._apply_filters(r, filters)]

            else:  # hybrid
                # 执行向量搜索
                vector_results = self.vector_engine.search(query, top_k=top_k * 2, filters=filters)

                # 执行关键词搜索
                keyword_results = self.keyword_engine.search(query, top_k=top_k * 2)

                # 合并结果
                merged_results = self._merge_results(vector_results, keyword_results, top_k)

                # 应用筛选条件
                if filters:
                    merged_results = [r for r in merged_results if self._apply_filters(r, filters)]

                results = merged_results[:top_k]

            search_time = time.time() - start_time
            logger.info(
                f"混合搜索完成，模式: {mode}, 查询: '{query}'，找到 {len(results)} 个结果，用时 {search_time:.3f}秒")

            return results

        except Exception as e:
            logger.error(f"混合搜索时出错: {str(e)}")
            raise

    def _apply_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        应用筛选条件

        Args:
            result: 搜索结果
            filters: 筛选条件

        Returns:
            是否通过筛选
        """
        for key, value in filters.items():
            if key in result and result[key] != value:
                return False
        return True

    # core/search/hybrid_search.py
    # 在文件末尾添加一个新的辅助函数

    def _convert_numpy_types(obj):
        """
        递归转换所有numpy类型为Python原生类型

        Args:
            obj: 任意对象

        Returns:
            转换后的对象
        """
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_convert_numpy_types(item) for item in obj)
        else:
            return obj

    # 然后修改HybridSearchEngine类的_merge_results方法
    def _merge_results(self, vector_results: List[Dict[str, Any]],
                       keyword_results: List[Dict[str, Any]],
                       top_k: int) -> List[Dict[str, Any]]:
        """
        合并向量搜索和关键词搜索的结果

        Args:
            vector_results: 向量搜索结果
            keyword_results: 关键词搜索结果
            top_k: 返回的最大结果数

        Returns:
            合并后的搜索结果
        """
        # 使用ID跟踪结果
        merged_map = {}

        # 处理向量搜索结果
        for i, result in enumerate(vector_results):
            result_id = result.get("vector_id", result.get("id", f"v{i}"))
            merged_map[result_id] = {
                "result": result,
                "vector_score": result.get("score", 0) * self.vector_weight,
                "keyword_score": 0,
                "final_score": result.get("score", 0) * self.vector_weight
            }

        # 处理关键词搜索结果
        for i, result in enumerate(keyword_results):
            result_id = result.get("id", f"k{i}")

            if result_id in merged_map:
                # 如果结果已存在，更新分数
                merged_map[result_id]["keyword_score"] = result.get("score", 0) * self.keyword_weight
                merged_map[result_id]["final_score"] += result.get("score", 0) * self.keyword_weight
            else:
                # 添加新结果
                merged_map[result_id] = {
                    "result": result,
                    "vector_score": 0,
                    "keyword_score": result.get("score", 0) * self.keyword_weight,
                    "final_score": result.get("score", 0) * self.keyword_weight
                }

        # 排序并取前top_k个结果
        sorted_results = sorted(merged_map.values(), key=lambda x: x["final_score"], reverse=True)[:top_k]

        # 格式化结果
        results = []
        for i, item in enumerate(sorted_results):
            result = item["result"].copy()
            result.update({
                "score": float(item["final_score"]),  # 确保是标准Python浮点数
                "vector_score": float(item["vector_score"] / self.vector_weight if self.vector_weight > 0 else 0),
                # 确保是标准Python浮点数
                "keyword_score": float(item["keyword_score"] / self.keyword_weight if self.keyword_weight > 0 else 0),
                # 确保是标准Python浮点数
                "rank": i + 1
            })
            # 转换所有numpy类型
            # result = _convert_numpy_types(result)
            results.append(result)

        return results