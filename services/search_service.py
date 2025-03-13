# services/search_service.py
import logging
from typing import List, Dict, Any, Optional
import time

from sympy.core.sympify import _convert_numpy_types

from core.vectorization.text_embedder import TextEmbedder
from core.vectorization.vector_store import VectorStore
from core.search.vector_search import VectorSearchEngine
from core.search.keyword_search import KeywordSearchEngine
from core.search.hybrid_search import HybridSearchEngine
from config.settings import VECTOR_STORE_PATH, VECTOR_STORE_TYPE, EMBEDDING_MODEL, DEVICE, EMBEDDING_MODEL_PATH

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
        model_name = self.config.get("model_name", EMBEDDING_MODEL)
        device = self.config.get("device", DEVICE)
        model_path = self.config.get("model_path", EMBEDDING_MODEL_PATH)
        self.embedder = TextEmbedder(model_name=model_name, device=device, model_path=model_path)

        # 初始化向量存储
        store_type = self.config.get("store_type", VECTOR_STORE_TYPE)
        index_path = self.config.get("index_path", VECTOR_STORE_PATH)
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

    # 修改SearchService类的search方法
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
                    "score": float(result.get("score", 0)),  # 确保使用Python float
                    "rank": int(result.get("rank", 0))  # 确保使用Python int
                }

                # 添加元数据
                metadata = {}
                for key, value in result.items():
                    if key not in ["text", "score", "rank"]:
                        metadata[key] = value

                formatted_result["metadata"] = metadata
                # 转换所有numpy类型为Python原生类型
                # formatted_result = _convert_numpy_types(formatted_result)
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
            "embedding_model": self.config.get("model_name", EMBEDDING_MODEL),
            "embedding_model_path": self.config.get("model_path", EMBEDDING_MODEL_PATH),
            "device": self.config.get("device", DEVICE),
            "vector_weight": self.hybrid_engine.vector_weight,
            "keyword_weight": self.hybrid_engine.keyword_weight
        }


    async def find_similar_by_id(self, document_id: str, top_k: int = 5, min_score: float = 0.7) -> Dict[str, Any]:
        """
        查找与指定文档相似的其他文档

        Args:
            document_id: 文档ID
            top_k: 最大返回结果数
            min_score: 最小相似度分数

        Returns:
            搜索结果
        """
        try:
            # 获取文档
            document = self.document_repository.get_document(document_id)
            if not document:
                return {
                    "query": f"similar to document {document_id}",
                    "results": [],
                    "total": 0,
                    "error": f"文档 {document_id} 不存在"
                }

            # 检查文档是否已索引
            vector_id = document.get("vector_id")
            if not vector_id:
                # 如果文档未索引，尝试使用内容进行搜索
                if document.get("text_content"):
                    return self.search(
                        query=document.get("text_content")[:1000],  # 使用前1000个字符作为查询
                        top_k=top_k,
                        search_type="vector"
                    )
                else:
                    return {
                        "query": f"similar to document {document_id}",
                        "results": [],
                        "total": 0,
                        "error": "文档未索引且无文本内容"
                    }

            # 使用向量ID进行相似性搜索
            start_time = time.time()
            similar_docs = self.vector_store.get_nearest_examples(vector_id, k=top_k + 1)  # +1是因为可能包含自身

            # 过滤掉自身并应用最小分数阈值
            results = []
            for i, (doc_id, score) in enumerate(zip(similar_docs["ids"], similar_docs["similarities"])):
                if doc_id != document_id and score >= min_score:
                    # 获取文档详情
                    doc = self.document_repository.get_document(doc_id)
                    if doc:
                        results.append({
                            "text": doc.get("text_content", "")[:500],  # 仅返回前500个字符
                            "score": score,
                            "rank": len(results) + 1,
                            "document_id": doc_id,
                            "file_name": doc.get("file_name"),
                            "metadata": doc.get("metadata", {})
                        })

            processing_time = time.time() - start_time

            return {
                "query": f"similar to document {document_id}",
                "results": results,
                "total": len(results),
                "processing_time": processing_time,
                "search_type": "vector_similarity"
            }

        except Exception as e:
            logger.error(f"查找相似文档失败: {str(e)}")
            return {
                "query": f"similar to document {document_id}",
                "results": [],
                "total": 0,
                "error": f"查找相似文档失败: {str(e)}"
            }

    async def recommend_documents(self, user_id: Optional[str] = None, context: Optional[str] = None,
                                  collection: Optional[str] = None, tags: Optional[List[str]] = None,
                                  top_k: int = 5) -> Dict[str, Any]:
        """
        推荐文档

        Args:
            user_id: 用户ID
            context: 上下文信息
            collection: 文档集合
            tags: 标签限制
            top_k: 最大返回结果数

        Returns:
            推荐结果
        """
        try:
            # 构建过滤条件
            filters = {}
            if collection:
                filters["collection"] = collection

            if tags:
                filters["tags"] = tags

            # 如果有上下文，使用上下文进行搜索
            if context:
                results = self.search(
                    query=context,
                    top_k=top_k,
                    filters=filters,
                    search_type="vector"  # 使用向量搜索可能更适合推荐
                )

                results["is_recommendation"] = True
                if user_id:
                    results["user_id"] = user_id

                return results

            # 如果有用户ID，尝试基于用户历史行为推荐
            # 这里只是一个简单示例，实际系统中应有更复杂的推荐逻辑
            if user_id:
                # 模拟获取用户感兴趣的主题
                user_interests = ["项目报告", "技术文档"]

                # 使用用户兴趣作为查询
                results = self.search(
                    query=" ".join(user_interests),
                    top_k=top_k,
                    filters=filters,
                    search_type="hybrid"
                )

                results["is_recommendation"] = True
                results["user_id"] = user_id
                results["based_on"] = "user_interests"

                return results

            # 如果没有上下文和用户ID，返回最近更新的文档
            # 这需要在实际系统中实现，这里只是示例
            recent_docs = self.document_repository.list_documents(
                filters=filters,
                skip=0,
                limit=top_k,
                sort_by="upload_time",
                sort_order="desc"
            )

            results = []
            for i, doc in enumerate(recent_docs):
                results.append({
                    "text": doc.get("text_content", "")[:500] if doc.get("text_content") else "",
                    "score": 1.0 - (i * 0.1),  # 简单的分数计算
                    "rank": i + 1,
                    "document_id": doc.get("id"),
                    "file_name": doc.get("file_name"),
                    "metadata": doc.get("metadata", {})
                })

            return {
                "query": "recent documents",
                "results": results,
                "total": len(results),
                "is_recommendation": True,
                "based_on": "recency"
            }

        except Exception as e:
            logger.error(f"推荐文档失败: {str(e)}")
            return {
                "query": "recommendation",
                "results": [],
                "total": 0,
                "is_recommendation": True,
                "error": f"推荐文档失败: {str(e)}"
            }

    async def autocomplete(self, prefix: str, limit: int = 10) -> List[str]:
        """
        自动完成建议

        Args:
            prefix: 输入前缀
            limit: 返回结果数量

        Returns:
            建议列表
        """
        try:
            # 实际系统中，应该从搜索历史、关键词索引等获取建议
            # 这里简单地模拟一些建议
            if not prefix:
                return []

            suggestions = [
                f"{prefix} 文档",
                f"{prefix} 教程",
                f"{prefix} 指南",
                f"{prefix} 示例",
                f"{prefix} 最佳实践",
                f"{prefix} 问题解决",
                f"{prefix} 常见问题",
                f"{prefix} 参考",
                f"{prefix} 技术规格",
                f"{prefix} 报告"
            ]

            return suggestions[:limit]

        except Exception as e:
            logger.error(f"生成自动完成建议失败: {str(e)}")
            return []

    async def delete_vector(self, vector_id: str) -> bool:
        """
        从向量存储中删除向量

        Args:
            vector_id: 向量ID

        Returns:
            是否成功删除
        """
        try:
            self.vector_store.delete(vector_id)
            # 保存更改
            self.vector_store.save()
            return True
        except Exception as e:
            logger.error(f"从向量存储中删除向量 {vector_id} 失败: {str(e)}")
            return False

    async def update_search_parameters(self, vector_weight: Optional[float] = None) -> Dict[str, Any]:
        """
        更新搜索参数

        Args:
            vector_weight: 向量搜索权重

        Returns:
            更新结果
        """
        try:
            if vector_weight is not None:
                if not 0 <= vector_weight <= 1:
                    raise ValueError("向量搜索权重必须在0到1之间")

                self.vector_weight = vector_weight
                self.hybrid_engine.vector_weight = vector_weight
                self.hybrid_engine.keyword_weight = 1.0 - vector_weight

                logger.info(f"更新搜索参数: vector_weight={vector_weight}")

            return {
                "success": True,
                "vector_weight": self.vector_weight,
                "keyword_weight": 1.0 - self.vector_weight
            }

        except Exception as e:
            logger.error(f"更新搜索参数失败: {str(e)}")
            return {
                "success": False,
                "error": f"更新搜索参数失败: {str(e)}"
            }