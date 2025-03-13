# core/search/keyword_search.py
import logging
from typing import List, Dict, Any, Optional
import re
import time
import jieba
from collections import Counter

logger = logging.getLogger(__name__)


class KeywordSearchEngine:
    """基于关键词的搜索引擎"""

    def __init__(self, use_jieba: bool = True):
        """
        初始化关键词搜索引擎

        Args:
            use_jieba: 是否使用结巴分词
        """
        self.use_jieba = use_jieba
        self.documents = []  # 文档列表
        self.index = {}  # 倒排索引

        logger.info(f"初始化关键词搜索引擎: use_jieba={use_jieba}")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        添加文档到搜索引擎

        Args:
            documents: 文档列表，每个文档是一个字典，包含text和metadata
        """
        start_idx = len(self.documents)

        for i, doc in enumerate(documents):
            idx = start_idx + i

            # 添加文档到列表
            self.documents.append(doc)

            # 提取文档文本
            text = doc.get("text", "")
            if not text:
                continue

            # 分词
            if self.use_jieba:
                tokens = jieba.lcut(text)
            else:
                # 简单的空格分词
                tokens = re.findall(r'\w+', text)

            # 更新倒排索引
            for token in set(tokens):  # 使用集合去重
                if token not in self.index:
                    self.index[token] = []
                self.index[token].append(idx)

        logger.info(f"添加了 {len(documents)} 个文档到关键词索引，当前共 {len(self.documents)} 个文档")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        执行关键词搜索

        Args:
            query: 查询文本
            top_k: 返回的最大结果数

        Returns:
            搜索结果列表
        """
        start_time = time.time()

        try:
            # 分词
            if self.use_jieba:
                query_tokens = jieba.lcut(query)
            else:
                query_tokens = re.findall(r'\w+', query)

            # 查找匹配的文档
            doc_scores = Counter()

            for token in query_tokens:
                if token in self.index:
                    for doc_idx in self.index[token]:
                        doc_scores[doc_idx] += 1

            # 排序并限制结果数
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            # 构建结果
            results = []
            for doc_idx, score in sorted_docs:
                doc = self.documents[doc_idx].copy()
                doc["score"] = score / len(query_tokens)  # 归一化分数
                doc["rank"] = len(results) + 1
                results.append(doc)

            search_time = time.time() - start_time
            logger.info(f"关键词搜索完成，查询: '{query}'，找到 {len(results)} 个结果，用时 {search_time:.3f}秒")

            return results

        except Exception as e:
            logger.error(f"关键词搜索时出错: {str(e)}")
            raise