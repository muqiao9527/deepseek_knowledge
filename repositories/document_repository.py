# repositories/document_repository.py
import logging
from typing import List, Dict, Any, Optional
import os
import json
from uuid import uuid4

logger = logging.getLogger(__name__)


class DocumentRepository:
    """文档仓库，处理文档的存储和检索"""

    def __init__(self):
        """初始化文档仓库"""
        # 示例实现使用基于文件的存储
        # 实际项目中应该使用数据库
        self.data_dir = "./data/documents"
        os.makedirs(self.data_dir, exist_ok=True)

        # 加载所有文档
        self.documents = self._load_documents()
        logger.info(f"文档仓库初始化完成，加载了 {len(self.documents)} 个文档")

    def _load_documents(self) -> Dict[str, Dict[str, Any]]:
        """加载所有文档"""
        documents = {}

        # 遍历数据目录中的所有json文件
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                try:
                    file_path = os.path.join(self.data_dir, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                        doc_id = doc.get("id")
                        if doc_id:
                            documents[doc_id] = doc
                except Exception as e:
                    logger.error(f"加载文档 {filename} 失败: {str(e)}")

        return documents

    def save_document(self, document_data: Dict[str, Any]) -> str:
        """
        保存文档

        Args:
            document_data: 文档数据

        Returns:
            文档ID
        """
        # 检查是否有ID，如果没有则生成新ID
        doc_id = document_data.get("id")
        if not doc_id:
            doc_id = str(uuid4())
            document_data["id"] = doc_id

        # 保存到内存
        self.documents[doc_id] = document_data

        # 持久化到文件
        try:
            file_path = os.path.join(self.data_dir, f"{doc_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(document_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存文档 {doc_id} 失败: {str(e)}")

        return doc_id

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文档

        Args:
            document_id: 文档ID

        Returns:
            文档数据，如果不存在则返回None
        """
        return self.documents.get(document_id)

    def update_document(self, document_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        更新文档

        Args:
            document_id: 文档ID
            update_data: 要更新的数据

        Returns:
            更新后的文档数据，如果文档不存在则返回None
        """
        # 检查文档是否存在
        if document_id not in self.documents:
            return None

        # 更新文档
        doc = self.documents[document_id]

        # 递归更新字典
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_dict(d[k], v)
                else:
                    d[k] = v

        update_dict(doc, update_data)

        # 持久化到文件
        try:
            file_path = os.path.join(self.data_dir, f"{document_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"更新文档 {document_id} 失败: {str(e)}")

        return doc

    def delete_document(self, document_id: str) -> bool:
        """
        删除文档

        Args:
            document_id: 文档ID

        Returns:
            是否成功删除
        """
        # 检查文档是否存在
        if document_id not in self.documents:
            return False

        # 从内存中删除
        del self.documents[document_id]

        # 从文件系统中删除
        try:
            file_path = os.path.join(self.data_dir, f"{document_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"删除文档 {document_id} 文件失败: {str(e)}")
            # 即使文件删除失败，我们仍然认为文档已删除（因为已从内存中删除）

        return True

    def list_documents(self, filters: Dict[str, Any] = None, skip: int = 0,
                       limit: int = 100, sort_by: str = None,
                       sort_order: str = "asc") -> List[Dict[str, Any]]:
        """
        获取文档列表

        Args:
            filters: 过滤条件
            skip: 跳过的文档数
            limit: 返回的最大文档数
            sort_by: 排序字段
            sort_order: 排序顺序 (asc, desc)

        Returns:
            文档列表
        """
        filters = filters or {}

        # 应用过滤器
        filtered_docs = []
        for doc in self.documents.values():
            include = True

            for key, value in filters.items():
                # 处理特殊的嵌套键
                if key == "tags":
                    # 检查标签是否存在
                    doc_tags = doc.get("metadata", {}).get("tags", [])
                    if not set(value).issubset(set(doc_tags)):
                        include = False
                        break
                elif key == "source":
                    # 检查来源是否匹配
                    doc_source = doc.get("metadata", {}).get("source")
                    if doc_source != value:
                        include = False
                        break
                elif key in doc:
                    # 直接比较文档属性
                    if doc[key] != value:
                        include = False
                        break
                else:
                    # 键不存在，不包括该文档
                    include = False
                    break

            if include:
                filtered_docs.append(doc)

        # 应用排序
        if sort_by:
            reverse = sort_order.lower() == "desc"

            def sort_key(doc):
                # 处理嵌套键
                if sort_by == "title":
                    return doc.get("metadata", {}).get("title", "")
                else:
                    return doc.get(sort_by, "")

            filtered_docs.sort(key=sort_key, reverse=reverse)

        # 应用分页
        return filtered_docs[skip:skip + limit]

    def count_documents(self, filters: Dict[str, Any] = None) -> int:
        """
        统计文档数量

        Args:
            filters: 过滤条件

        Returns:
            文档数量
        """
        if not filters:
            return len(self.documents)

        # 应用过滤器进行计数
        count = 0
        for doc in self.documents.values():
            include = True

            for key, value in filters.items():
                # 处理特殊的嵌套键
                if key == "tags":
                    # 检查标签是否存在
                    doc_tags = doc.get("metadata", {}).get("tags", [])
                    if not set(value).issubset(set(doc_tags)):
                        include = False
                        break
                elif key == "source":
                    # 检查来源是否匹配
                    doc_source = doc.get("metadata", {}).get("source")
                    if doc_source != value:
                        include = False
                        break
                elif key in doc:
                    # 直接比较文档属性
                    if doc[key] != value:
                        include = False
                        break
                else:
                    # 键不存在，不包括该文档
                    include = False
                    break

            if include:
                count += 1

        return count

    def search_by_text(self, text: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        根据文本内容搜索文档（简单实现）

        Args:
            text: 搜索文本
            limit: 返回的最大文档数

        Returns:
            文档列表
        """
        if not text:
            return []

        text = text.lower()
        results = []

        for doc in self.documents.values():
            # 在文档内容中搜索
            content = doc.get("text_content", "").lower()
            if text in content:
                # 匹配成功，添加到结果中
                score = content.count(text)  # 简单的匹配次数作为分数
                results.append((doc, score))

            # 在元数据中搜索
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "").lower()
            description = metadata.get("description", "").lower()

            if text in title:
                # 标题匹配权重更高
                score = title.count(text) * 3
                results.append((doc, score))

            if text in description:
                # 描述匹配
                score = description.count(text) * 2
                results.append((doc, score))

        # 对结果去重并排序
        unique_results = {}
        for doc, score in results:
            doc_id = doc.get("id")
            if doc_id in unique_results:
                unique_results[doc_id] = (doc, unique_results[doc_id][1] + score)
            else:
                unique_results[doc_id] = (doc, score)

        # 按分数排序
        sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)

        # 返回文档列表
        return [doc for doc, _ in sorted_results[:limit]]