# services/document_service.py
import hashlib
import os
import shutil
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
from fastapi import UploadFile
from core.document_processors.processor_factory import DocumentProcessorFactory
from core.vectorization import TextChunker
from repositories.document_repository import DocumentRepository
from services.search_service import SearchService

logger = logging.getLogger(__name__)


class DocumentService:
    """文档服务层，处理与文档相关的业务逻辑"""

    def __init__(self):
        """初始化文档服务"""
        self.config = {}
        # 不传递参数给DocumentProcessorFactory
        self.processor_factory = DocumentProcessorFactory()

        # 初始化文本分块器
        self.chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=50,
            chunking_method="text"
        )

        self.document_repository = DocumentRepository()

        # 设置上传目录
        self.upload_dir = os.path.join("./data", "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)

        logger.info("文档服务初始化完成")


    async def upload_document(self, file: UploadFile, document_request: Any, index_immediately: bool = False) -> Dict[
        str, Any]:
        """
        处理文档上传

        Args:
            file: 上传的文件
            document_request: 包含文件信息和元数据的请求
            index_immediately: 是否立即索引

        Returns:
            文档响应信息
        """
        # 从请求中提取文件信息和元数据
        file_info = document_request.file_info
        metadata = document_request.metadata

        # 生成唯一文件名以避免冲突
        unique_filename = f"{uuid4().hex}_{file.filename}"
        file_path = os.path.join(self.upload_dir, unique_filename)

        # 保存上传的文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"文件上传成功: {file.filename} -> {file_path}")

        # 处理文档
        result = self.processor_factory.process_document(file_path)

        # 准备文档数据
        document_data = {
            "file_name": file.filename,
            "file_path": file_path,
            "file_size": file_info.file_size,
            "file_type": os.path.splitext(file.filename)[1].lower(),
            "content_type": file_info.content_type,
            "upload_time": datetime.now().isoformat(),
            "text_content": result.get("text", ""),
            "source_path": file_info.filePath,
            "metadata": {
                **metadata.dict(),
                **result.get("metadata", {})
            }
        }

        # 保存文档到数据库
        document_id = self.document_repository.save_document(document_data)

        # 如果选择立即索引，则创建搜索索引
        if index_immediately and result.get("text"):
            index_result = await self.search_service.index_documents([{
                "id": document_id,
                "text": result.get("text", ""),
                "file_name": file.filename,
                "title": metadata.title,
                "tags": metadata.tags,
                "source": metadata.source,
                **result.get("metadata", {})
            }])

            document_data["indexed"] = True
            document_data["vector_id"] = index_result.get("vector_ids", [])[0] if index_result.get(
                "vector_ids") else None

            # 更新文档索引状态
            self.document_repository.update_document(
                document_id=document_id,
                update_data={
                    "indexed": True,
                    "vector_id": document_data["vector_id"]
                }
            )
        else:
            document_data["indexed"] = False

        # 构建响应
        return {
            "document_id": document_id,
            "file_name": file.filename,
            "file_info": {
                "file_size": file_info.file_size,
                "content_type": file_info.content_type,
                "file_path": file_info.filePath
            },
            "metadata": metadata.dict(),
            "indexed": document_data["indexed"],
            "upload_time": document_data["upload_time"],
            "status": "success",
            "message": "文档上传并处理成功"
        }

    async def create_document(self, document_request: Any, file_content: Optional[str] = None) -> Dict[str, Any]:
        """
        创建文档记录（无需上传文件）

        Args:
            document_request: 包含文件信息和元数据的请求
            file_content: 可选的文本内容

        Returns:
            文档响应信息
        """
        file_info = document_request.file_info
        metadata = document_request.metadata

        # 判断是使用文件路径还是直接内容
        if os.path.exists(file_info.filePath):
            # 使用现有文件
            result = self.processor_factory.process_document(file_info.filePath)
            text_content = result.get("text", "")
            extracted_metadata = result.get("metadata", {})
        elif file_content:
            # 使用提供的文本内容
            text_content = file_content
            extracted_metadata = {}
        else:
            raise ValueError("需要提供有效的文件路径或文本内容")

        # 准备文档数据
        document_data = {
            "file_name": file_info.filename,
            "file_path": file_info.filePath,
            "file_size": file_info.file_size,
            "file_type": os.path.splitext(file_info.filename)[1].lower(),
            "content_type": file_info.content_type,
            "upload_time": datetime.now().isoformat(),
            "text_content": text_content,
            "source_path": file_info.filePath,
            "metadata": {
                **metadata.dict(),
                **extracted_metadata
            }
        }

        # 保存文档到数据库
        document_id = self.document_repository.save_document(document_data)

        return {
            "document_id": document_id,
            "file_name": file_info.filename,
            "file_info": {
                "file_size": file_info.file_size,
                "content_type": file_info.content_type,
                "file_path": file_info.filePath
            },
            "metadata": metadata.dict(),
            "indexed": False,
            "upload_time": document_data["upload_time"],
            "status": "success",
            "message": "文档记录创建成功"
        }

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文档详情

        Args:
            document_id: 文档ID

        Returns:
            文档详情或None（如不存在）
        """
        document = self.document_repository.get_document(document_id)

        if not document:
            return None

        # 格式化响应
        return {
            "document_id": document_id,
            "file_name": document.get("file_name"),
            "file_info": {
                "file_size": document.get("file_size"),
                "content_type": document.get("content_type"),
                "file_path": document.get("source_path")
            },
            "metadata": document.get("metadata", {}),
            "indexed": document.get("indexed", False),
            "upload_time": document.get("upload_time"),
            "status": "success"
        }

    async def list_documents(self, filters: Dict[str, Any], skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """
        获取文档列表

        Args:
            filters: 过滤条件
            skip: 分页起始位置
            limit: 每页数量

        Returns:
            文档列表响应
        """
        documents = self.document_repository.list_documents(
            filters=filters,
            skip=skip,
            limit=limit
        )

        # 格式化返回列表
        formatted_documents = []
        for doc in documents:
            formatted_documents.append({
                "document_id": doc.get("id"),
                "file_name": doc.get("file_name"),
                "file_info": {
                    "file_size": doc.get("file_size"),
                    "content_type": doc.get("content_type"),
                    "file_path": doc.get("source_path")
                },
                "metadata": doc.get("metadata", {}),
                "indexed": doc.get("indexed", False),
                "upload_time": doc.get("upload_time")
            })

        total = self.document_repository.count_documents(filters)

        return {
            "documents": formatted_documents,
            "total": total,
            "skip": skip,
            "limit": limit,
            "filters": filters
        }

    async def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        更新文档元数据

        Args:
            document_id: 文档ID
            metadata: 要更新的元数据

        Returns:
            更新后的文档或None（如不存在）
        """
        # 检查文档是否存在
        document = self.document_repository.get_document(document_id)

        if not document:
            return None

        # 更新元数据
        current_metadata = document.get("metadata", {})
        updated_metadata = {**current_metadata, **metadata}

        # 更新文档
        updated_document = self.document_repository.update_document(
            document_id=document_id,
            update_data={"metadata": updated_metadata}
        )

        if not updated_document:
            return None

        # 格式化响应
        return {
            "document_id": document_id,
            "file_name": updated_document.get("file_name"),
            "file_info": {
                "file_size": updated_document.get("file_size"),
                "content_type": updated_document.get("content_type"),
                "file_path": updated_document.get("source_path")
            },
            "metadata": updated_metadata,
            "indexed": updated_document.get("indexed", False),
            "upload_time": updated_document.get("upload_time"),
            "status": "success",
            "message": "元数据更新成功"
        }

    async def index_document(self, document_id: str) -> Dict[str, Any]:
        """
        为文档创建索引

        Args:
            document_id: 文档ID

        Returns:
            索引结果
        """
        # 检查文档是否存在
        document = self.document_repository.get_document(document_id)

        if not document:
            return {
                "success": False,
                "not_found": True,
                "message": f"文档ID {document_id} 不存在"
            }

        # 检查文档是否已经被索引
        if document.get("indexed", False):
            return {
                "success": True,
                "message": f"文档 {document_id} 已经索引过"
            }

        # 创建索引
        try:
            metadata = document.get("metadata", {})
            index_result = await self.search_service.index_documents([{
                "id": document_id,
                "text": document.get("text_content", ""),
                "file_name": document.get("file_name", ""),
                "title": metadata.get("title"),
                "tags": metadata.get("tags", []),
                "source": metadata.get("source"),
                **{k: v for k, v in metadata.items() if k not in ["title", "tags", "source"]}
            }])

            # 更新文档索引状态
            self.document_repository.update_document(
                document_id=document_id,
                update_data={
                    "indexed": True,
                    "vector_id": index_result.get("vector_ids", [])[0] if index_result.get("vector_ids") else None
                }
            )

            return {
                "success": True,
                "message": f"文档 {document_id} 索引成功"
            }

        except Exception as e:
            logger.error(f"文档索引失败: {str(e)}")
            return {
                "success": False,
                "message": f"文档索引失败: {str(e)}"
            }

    async def delete_document(self, document_id: str, delete_file: bool = False) -> Dict[str, Any]:
        """
        删除文档

        Args:
            document_id: 文档ID
            delete_file: 是否删除物理文件

        Returns:
            删除结果
        """
        # 检查文档是否存在
        document = self.document_repository.get_document(document_id)

        if not document:
            return {
                "success": False,
                "not_found": True,
                "message": f"文档ID {document_id} 不存在"
            }

        # 如果文档已索引，从向量存储中删除
        if document.get("indexed", False) and document.get("vector_id"):
            try:
                await self.search_service.delete_vector(document.get("vector_id"))
                logger.info(f"从向量存储中删除文档 {document_id}")
            except Exception as e:
                logger.warning(f"从向量存储中删除文档 {document_id} 失败: {str(e)}")

        # 从数据库中删除文档
        self.document_repository.delete_document(document_id)

        # 如果选择删除文件，则删除物理文件
        file_deleted = False
        if delete_file and document.get("file_path") and os.path.exists(document.get("file_path")):
            try:
                os.remove(document.get("file_path"))
                logger.info(f"删除文件: {document.get('file_path')}")
                file_deleted = True
            except Exception as e:
                logger.warning(f"删除文件失败: {str(e)}")

        return {
            "success": True,
            "message": f"文档 {document_id} 删除成功" + (", 物理文件已删除" if file_deleted else ""),
            "file_deleted": file_deleted
        }

    async def get_document_content(self, document_id: str) -> Optional[str]:
        """
        获取文档文本内容

        Args:
            document_id: 文档ID

        Returns:
            文档内容或None（如不存在）
        """
        document = self.document_repository.get_document(document_id)

        if not document:
            return None

        return document.get("text_content", "")

    async def batch_index_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        批量索引多个文档

        Args:
            document_ids: 文档ID列表

        Returns:
            批量索引结果
        """
        if not document_ids:
            return {
                "success": True,
                "message": "没有提供要索引的文档ID"
            }

        # 获取文档详情
        documents = []
        doc_ids_to_update = []

        for doc_id in document_ids:
            doc = self.document_repository.get_document(doc_id)
            if doc and not doc.get("indexed", False):
                doc_ids_to_update.append(doc_id)
                metadata = doc.get("metadata", {})
                documents.append({
                    "id": doc_id,
                    "text": doc.get("text_content", ""),
                    "file_name": doc.get("file_name", ""),
                    "title": metadata.get("title"),
                    "tags": metadata.get("tags", []),
                    "source": metadata.get("source"),
                    **{k: v for k, v in metadata.items() if k not in ["title", "tags", "source"]}
                })

        if not documents:
            return {
                "success": True,
                "message": "没有需要索引的文档"
            }

        # 批量创建索引
        try:
            index_result = await self.search_service.index_documents(documents)

            # 更新文档索引状态
            for i, doc_id in enumerate(doc_ids_to_update):
                if i < len(index_result.get("vector_ids", [])):
                    self.document_repository.update_document(
                        document_id=doc_id,
                        update_data={
                            "indexed": True,
                            "vector_id": index_result.get("vector_ids", [])[i]
                        }
                    )

            return {
                "success": True,
                "message": f"成功索引 {len(documents)} 个文档",
                "indexed_count": len(documents),
                "total_documents": len(document_ids)
            }

        except Exception as e:
            logger.error(f"批量索引文档失败: {str(e)}")
            return {
                "success": False,
                "message": f"批量索引文档失败: {str(e)}"
            }

    async def batch_update_tags(self, document_ids: List[str], tags: List[str], operation: str) -> Dict[str, Any]:
        """
        批量更新文档标签

        Args:
            document_ids: 文档ID列表
            tags: 标签列表
            operation: 操作类型（add, remove, replace）

        Returns:
            更新结果
        """
        if not document_ids or not tags:
            return {
                "success": True,
                "message": "没有提供要更新的文档ID或标签"
            }

        updated_count = 0

        # 更新每个文档的标签
        for doc_id in document_ids:
            doc = self.document_repository.get_document(doc_id)
            if not doc:
                continue

            # 获取当前标签
            metadata = doc.get("metadata", {})
            current_tags = metadata.get("tags", [])

            # 根据操作类型更新标签
            if operation == "add":
                # 添加新标签（避免重复）
                new_tags = list(set(current_tags + tags))
            elif operation == "remove":
                # 移除指定标签
                new_tags = [tag for tag in current_tags if tag not in tags]
            elif operation == "replace":
                # 替换为新标签
                new_tags = tags
            else:
                # 不支持的操作
                continue

            # 更新元数据
            metadata["tags"] = new_tags
            self.document_repository.update_document(
                document_id=doc_id,
                update_data={"metadata": metadata}
            )

            updated_count += 1

        return {
            "success": True,
            "message": f"成功更新 {updated_count} 个文档的标签",
            "updated_count": updated_count,
            "total_documents": len(document_ids)
        }

    logger = logging.getLogger(__name__)

    # def process_file_path(self, file_path: str, additional_metadata: Optional[Dict[str, Any]] = None,
    #                       add_to_knowledge_base: bool = False) -> Dict[str, Any]:
    #     """
    #     处理服务器上的文件（通过绝对路径）
    #
    #     Args:
    #         file_path: 服务器上文件的绝对路径
    #         additional_metadata: 可选的附加元数据
    #         add_to_knowledge_base: 是否将处理后的文件添加到知识库
    #
    #     Returns:
    #         包含处理结果的字典
    #     """
    #     try:
    #         # 检查文件是否存在
    #         if not os.path.exists(file_path):
    #             logger.error(f"文件不存在: {file_path}")
    #             return {
    #                 "status": "error",
    #                 "error": f"文件不存在: {file_path}",
    #                 "content": "",
    #                 "metadata": {}
    #             }
    #
    #         # 检查文件是否为常规文件
    #         if not os.path.isfile(file_path):
    #             logger.error(f"路径不是文件: {file_path}")
    #             return {
    #                 "status": "error",
    #                 "error": f"路径不是文件: {file_path}",
    #                 "content": "",
    #                 "metadata": {}
    #             }
    #
    #         # 处理文件
    #         logger.info(f"开始处理文件: {file_path}")
    #
    #         # 获取合适的处理器
    #         processor = self.processor_factory.get_processor(file_path)
    #
    #         # 提取文本和元数据
    #         content = processor.extract_text(file_path)
    #         metadata = processor.extract_metadata(file_path)
    #
    #         # 如果内容为空，返回错误
    #         if not content:
    #             logger.warning(f"无法从文件中提取内容: {file_path}")
    #             return {
    #                 "status": "warning",
    #                 "error": "提取内容为空",
    #                 "content": "",
    #                 "metadata": metadata
    #             }
    #
    #         # 合并附加元数据
    #         if additional_metadata:
    #             metadata.update(additional_metadata)
    #
    #         # 添加基本文件信息
    #         metadata.update({
    #             "source_file_path": file_path,
    #             "processed_time": datetime.now().isoformat(),
    #             "processing_time": 0.0
    #         })
    #
    #         # 生成文档ID
    #         doc_id = hashlib.md5(f"{file_path}_{datetime.now().isoformat()}".encode()).hexdigest()
    #         metadata["doc_id"] = doc_id
    #
    #         # 对内容进行分块
    #         chunks = self.chunker.chunk_text(content)
    #
    #         # 如果需要添加到知识库
    #         index_result = None
    #         if add_to_knowledge_base:
    #             try:
    #                 # 导入SearchService
    #                 from services.search_service import SearchService
    #                 from config.settings import EMBEDDING_MODEL, DEVICE, VECTOR_STORE_TYPE, VECTOR_STORE_PATH
    #
    #                 # 初始化搜索服务
    #                 search_config = {
    #                     "model_name": EMBEDDING_MODEL,
    #                     "device": DEVICE,
    #                     "store_type": VECTOR_STORE_TYPE,
    #                     "index_path": VECTOR_STORE_PATH
    #                 }
    #                 search_service = SearchService()
    #
    #                 # 准备文档列表
    #                 documents = []
    #
    #                 # 如果有分块，为每个分块创建一个文档
    #                 if chunks:
    #                     for chunk in chunks:
    #                         chunk_metadata = metadata.copy()
    #                         chunk_metadata.update({
    #                             "chunk_id": chunk["chunk_id"],
    #                             "chunk_index": chunk["chunk_index"]
    #                         })
    #                         documents.append({
    #                             "text": chunk["text"],
    #                             **chunk_metadata
    #                         })
    #                 else:
    #                     # 没有分块，使用完整内容
    #                     documents.append({
    #                         "text": content,
    #                         **metadata
    #                     })
    #
    #                 # 索引文档
    #                 logger.info(f"将文件 '{file_path}' 添加到知识库")
    #                 index_result = search_service.index_documents(documents)
    #
    #                 logger.info(f"文件已添加到知识库: {index_result}")
    #
    #             except Exception as index_err:
    #                 logger.error(f"添加到知识库时出错: {str(index_err)}")
    #                 index_result = {
    #                     "status": "error",
    #                     "error": f"添加到知识库失败: {str(index_err)}"
    #                 }
    #
    #         # 返回结果
    #         result = {
    #             "doc_id": doc_id,
    #             "content": content,
    #             "metadata": metadata,
    #             "chunks": chunks,
    #             "status": "success"
    #         }
    #
    #         # 如果添加到知识库，包含索引结果
    #         if index_result:
    #             result["index_result"] = index_result
    #
    #         return result
    #
    #     except Exception as e:
    #         logger.error(f"处理文件时出错: {str(e)}")
    #         return {
    #             "status": "error",
    #             "error": f"处理失败: {str(e)}",
    #             "content": "",
    #             "metadata": {}
    #         }

    def process_file_path(self, file_path: str, additional_metadata: Optional[Dict[str, Any]] = None,
                          add_to_knowledge_base: bool = False) -> Dict[str, Any]:
        """
        处理服务器上的文件（通过绝对路径）

        Args:
            file_path: 服务器上文件的绝对路径
            additional_metadata: 可选的附加元数据
            add_to_knowledge_base: 是否将处理后的文件添加到知识库

        Returns:
            包含处理结果的字典
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return {
                    "status": "error",
                    "error": f"文件不存在: {file_path}",
                    "content": "",
                    "metadata": {}
                }

            # 检查文件是否为常规文件
            if not os.path.isfile(file_path):
                logger.error(f"路径不是文件: {file_path}")
                return {
                    "status": "error",
                    "error": f"路径不是文件: {file_path}",
                    "content": "",
                    "metadata": {}
                }

            # 处理文件
            logger.info(f"开始处理文件: {file_path}")

            # 获取合适的处理器
            processor = self.processor_factory.get_processor(file_path)

            # 提取文本和元数据
            content = processor.extract_text(file_path)
            metadata = processor.extract_metadata(file_path)

            # 如果内容为空，返回错误
            if not content:
                logger.warning(f"无法从文件中提取内容: {file_path}")
                return {
                    "status": "warning",
                    "error": "提取内容为空",
                    "content": "",
                    "metadata": metadata
                }

            # 合并附加元数据
            if additional_metadata:
                metadata.update(additional_metadata)

            # 添加基本文件信息
            metadata.update({
                "source_file_path": file_path,
                "processed_time": datetime.now().isoformat(),
                "processing_time": 0.0
            })

            # 生成文档ID
            doc_id = hashlib.md5(f"{file_path}_{datetime.now().isoformat()}".encode()).hexdigest()
            metadata["doc_id"] = doc_id

            # 对内容进行分块
            chunks = self.chunker.chunk_text(content)

            # 保存文档到文档仓库
            document_data = {
                "id": doc_id,
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "file_type": os.path.splitext(file_path)[1].lower(),
                "content_type": metadata.get("content_type", "text/plain"),
                "upload_time": datetime.now().isoformat(),
                "text_content": content,
                "source_path": file_path,
                "metadata": metadata,
                "indexed": False
            }

            # 保存到文档仓库
            self.document_repository.save_document(document_data)
            logger.info(f"已保存文档到仓库: {doc_id}")

            # 如果需要添加到知识库
            index_result = None
            if add_to_knowledge_base:
                try:
                    # 导入SearchService
                    from services.search_service import SearchService
                    from config.settings import EMBEDDING_MODEL, DEVICE, VECTOR_STORE_TYPE, VECTOR_STORE_PATH

                    # 初始化搜索服务
                    search_config = {
                        "model_name": EMBEDDING_MODEL,
                        "device": DEVICE,
                        "store_type": VECTOR_STORE_TYPE,
                        "index_path": VECTOR_STORE_PATH
                    }
                    search_service = SearchService(search_config)

                    # 准备文档列表
                    documents = []

                    # 如果有分块，为每个分块创建一个文档
                    if chunks:
                        for chunk in chunks:
                            chunk_metadata = metadata.copy()
                            chunk_metadata.update({
                                "chunk_id": chunk["chunk_id"],
                                "chunk_index": chunk["chunk_index"],
                                "document_id": doc_id  # 添加文档ID到元数据
                            })
                            documents.append({
                                "text": chunk["text"],
                                **chunk_metadata
                            })
                    else:
                        # 没有分块，使用完整内容
                        documents.append({
                            "text": content,
                            "document_id": doc_id,  # 添加文档ID到元数据
                            **metadata
                        })

                    # 索引文档
                    logger.info(f"将文件 '{file_path}' 添加到知识库，共 {len(documents)} 个文档")
                    index_result = search_service.index_documents(documents)

                    # 更新文档索引状态
                    if index_result.get("status") == "success":
                        self.document_repository.update_document(
                            document_id=doc_id,
                            update_data={
                                "indexed": True,
                                "vector_ids": index_result.get("vector_ids", [])
                            }
                        )
                        logger.info(f"文件已成功添加到知识库: {index_result}")
                    else:
                        logger.error(f"添加到知识库失败: {index_result}")

                except Exception as index_err:
                    logger.error(f"添加到知识库时出错: {str(index_err)}", exc_info=True)
                    index_result = {
                        "status": "error",
                        "error": f"添加到知识库失败: {str(index_err)}"
                    }

            # 返回结果
            result = {
                "doc_id": doc_id,
                "content": content,
                "metadata": metadata,
                "chunks": chunks,
                "status": "success"
            }

            # 如果添加到知识库，包含索引结果
            if index_result:
                result["index_result"] = index_result

            return result

        except Exception as e:
            logger.error(f"处理文件时出错: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": f"处理失败: {str(e)}",
                "content": "",
                "metadata": {}
            }