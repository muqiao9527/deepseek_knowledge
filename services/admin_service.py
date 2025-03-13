# services/admin_service.py
import os
import logging
import time
import shutil
from typing import Dict, Any, List, Optional

from services.document_service import DocumentService
from services.search_service import SearchService

logger = logging.getLogger(__name__)


class AdminService:
    """管理服务，提供系统维护和管理功能"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化管理服务

        Args:
            config: 服务配置
        """
        self.config = config or {}

        # 获取文档服务和搜索服务实例
        self.document_service = DocumentService(self.config.get("document", {}))
        self.search_service = SearchService(self.config.get("search", {}))

        logger.info("管理服务初始化完成")

    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态信息

        Returns:
            系统状态信息
        """
        # 获取文档服务统计
        doc_stats = self.document_service.get_statistics()

        # 获取搜索服务统计
        search_stats = self.search_service.get_stats()

        # 获取系统信息
        import platform
        import psutil

        try:
            system_info = {
                "os": platform.system(),
                "os_version": platform.version(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": {
                    "total": psutil.disk_usage('/').total,
                    "used": psutil.disk_usage('/').used,
                    "free": psutil.disk_usage('/').free
                }
            }
        except ImportError:
            system_info = {
                "os": platform.system(),
                "os_version": platform.version(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count()
            }

        return {
            "document_stats": doc_stats,
            "search_stats": search_stats,
            "system_info": system_info,
            "timestamp": time.time(),
            "status": "running"
        }

    def rebuild_index(self) -> Dict[str, Any]:
        """
        重建搜索索引

        Returns:
            重建结果
        """
        try:
            start_time = time.time()

            # 获取所有文档
            documents = self.document_service.get_documents(limit=1000)

            # 准备索引数据
            index_docs = []
            for doc in documents:
                # 获取文档内容
                doc_with_content = self.document_service.get_document(doc["id"], include_content=True)
                if not doc_with_content or not doc_with_content.get("content"):
                    continue

                # 获取文档分块
                chunks = self.document_service.document_repository.get_document_chunks(doc["id"])

                # 如果没有分块，使用文档服务的分块器重新分块
                if not chunks:
                    content = doc_with_content.get("content", "")
                    chunks = self.document_service.chunker.chunk_text(content)

                # 准备索引数据
                for chunk in chunks:
                    text = chunk.get("content") or chunk.get("text", "")
                    if not text:
                        continue

                    metadata = {
                        "document_id": doc["id"],
                        "chunk_index": chunk.get("chunk_index", 0),
                        "title": doc.get("title", ""),
                        "file_name": doc.get("file_name", ""),
                        "file_extension": doc.get("file_extension", "")
                    }

                    # 添加原始元数据
                    if "metadata" in doc and isinstance(doc["metadata"], dict):
                        metadata.update(doc["metadata"])

                    index_docs.append({
                        "text": text,
                        **metadata
                    })

            # 清除旧索引
            self._clear_vector_store()

            # 索引文档
            index_result = self.search_service.index_documents(index_docs)

            processing_time = time.time() - start_time

            return {
                "status": "success",
                "document_count": len(documents),
                "indexed_chunk_count": index_result.get("indexed_count", 0),
                "processing_time": processing_time,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"重建索引时出错: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }

    def _clear_vector_store(self):
        """清除向量存储"""
        try:
            # 获取向量存储路径
            vector_store_path = self.search_service.vector_store.index_path

            # 备份原始向量存储
            if os.path.exists(vector_store_path):
                backup_path = f"{vector_store_path}_backup_{int(time.time())}"
                shutil.copytree(vector_store_path, backup_path)
                logger.info(f"向量存储已备份到 {backup_path}")

                # 删除原始向量存储
                shutil.rmtree(vector_store_path)
                logger.info(f"向量存储已清除: {vector_store_path}")

            # 创建新的向量存储目录
            os.makedirs(vector_store_path, exist_ok=True)

            # 重新初始化向量存储
            self.search_service.vector_store._create_empty_index()

        except Exception as e:
            logger.error(f"清除向量存储时出错: {str(e)}")
            raise

    def backup_system(self, backup_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        备份系统数据

        Args:
            backup_dir: 备份目录，默认为当前目录下的backup文件夹

        Returns:
            备份结果
        """
        try:
            # 设置备份目录
            if not backup_dir:
                backup_dir = f"./backup/backup_{int(time.time())}"

            os.makedirs(backup_dir, exist_ok=True)

            # 备份向量存储
            vector_store_path = self.search_service.vector_store.index_path
            if os.path.exists(vector_store_path):
                vector_backup_path = os.path.join(backup_dir, "vector_store")
                shutil.copytree(vector_store_path, vector_backup_path)
                logger.info(f"向量存储已备份到 {vector_backup_path}")

            # 备份文档数据库
            db_path = self.document_service.document_repository.db_path
            if os.path.exists(db_path):
                db_backup_path = os.path.join(backup_dir, os.path.basename(db_path))
                shutil.copy2(db_path, db_backup_path)
                logger.info(f"文档数据库已备份到 {db_backup_path}")

            # 备份上传文件
            upload_dir = self.document_service.upload_dir
            if os.path.exists(upload_dir):
                upload_backup_path = os.path.join(backup_dir, "uploads")
                shutil.copytree(upload_dir, upload_backup_path)
                logger.info(f"上传文件已备份到 {upload_backup_path}")

            return {
                "status": "success",
                "backup_directory": backup_dir,
                "backup_time": time.time(),
                "backup_items": {
                    "vector_store": os.path.exists(vector_store_path),
                    "document_database": os.path.exists(db_path),
                    "uploaded_files": os.path.exists(upload_dir)
                }
            }

        except Exception as e:
            logger.error(f"备份系统时出错: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "backup_directory": backup_dir,
                "backup_time": time.time()
            }

    def restore_system(self, backup_dir: str) -> Dict[str, Any]:
        """
        从备份恢复系统数据

        Args:
            backup_dir: 备份目录

        Returns:
            恢复结果
        """
        try:
            if not os.path.exists(backup_dir):
                return {
                    "status": "error",
                    "error": f"备份目录不存在: {backup_dir}"
                }

            # 恢复向量存储
            vector_backup_path = os.path.join(backup_dir, "vector_store")
            if os.path.exists(vector_backup_path):
                vector_store_path = self.search_service.vector_store.index_path

                # 备份当前向量存储
                if os.path.exists(vector_store_path):
                    current_backup = f"{vector_store_path}_before_restore_{int(time.time())}"
                    shutil.copytree(vector_store_path, current_backup)
                    logger.info(f"当前向量存储已备份到 {current_backup}")

                    # 删除当前向量存储
                    shutil.rmtree(vector_store_path)

                # 恢复向量存储
                shutil.copytree(vector_backup_path, vector_store_path)
                logger.info(f"向量存储已从 {vector_backup_path} 恢复")

                # 重新初始化向量存储
                self.search_service.vector_store._load_or_create_index()

            # 恢复文档数据库
            db_backup_path = None
            for file in os.listdir(backup_dir):
                if file.endswith(".db"):
                    db_backup_path = os.path.join(backup_dir, file)
                    break

            if db_backup_path and os.path.exists(db_backup_path):
                db_path = self.document_service.document_repository.db_path

                # 备份当前数据库
                if os.path.exists(db_path):
                    current_backup = f"{db_path}_before_restore_{int(time.time())}"
                    shutil.copy2(db_path, current_backup)
                    logger.info(f"当前数据库已备份到 {current_backup}")

                # 恢复数据库
                shutil.copy2(db_backup_path, db_path)
                logger.info(f"文档数据库已从 {db_backup_path} 恢复")

            # 恢复上传文件
            upload_backup_path = os.path.join(backup_dir, "uploads")
            if os.path.exists(upload_backup_path):
                upload_dir = self.document_service.upload_dir

                # 备份当前上传文件
                if os.path.exists(upload_dir):
                    current_backup = f"{upload_dir}_before_restore_{int(time.time())}"
                    shutil.copytree(upload_dir, current_backup)
                    logger.info(f"当前上传文件已备份到 {current_backup}")

                    # 删除当前上传文件
                    shutil.rmtree(upload_dir)

                # 恢复上传文件
                shutil.copytree(upload_backup_path, upload_dir)
                logger.info(f"上传文件已从 {upload_backup_path} 恢复")

            return {
                "status": "success",
                "restore_time": time.time(),
                "restored_items": {
                    "vector_store": os.path.exists(vector_backup_path),
                    "document_database": db_backup_path is not None,
                    "uploaded_files": os.path.exists(upload_backup_path)
                }
            }

        except Exception as e:
            logger.error(f"恢复系统时出错: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "restore_time": time.time()
            }

    def get_logs(self, limit: int = 100, level: str = "INFO") -> List[Dict[str, Any]]:
        """
        获取系统日志

        Args:
            limit: 返回的最大日志条数
            level: 日志级别过滤

        Returns:
            日志列表
        """
        # 注意：实际实现需要根据系统的日志配置
        # 这里仅提供一个示例实现
        try:
            log_file = "./logs/app.log"
            level_map = {
                "DEBUG": 10,
                "INFO": 20,
                "WARNING": 30,
                "ERROR": 40,
                "CRITICAL": 50
            }
            min_level = level_map.get(level.upper(), 20)

            if not os.path.exists(log_file):
                return []

            logs = []
            import re

            # 简单的日志解析正则
            log_pattern = re.compile(
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (.+?) - (.+)'
            )

            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    match = log_pattern.match(line)
                    if match:
                        timestamp, log_level, logger_name, message = match.groups()

                        # 过滤日志级别
                        if level_map.get(log_level, 0) >= min_level:
                            logs.append({
                                "timestamp": timestamp,
                                "level": log_level,
                                "logger": logger_name,
                                "message": message
                            })

            # 返回最近的日志
            return logs[-limit:]

        except Exception as e:
            logger.error(f"获取日志时出错: {str(e)}")
            return [{
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "level": "ERROR",
                "logger": "admin_service",
                "message": f"获取日志时出错: {str(e)}"
            }]