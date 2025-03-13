# core/document_processors/base_processor.py
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional


class BaseDocumentProcessor(ABC):
    """文档处理器的基类，定义所有文档处理器必须实现的接口"""

    def __init__(self, config=None):
        """初始化处理器

        Args:
            config: 可选的配置参数
        """
        self.config = config or {}

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """从文档中提取纯文本"""
        pass

    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """从文档中提取元数据"""
        pass

    def process(self, file_path: str) -> Dict[str, Any]:
        """处理文档，提取文本和元数据

        Args:
            file_path: 文件路径

        Returns:
            包含文本和元数据的字典
        """
        text = self.extract_text(file_path)
        metadata = self.extract_metadata(file_path)

        return {
            "text": text,
            "metadata": metadata
        }