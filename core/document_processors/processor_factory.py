# core/document_processors/processor_factory.py
import os
import logging
from typing import Dict, Any, Optional, Type, List

from .base_processor import BaseDocumentProcessor
from .pdf_processor import PDFProcessor
from .docx_processor import DocxProcessor
from .excel_processor import ExcelProcessor
from .text_processor import TextProcessor

logger = logging.getLogger(__name__)


class DocumentProcessorFactory:
    """文档处理器工厂，根据文件类型创建相应的处理器"""

    def __init__(self, config=None):
        """
        初始化处理器工厂

        Args:
            config: 处理器配置，可以为每种类型的处理器提供单独的配置
        """
        self.config = config or {}
        self.processors = {}  # 文件扩展名到处理器类的映射
        self._register_default_processors()

        logger.info("初始化文档处理器工厂")

    def _register_default_processors(self):
        """注册默认的文档处理器"""
        # PDF处理器
        self.register_processor(
            extensions=["pdf"],
            processor_class=PDFProcessor,
            config=self.config.get("pdf", {})
        )

        # Word处理器
        self.register_processor(
            extensions=["docx", "doc"],
            processor_class=DocxProcessor,
            config=self.config.get("docx", {})
        )

        # Excel处理器
        self.register_processor(
            extensions=["xlsx", "xls", "xlsm"],
            processor_class=ExcelProcessor,
            config=self.config.get("excel", {})
        )

        # 文本处理器
        self.register_processor(
            extensions=["txt", "csv", "json", "xml", "html", "md", "markdown", "log"],
            processor_class=TextProcessor,
            config=self.config.get("text", {})
        )

        logger.debug(f"已注册处理器: {', '.join(self.processors.keys())}")

    def register_processor(self, extensions: List[str], processor_class: Type[BaseDocumentProcessor],
                           config: Optional[Dict[str, Any]] = None):
        """
        注册文档处理器

        Args:
            extensions: 文件扩展名列表
            processor_class: 处理器类
            config: 处理器配置
        """
        processor_instance = processor_class(config)

        for ext in extensions:
            ext = ext.lower().lstrip('.')
            self.processors[ext] = processor_instance
            logger.debug(f"注册处理器: {ext} -> {processor_class.__name__}")

    def get_processor(self, file_path: str) -> BaseDocumentProcessor:
        """
        根据文件路径获取适当的处理器

        Args:
            file_path: 文件路径

        Returns:
            适合处理该文件的处理器
        """
        # 获取文件扩展名
        _, ext = os.path.splitext(file_path)
        ext = ext.lstrip('.').lower()

        # 查找匹配的处理器
        processor = self.processors.get(ext)

        if not processor:
            logger.warning(f"未找到处理扩展名 '{ext}' 的处理器，使用默认文本处理器")
            processor = TextProcessor(self.config.get("text", {}))

        logger.info(f"为文件 '{os.path.basename(file_path)}' 选择处理器: {processor.__class__.__name__}")
        return processor

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        处理文档，自动选择合适的处理器

        Args:
            file_path: 文件路径

        Returns:
            处理结果，包含文本内容和元数据
        """
        import time

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            # 获取合适的处理器
            processor = self.get_processor(file_path)

            # 处理文档
            start_time = time.time()
            result = processor.process(file_path)
            processing_time = time.time() - start_time

            # 添加处理信息
            result.update({
                "processing_time": processing_time,
                "processor_type": processor.__class__.__name__
            })

            logger.info(f"文档处理完成: '{os.path.basename(file_path)}', 用时 {processing_time:.2f}秒")
            return result

        except Exception as e:
            logger.error(f"处理文档 '{file_path}' 时出错: {str(e)}")
            raise