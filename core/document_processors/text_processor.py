# core/document_processors/text_processor.py
import os
import csv
import json
import chardet
from typing import Dict, Any
import logging
from datetime import datetime
import re

from .base_processor import BaseDocumentProcessor

logger = logging.getLogger(__name__)


class TextProcessor(BaseDocumentProcessor):
    """文本处理器，用于处理各种文本文件(txt, csv, json等)"""

    def __init__(self, config=None):
        """
        初始化文本处理器

        Args:
            config: 可选配置参数
        """
        super().__init__(config or {})
        self.encoding = config.get("encoding", None)  # 自动检测或指定编码
        self.csv_delimiter = config.get("csv_delimiter", None)  # CSV分隔符，None为自动检测
        self.max_file_size = config.get("max_file_size", 10 * 1024 * 1024)  # 默认10MB
        self.preserve_line_breaks = config.get("preserve_line_breaks", True)

        logger.info(
            f"初始化文本处理器: encoding={self.encoding or 'auto'}, csv_delimiter={self.csv_delimiter or 'auto'}")

    def extract_text(self, file_path: str) -> str:
        """
        从文本文件中提取内容

        Args:
            file_path: 文本文件路径

        Returns:
            提取的文本内容
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            logger.warning(f"文件大小 ({file_size} 字节) 超过限制 ({self.max_file_size} 字节)")
            return f"[文件过大无法处理: {file_size} 字节]"

        try:
            # 根据文件扩展名判断处理方式
            _, ext = os.path.splitext(file_path)
            ext = ext.lower().lstrip('.')

            if ext == 'csv':
                return self._process_csv(file_path)
            elif ext == 'json':
                return self._process_json(file_path)
            else:
                return self._process_plain_text(file_path)

        except Exception as e:
            logger.error(f"提取文本时出错: {str(e)}")
            raise

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        提取文本文件的元数据

        Args:
            file_path: 文本文件路径

        Returns:
            包含文档元数据的字典
        """
        try:
            # 获取文件基本信息
            file_size = os.path.getsize(file_path)
            _, ext = os.path.splitext(file_path)
            ext = ext.lower().lstrip('.')

            metadata = {
                "file_name": os.path.basename(file_path),
                "file_extension": ext,
                "file_size": file_size,
                "last_modified": datetime.fromtimestamp(
                    os.path.getmtime(file_path)
                ).isoformat(),
                "created": datetime.fromtimestamp(
                    os.path.getctime(file_path)
                ).isoformat(),
            }

            # 根据文件类型添加特定元数据
            if ext == 'csv':
                csv_metadata = self._extract_csv_metadata(file_path)
                metadata.update(csv_metadata)
            elif ext == 'json':
                json_metadata = self._extract_json_metadata(file_path)
                metadata.update(json_metadata)
            else:
                text_metadata = self._extract_plain_text_metadata(file_path)
                metadata.update(text_metadata)

            return metadata

        except Exception as e:
            logger.error(f"提取文本元数据时出错: {str(e)}")
            raise

    def _detect_encoding(self, file_path: str) -> str:
        """
        检测文件编码

        Args:
            file_path: 文件路径

        Returns:
            检测到的编码
        """
        if self.encoding:
            return self.encoding

        # 读取文件的前10KB用于检测编码
        with open(file_path, 'rb') as f:
            raw_data = f.read(10240)

        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'
        confidence = result['confidence']

        logger.debug(f"检测到编码: {encoding} (置信度: {confidence:.2f})")
        return encoding

    def _process_plain_text(self, file_path: str) -> str:
        """处理普通文本文件"""
        encoding = self._detect_encoding(file_path)

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            if not self.preserve_line_breaks:
                # 将连续的换行符替换为单个换行符
                content = re.sub(r'\n\s*\n', '\n', content)

            return content

        except UnicodeDecodeError:
            logger.warning(f"使用 {encoding} 解码失败，尝试 utf-8")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                logger.error("无法解码文件，尝试二进制模式")
                with open(file_path, 'rb') as f:
                    return str(f.read())

    def _process_csv(self, file_path: str) -> str:
        """处理CSV文件"""
        encoding = self._detect_encoding(file_path)

        try:
            with open(file_path, 'r', encoding=encoding, newline='') as f:
                # 检测分隔符
                if not self.csv_delimiter:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    f.seek(0)
                    reader = csv.reader(f, dialect)
                else:
                    reader = csv.reader(f, delimiter=self.csv_delimiter)

                # 读取CSV数据
                rows = list(reader)

                if not rows:
                    return "[空CSV文件]"

                # 格式化为表格文本
                header = rows[0]
                data_rows = rows[1:] if len(rows) > 1 else []

                # 计算每列的最大宽度
                col_widths = [max(len(str(row[i])) for row in rows if i < len(row)) for i in range(len(header))]

                # 生成表头
                header_text = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(header))
                separator = "-+-".join("-" * w for w in col_widths)

                # 生成数据行
                data_text = []
                for row in data_rows:
                    row_text = " | ".join(str(cell).ljust(col_widths[i]) if i < len(row) else " " * col_widths[i]
                                          for i, cell in enumerate(row))
                    data_text.append(row_text)

                # 组合表格
                return header_text + "\n" + separator + "\n" + "\n".join(data_text)

        except Exception as e:
            logger.warning(f"CSV处理失败: {str(e)}，尝试作为普通文本处理")
            return self._process_plain_text(file_path)

    def _process_json(self, file_path: str) -> str:
        """处理JSON文件"""
        encoding = self._detect_encoding(file_path)

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)

            # 格式化JSON
            formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
            return formatted_json

        except json.JSONDecodeError:
            logger.warning("JSON解析失败，尝试作为普通文本处理")
            return self._process_plain_text(file_path)

    def _extract_plain_text_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取普通文本文件的元数据"""
        try:
            encoding = self._detect_encoding(file_path)

            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            # 统计行数、字数和字符数
            lines = content.splitlines()
            words = re.findall(r'\b\w+\b', content)

            return {
                "encoding": encoding,
                "line_count": len(lines),
                "word_count": len(words),
                "char_count": len(content),
                "content_type": "text/plain"
            }
        except Exception as e:
            logger.warning(f"提取文本元数据时出错: {str(e)}")
            return {"content_type": "text/plain"}

    def _extract_csv_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取CSV文件的元数据"""
        try:
            encoding = self._detect_encoding(file_path)

            with open(file_path, 'r', encoding=encoding, newline='') as f:
                # 检测分隔符
                if not self.csv_delimiter:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    delimiter = dialect.delimiter
                    f.seek(0)
                else:
                    delimiter = self.csv_delimiter

                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)

                if not rows:
                    return {"content_type": "text/csv", "row_count": 0, "column_count": 0}

                header = rows[0]
                data_rows = rows[1:] if len(rows) > 1 else []

                return {
                    "encoding": encoding,
                    "content_type": "text/csv",
                    "delimiter": delimiter,
                    "row_count": len(data_rows),
                    "column_count": len(header),
                    "header": header,
                    "has_header": csv.Sniffer().has_header(open(file_path, 'r', encoding=encoding).read(1024))
                }
        except Exception as e:
            logger.warning(f"提取CSV元数据时出错: {str(e)}")
            return {"content_type": "text/csv"}

    def _extract_json_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取JSON文件的元数据"""
        try:
            encoding = self._detect_encoding(file_path)

            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)

            # 判断JSON结构类型
            json_type = self._determine_json_structure(data)

            metadata = {
                "encoding": encoding,
                "content_type": "application/json",
                "json_type": json_type
            }

            # 添加更多结构信息
            if json_type == "array":
                metadata["array_length"] = len(data)
                if data and isinstance(data[0], dict):
                    metadata["schema_fields"] = list(data[0].keys())
            elif json_type == "object":
                metadata["object_keys"] = list(data.keys())

            return metadata
        except Exception as e:
            logger.warning(f"提取JSON元数据时出错: {str(e)}")
            return {"content_type": "application/json"}

    def _determine_json_structure(self, data: Any) -> str:
        """确定JSON数据的结构类型"""
        if isinstance(data, list):
            return "array"
        elif isinstance(data, dict):
            return "object"
        else:
            return "primitive"