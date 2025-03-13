# core/vectorization/chunking.py
import re
import logging
from typing import List, Dict, Any
import hashlib

logger = logging.getLogger(__name__)


class TextChunker:
    """文本分块器，负责将长文本分解为适合嵌入的短文本块"""

    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 chunking_method: str = "text"):
        """
        初始化文本分块器

        Args:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
            chunking_method: 分块方法 ('text', 'sentence', 'paragraph')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_method = chunking_method

        logger.info(f"初始化文本分块器: 大小={chunk_size}, 重叠={chunk_overlap}, 方法={chunking_method}")

        # 分块方法映射
        self.chunking_methods = {
            "text": self._chunk_by_text,
            "sentence": self._chunk_by_sentence,
            "paragraph": self._chunk_by_paragraph
        }

    def _chunk_by_text(self, text: str) -> List[str]:
        """按固定大小分块，无视文本结构"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            # 确定当前块的结束位置
            end = min(start + self.chunk_size, text_len)

            # 添加当前块
            chunks.append(text[start:end])

            # 移动开始位置，考虑重叠
            start = start + self.chunk_size - self.chunk_overlap

        return chunks

    def _chunk_by_sentence(self, text: str) -> List[str]:
        """按句子边界分块"""
        # 简单的句子分割正则表达式，同时处理中英文句子
        sentence_endings = r'(?<=[。！？.!?])\s*'
        sentences = re.split(sentence_endings, text)

        # 过滤空句子
        sentences = [s for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # 如果单个句子超过块大小，按文本分块处理
            if sentence_size > self.chunk_size:
                # 先处理当前累积的块
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # 处理大句子
                sentence_chunks = self._chunk_by_text(sentence)
                chunks.extend(sentence_chunks)
                continue

            # 如果添加当前句子会超过块大小，先保存当前块
            if current_size + sentence_size > self.chunk_size:
                chunks.append("".join(current_chunk))

                # 考虑重叠，保留部分句子
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(s) for s in current_chunk)

            # 添加当前句子到块
            current_chunk.append(sentence)
            current_size += sentence_size

        # 添加最后一个块
        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """按段落边界分块"""
        # 使用双换行作为段落分隔符
        paragraphs = re.split(r'\n\s*\n', text)

        # 过滤空段落
        paragraphs = [p for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            paragraph_size = len(paragraph)

            # 如果单个段落超过块大小，按句子分块处理
            if paragraph_size > self.chunk_size:
                # 先处理当前累积的块
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # 处理大段落
                paragraph_chunks = self._chunk_by_sentence(paragraph)
                chunks.extend(paragraph_chunks)
                continue

            # 如果添加当前段落会超过块大小，先保存当前块
            if current_size + paragraph_size > self.chunk_size:
                chunks.append("\n\n".join(current_chunk))

                # 对于段落，我们不考虑重叠
                current_chunk = []
                current_size = 0

            # 添加当前段落到块
            current_chunk.append(paragraph)
            current_size += paragraph_size

        # 添加最后一个块
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        将文本分块

        Args:
            text: 输入文本

        Returns:
            包含文本块及其元数据的字典列表
        """
        try:
            # 获取分块方法
            chunk_method = self.chunking_methods.get(
                self.chunking_method, self._chunk_by_text
            )

            # 执行分块
            text_chunks = chunk_method(text)

            # 为每个块创建元数据
            result = []
            for i, chunk in enumerate(text_chunks):
                # 计算块的哈希值作为ID
                chunk_id = hashlib.md5(chunk.encode('utf-8')).hexdigest()

                result.append({
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "text": chunk,
                    "char_count": len(chunk),
                    "chunk_method": self.chunking_method
                })

            logger.info(f"文本分块完成，共 {len(result)} 个块")
            return result

        except Exception as e:
            logger.error(f"文本分块时出错: {str(e)}")
            raise