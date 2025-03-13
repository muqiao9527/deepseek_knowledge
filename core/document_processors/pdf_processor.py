# core/document_processors/pdf_processor.py
import os
from typing import List, Dict, Any
import logging
from datetime import datetime
from pypdf import PdfReader
from .base_processor import BaseDocumentProcessor

logger = logging.getLogger(__name__)


class PDFProcessor(BaseDocumentProcessor):
    """PDF文档处理器，用于处理PDF文件"""

    def __init__(self, config=None):
        """
        初始化PDF处理器

        Args:
            config: 可选配置，如OCR设置等
        """
        super().__init__(config or {})
        self.use_ocr = config.get("use_ocr", False)
        self.ocr_language = config.get("ocr_language", "chi_sim+eng")
        self.ocr_config = config.get("ocr_config", "")

        if self.use_ocr:
            logger.info(f"PDF处理器启用OCR，语言: {self.ocr_language}")

    def extract_text(self, file_path: str) -> str:
        """
        从PDF文档中提取文本，支持OCR

        Args:
            file_path: PDF文件路径

        Returns:
            提取的文本内容
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            reader = PdfReader(file_path)
            text_content = []

            # 遍历每一页提取文本
            for i, page in enumerate(reader.pages):
                logger.debug(f"处理PDF第 {i + 1}/{len(reader.pages)} 页")
                page_text = page.extract_text()

                # 如果页面没有文本且启用了OCR，则使用OCR
                if not page_text and self.use_ocr:
                    logger.debug(f"第 {i + 1} 页没有可提取文本，尝试OCR")
                    try:
                        # 注意：实际项目中需要使用pdf2image等库将PDF页面转为图像
                        # 这里只是示例代码，实际应用中需要替换
                        # from pdf2image import convert_from_path
                        # images = convert_from_path(file_path, first_page=i+1, last_page=i+1)
                        # if images:
                        #     page_text = pytesseract.image_to_string(
                        #         images[0],
                        #         lang=self.ocr_language,
                        #         config=self.ocr_config
                        #     )

                        page_text = f"[OCR处理的页面 {i + 1}]"  # 占位，实际实现需替换
                        logger.debug(f"OCR提取的文本长度: {len(page_text)}")
                    except Exception as ocr_err:
                        logger.error(f"OCR处理失败: {str(ocr_err)}")
                        page_text = f"[OCR失败的页面 {i + 1}]"

                text_content.append(page_text)

            return "\n\n".join([text for text in text_content if text])

        except Exception as e:
            logger.error(f"提取PDF文本时出错: {str(e)}")
            raise

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        提取PDF文档的元数据

        Args:
            file_path: PDF文件路径

        Returns:
            包含文档元数据的字典
        """
        try:
            reader = PdfReader(file_path)
            metadata = {}

            # 提取文档信息
            if reader.metadata:
                for key, value in reader.metadata.items():
                    # 去掉/前缀并转换为小写
                    clean_key = key[1:].lower() if key.startswith('/') else key.lower()
                    metadata[clean_key] = value

            # 提取更多元数据
            try:
                metadata["is_encrypted"] = reader.is_encrypted

                # 如果有XMP元数据，也可以提取
                # if hasattr(reader, "xmp_metadata") and reader.xmp_metadata:
                #     metadata["xmp"] = reader.xmp_metadata

                # 获取页面尺寸（第一页）
                if len(reader.pages) > 0:
                    first_page = reader.pages[0]
                    if hasattr(first_page, "mediabox"):
                        width = float(first_page.mediabox.width)
                        height = float(first_page.mediabox.height)
                        metadata["page_width"] = width
                        metadata["page_height"] = height
            except Exception as meta_err:
                logger.warning(f"提取额外元数据时出错: {str(meta_err)}")

            # 添加基本信息
            metadata.update({
                "page_count": len(reader.pages),
                "file_size": os.path.getsize(file_path),
                "file_extension": "pdf",
                "file_name": os.path.basename(file_path),
                "last_modified": datetime.fromtimestamp(
                    os.path.getmtime(file_path)
                ).isoformat()
            })

            return metadata

        except Exception as e:
            logger.error(f"提取PDF元数据时出错: {str(e)}")
            raise


    @staticmethod
    def get_pdf_outline(file_path: str) -> List[Dict[str, Any]]:
        """
        提取PDF的目录结构（如果存在）

        Args:
            file_path: PDF文件路径

        Returns:
            目录项列表，每项包含标题和页码
        """
        try:
            reader = PdfReader(file_path)
            outline = []

            # 检查PDF是否有目录
            if hasattr(reader, "outline") and reader.outline:
                # 递归处理目录项
                def process_outline_item(item, level=0):
                    result = []

                    if isinstance(item, list):
                        # 处理子项列表
                        for subitem in item:
                            result.extend(process_outline_item(subitem, level))
                    else:
                        # 处理单个目录项
                        title = item.title if hasattr(item, "title") else "未命名"
                        page = item.page.id if hasattr(item, "page") and item.page else -1

                        result.append({
                            "title": title,
                            "page": page,
                            "level": level
                        })

                        # 处理子项
                        if hasattr(item, "children") and item.children:
                            for child in item.children:
                                result.extend(process_outline_item(child, level + 1))

                    return result

                outline = process_outline_item(reader.outline)

            return outline

        except Exception as e:
            logger.error(f"提取PDF目录时出错: {str(e)}")
            return []