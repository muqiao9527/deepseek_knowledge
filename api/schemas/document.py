# api/schemas/document.py
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional
from datetime import datetime


class FileInfo(BaseModel):
    """文件信息模型"""
    filename: str = Field(..., description="文件名")
    filePath: str = Field(..., description="文件路径")
    content_type: str = Field(..., description="文件类型")
    file_size: int = Field(..., description="文件大小（字节）")


class DocumentMetadata(BaseModel):
    """文档元数据模型"""
    title: str = Field(..., description="文档标题")
    description: Optional[str] = Field(None, description="文档描述")
    tags: Optional[List[str]] = Field(default_factory=list, description="文档标签")
    source: Optional[str] = Field(None, description="文档来源")
    author: Optional[str] = Field(None, description="文档作者")
    created_at: Optional[str] = Field(None, description="文档创建时间")
    updated_at: Optional[str] = Field(None, description="文档更新时间")
    language: Optional[str] = Field(None, description="文档语言")
    version: Optional[str] = Field(None, description="文档版本")
    classification: Optional[str] = Field(None, description="文档分类")
    importance: Optional[int] = Field(None, ge=1, le=5, description="文档重要性（1-5）")
    expiration_date: Optional[str] = Field(None, description="文档过期时间")
    custom_properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="自定义属性")


class DocumentRequest(BaseModel):
    """文档上传/创建请求模型"""
    file_info: FileInfo = Field(..., description="文件信息")
    metadata: DocumentMetadata = Field(..., description="文档元数据")


class DocumentResponseFileInfo(BaseModel):
    """文档响应中的文件信息"""
    file_size: int = Field(..., description="文件大小（字节）")
    content_type: str = Field(..., description="文件类型")
    file_path: str = Field(..., description="文件路径")


class DocumentResponse(BaseModel):
    """文档响应模型"""
    document_id: str = Field(..., description="文档ID")
    file_name: str = Field(..., description="文件名")
    file_info: DocumentResponseFileInfo = Field(..., description="文件信息")
    metadata: Dict[str, Any] = Field(..., description="文档元数据")
    indexed: bool = Field(False, description="是否已索引")
    upload_time: str = Field(..., description="上传时间")
    status: str = Field("success", description="状态")
    message: Optional[str] = Field(None, description="消息")


class DocumentListResponse(BaseModel):
    """文档列表响应模型"""
    documents: List[DocumentResponse] = Field(..., description="文档列表")
    total: int = Field(..., description="总文档数")
    skip: int = Field(0, description="跳过的文档数")
    limit: int = Field(100, description="限制返回的文档数")
    filters: Optional[Dict[str, Any]] = Field(None, description="应用的过滤条件")


class DocumentMetadataUpdate(BaseModel):
    """文档元数据更新模型"""
    title: Optional[str] = Field(None, description="文档标题")
    description: Optional[str] = Field(None, description="文档描述")
    tags: Optional[List[str]] = Field(None, description="文档标签")
    source: Optional[str] = Field(None, description="文档来源")
    author: Optional[str] = Field(None, description="文档作者")
    language: Optional[str] = Field(None, description="文档语言")
    version: Optional[str] = Field(None, description="文档版本")
    classification: Optional[str] = Field(None, description="文档分类")
    importance: Optional[int] = Field(None, ge=1, le=5, description="文档重要性（1-5）")
    expiration_date: Optional[str] = Field(None, description="文档过期时间")
    custom_properties: Optional[Dict[str, Any]] = Field(None, description="自定义属性")


class DocumentUploadResponse(BaseModel):
    """文档上传响应模型（兼容旧版本）"""
    document_id: str = Field(..., description="文档ID")
    file_name: str = Field(..., description="文件名")
    indexed: bool = Field(False, description="是否已索引")
    file_size: int = Field(..., description="文件大小（字节）")
    file_type: str = Field(..., description="文件类型")
    collection: Optional[str] = Field(None, description="文档集合")
    message: str = Field("文档上传成功", description="消息")

class FilePathRequest(BaseModel):
    """表示服务器上文件绝对路径的请求模型"""
    file_path: str = Field(..., description="服务器上文件的绝对路径")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="可选的附加元数据")

class ProcessedDocumentResponse(BaseModel):
    """文档处理响应模型"""
    content: str = Field(..., description="提取的文档内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    chunks: Optional[List[Dict[str, Any]]] = Field(default=None, description="文档分块")
    doc_id: Optional[str] = Field(default=None, description="文档ID")
    status: str = Field(default="success", description="处理状态")
    error: Optional[str] = Field(default=None, description="错误信息，如果处理失败")
    index_result: Optional[Dict[str, Any]] = Field(default=None, description="索引结果，如果文件被添加到知识库")
