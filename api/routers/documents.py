# api/routers/documents.py
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, Body, Path
from typing import List, Dict, Any, Optional
import logging

from api.schemas.document import (
    DocumentRequest,
    DocumentResponse,
    DocumentListResponse,
    DocumentMetadataUpdate, FilePathRequest, ProcessedDocumentResponse
)
from api.schemas.common import ErrorResponse, SuccessResponse
from api.dependencies.services import get_document_service
from services.document_service import DocumentService
import os

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}
)
async def upload_document(
        file: UploadFile = File(...),
        document_request: DocumentRequest = Body(...),
        index_immediately: bool = Query(False, description="是否立即索引"),
        document_service: DocumentService = Depends(get_document_service)
):
    """
    上传并处理文档

    - 支持PDF、DOCX、XLSX、TXT等格式
    - 接收文件信息和元数据
    - 可选择是否立即创建索引
    """
    try:
        result = await document_service.upload_document(
            file=file,
            document_request=document_request,
            index_immediately=index_immediately
        )
        return result
    except Exception as e:
        logger.error(f"文档上传处理失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"文档处理失败: {str(e)}")


@router.post(
    "/create",
    response_model=DocumentResponse,
    responses={400: {"model": ErrorResponse}}
)
async def create_document(
        document_request: DocumentRequest = Body(...),
        file_content: Optional[str] = Body(None, description="文档内容，用于不上传文件的情况"),
        document_service: DocumentService = Depends(get_document_service)
):
    """
    创建文档记录（无需上传文件或使用已有文件）

    - 可以基于已存在的文件路径创建
    - 可以直接提供文本内容创建
    """
    try:
        result = await document_service.create_document(
            document_request=document_request,
            file_content=file_content
        )
        return result
    except Exception as e:
        logger.error(f"创建文档记录失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"创建文档记录失败: {str(e)}")


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse}}
)
async def get_document(
        document_id: str = Path(..., description="文档ID"),
        document_service: DocumentService = Depends(get_document_service)
):
    """获取单个文档的详细信息"""
    document = await document_service.get_document(document_id)

    if not document:
        raise HTTPException(status_code=404, detail=f"文档ID {document_id} 不存在")

    return document


@router.get(
    "/",
    response_model=DocumentListResponse
)
async def list_documents(
        tags: Optional[List[str]] = Query(None, description="按标签筛选"),
        source: Optional[str] = Query(None, description="按来源筛选"),
        file_type: Optional[str] = Query(None, description="按文件类型筛选"),
        indexed: Optional[bool] = Query(None, description="按索引状态筛选"),
        skip: int = Query(0, description="分页起始位置"),
        limit: int = Query(100, description="每页数量"),
        document_service: DocumentService = Depends(get_document_service)
):
    """获取文档列表，支持分页和筛选"""
    filters = {}

    if tags:
        filters["tags"] = tags

    if source:
        filters["source"] = source

    if file_type:
        filters["file_type"] = file_type.lower()

    if indexed is not None:
        filters["indexed"] = indexed

    result = await document_service.list_documents(
        filters=filters,
        skip=skip,
        limit=limit
    )

    return result


@router.patch(
    "/{document_id}/metadata",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse}}
)
async def update_document_metadata(
        document_id: str = Path(..., description="文档ID"),
        metadata: Dict[str, Any] = Body(..., description="要更新的元数据"),
        document_service: DocumentService = Depends(get_document_service)
):
    """更新文档元数据"""
    try:
        result = await document_service.update_document_metadata(
            document_id=document_id,
            metadata=metadata
        )

        if not result:
            raise HTTPException(status_code=404, detail=f"文档ID {document_id} 不存在")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新文档元数据失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"更新文档元数据失败: {str(e)}")


@router.post(
    "/{document_id}/index",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}}
)
async def index_document(
        document_id: str = Path(..., description="文档ID"),
        document_service: DocumentService = Depends(get_document_service)
):
    """为文档创建搜索索引"""
    try:
        result = await document_service.index_document(document_id)

        if not result["success"]:
            if result.get("not_found"):
                raise HTTPException(status_code=404, detail=f"文档ID {document_id} 不存在")
            raise HTTPException(status_code=400, detail=result.get("message", "索引失败"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档索引失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"文档索引失败: {str(e)}")


@router.delete(
    "/{document_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}}
)
async def delete_document(
        document_id: str = Path(..., description="文档ID"),
        delete_file: bool = Query(False, description="是否删除文件"),
        document_service: DocumentService = Depends(get_document_service)
):
    """删除文档"""
    try:
        result = await document_service.delete_document(
            document_id=document_id,
            delete_file=delete_file
        )

        if not result["success"]:
            if result.get("not_found"):
                raise HTTPException(status_code=404, detail=f"文档ID {document_id} 不存在")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"删除文档失败: {str(e)}")


@router.get(
    "/{document_id}/content",
    responses={404: {"model": ErrorResponse}}
)
async def get_document_content(
        document_id: str = Path(..., description="文档ID"),
        document_service: DocumentService = Depends(get_document_service)
):
    """获取文档的文本内容"""
    content = await document_service.get_document_content(document_id)

    if content is None:
        raise HTTPException(status_code=404, detail=f"文档ID {document_id} 不存在")

    return {"content": content}


@router.post(
    "/batch-index",
    response_model=SuccessResponse,
    responses={400: {"model": ErrorResponse}}
)
async def batch_index_documents(
        document_ids: List[str] = Body(..., description="文档ID列表"),
        document_service: DocumentService = Depends(get_document_service)
):
    """批量索引多个文档"""
    try:
        result = await document_service.batch_index_documents(document_ids)
        return result
    except Exception as e:
        logger.error(f"批量索引文档失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"批量索引文档失败: {str(e)}")


@router.post(
    "/batch-tag",
    response_model=SuccessResponse,
    responses={400: {"model": ErrorResponse}}
)
async def batch_update_tags(
        document_ids: List[str] = Body(..., description="文档ID列表"),
        tags: List[str] = Body(..., description="要添加的标签"),
        operation: str = Body("add", description="操作类型：add(添加), remove(移除), replace(替换)"),
        document_service: DocumentService = Depends(get_document_service)
):
    """批量更新多个文档的标签"""
    try:
        if operation not in ["add", "remove", "replace"]:
            raise HTTPException(status_code=400, detail=f"不支持的操作类型: {operation}")

        result = await document_service.batch_update_tags(
            document_ids=document_ids,
            tags=tags,
            operation=operation
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量更新标签失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"批量更新标签失败: {str(e)}")


@router.post("/process-file-path", response_model=ProcessedDocumentResponse)
async def process_file_path(
        request: FilePathRequest,
        add_to_kb: bool = Query(False, description="是否将文件添加到知识库"),
        document_service: DocumentService = Depends(get_document_service)
):
    """
    处理服务器上已存在的文件（通过绝对路径）

    - **file_path**: 服务器上文件的绝对路径
    - **metadata**: 可选的附加元数据
    - **add_to_kb**: 设置为true将处理后的文件添加到知识库中（向量化和索引）
    """
    result = document_service.process_file_path(
        file_path=request.file_path,
        additional_metadata=request.metadata,
        add_to_knowledge_base=add_to_kb
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error", "处理文件失败"))

    return result

@router.post(
    "/upload/v2",
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}
)
async def upload_document_v2(
        request: Dict[str, Any] = Body(...),
        index_immediately: bool = Query(False, description="是否立即索引"),
        document_service: DocumentService = Depends(get_document_service)
):
    """
    新版本的文档上传接口 - 适配特定前端需求格式

    - 接收客户端指定格式的请求体
    - 返回客户端要求的响应格式
    """
    try:
        # 从请求中提取必要信息
        if "file_info" not in request or "metadata" not in request:
            raise HTTPException(status_code=400, detail="请求格式错误，缺少file_info或metadata字段")

        file_info = request.get("file_info", {})
        metadata = request.get("metadata", {})

        # 验证必要的字段
        required_fields = ["filename", "content_type", "filePath", "file_size"]
        for field in required_fields:
            if field not in file_info:
                raise HTTPException(status_code=400, detail=f"file_info缺少必要字段: {field}")

        # 验证文件路径是否存在
        file_path = file_info.get("filePath")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail=f"文件路径不存在: {file_path}")

        # 处理文件
        result = document_service.process_file_path(
            file_path=file_path,
            additional_metadata=metadata,
            add_to_knowledge_base=index_immediately
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "处理文件失败"))

        # 生成唯一ID (如果处理结果中没有)
        document_id = result.get("doc_id") or f"doc{uuid4().hex[:6]}"

        # 构建客户端期望的响应格式
        response = {
            "document_id": document_id,
            "status": "indexed" if index_immediately else "processed",
            "processing_details": {
                "processing_time": result.get("processing_time", 0),
                "chunk_count": len(result.get("chunks", [])),
                "processor_type": result.get("processor_type", "DocumentProcessor")
            },
            "extracted_metadata": {
                **result.get("metadata", {})
            }
        }

        # 如果有索引结果，添加到响应中
        if result.get("index_result"):
            response["index_result"] = result.get("index_result")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档上传处理失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"文档处理失败: {str(e)}")

