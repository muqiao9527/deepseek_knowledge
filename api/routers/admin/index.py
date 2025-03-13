# api/routers/admin/index.py
from fastapi import APIRouter, Depends, HTTPException, Body, Path
from typing import List, Dict, Any
import logging

from api.schemas.common import SuccessResponse, ErrorResponse
from api.dependencies.services import get_admin_service
from api.dependencies.auth import get_admin_user
from services.admin_service import AdminService

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/rebuild",
    response_model=SuccessResponse,
    responses={500: {"model": ErrorResponse}}
)
async def rebuild_search_index(
        full_rebuild: bool = Body(False, description="是否完全重建索引（删除现有索引）"),
        admin_user: Dict[str, Any] = Depends(get_admin_user),
        admin_service: AdminService = Depends(get_admin_service)
):
    """重建搜索索引（需要管理员权限）"""
    try:
        result = await admin_service.rebuild_search_index(full_rebuild)
        return result
    except Exception as e:
        logger.error(f"重建索引失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重建索引失败: {str(e)}")


@router.post(
    "/document/{document_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def reindex_single_document(
        document_id: str = Path(..., description="文档ID"),
        force: bool = Body(False, description="强制重新索引，即使文档已索引"),
        admin_user: Dict[str, Any] = Depends(get_admin_user),
        admin_service: AdminService = Depends(get_admin_service)
):
    """重新索引单个文档（需要管理员权限）"""
    try:
        result = await admin_service.reindex_document(document_id, force)

        if not result["success"]:
            if result.get("not_found"):
                raise HTTPException(status_code=404, detail=f"文档ID {document_id} 不存在")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重新索引文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重新索引文档失败: {str(e)}")


@router.post(
    "/optimize",
    response_model=SuccessResponse,
    responses={500: {"model": ErrorResponse}}
)
async def optimize_index(
        admin_user: Dict[str, Any] = Depends(get_admin_user),
        admin_service: AdminService = Depends(get_admin_service)
):
    """优化搜索索引（需要管理员权限）"""
    try:
        result = await admin_service.optimize_index()
        return result
    except Exception as e:
        logger.error(f"优化索引失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"优化索引失败: {str(e)}")


@router.post(
    "/cleanup-orphaned",
    response_model=SuccessResponse,
    responses={500: {"model": ErrorResponse}}
)
async def cleanup_orphaned_indices(
        dry_run: bool = Body(True, description="是否仅模拟运行"),
        admin_user: Dict[str, Any] = Depends(get_admin_user),
        admin_service: AdminService = Depends(get_admin_service)
):
    """清理孤立索引（需要管理员权限）"""
    try:
        result = await admin_service.cleanup_orphaned_indices(dry_run)
        return result
    except Exception as e:
        logger.error(f"清理孤立索引失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理孤立索引失败: {str(e)}")