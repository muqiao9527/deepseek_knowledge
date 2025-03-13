# api/routers/admin/maintenance.py
from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any
import logging

from api.schemas.common import SuccessResponse, ErrorResponse
from api.dependencies.services import get_admin_service
from api.dependencies.auth import get_admin_user
from services.admin_service import AdminService

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/backup",
    response_model=SuccessResponse,
    responses={500: {"model": ErrorResponse}}
)
async def backup_system(
    include_vector_store: bool = Body(True, description="是否包含向量存储"),
    include_documents: bool = Body(True, description="是否包含文档内容"),
    backup_path: str = Body(None, description="备份保存路径"),
    admin_user: Dict[str, Any] = Depends(get_admin_user),
    admin_service: AdminService = Depends(get_admin_service)
):
    """备份系统数据（需要管理员权限）"""
    try:
        result = await admin_service.backup_system(
            include_vector_store=include_vector_store,
            include_documents=include_documents,
            backup_path=backup_path
        )
        return result
    except Exception as e:
        logger.error(f"系统备份失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"系统备份失败: {str(e)}")


@router.post(
    "/restore",
    response_model=SuccessResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def restore_system(
    backup_path: str = Body(..., description="备份路径"),
    restore_vector_store: bool = Body(True, description="是否恢复向量存储"),
    restore_documents: bool = Body(True, description="是否恢复文档内容"),
    admin_user: Dict[str, Any] = Depends(get_admin_user),
    admin_service: AdminService = Depends(get_admin_service)
):
    """从备份恢复系统（需要管理员权限）"""
    try:
        result = await admin_service.restore_system(
            backup_path=backup_path,
            restore_vector_store=restore_vector_store,
            restore_documents=restore_documents
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"系统恢复失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"系统恢复失败: {str(e)}")


@router.post(
    "/cleanup",
    response_model=SuccessResponse,
    responses={500: {"model": ErrorResponse}}
)
async def cleanup_system(
    clean_unindexed: bool = Body(False, description="清理未索引的文档"),
    clean_missing_files: bool = Body(True, description="清理丢失文件的文档记录"),
    clean_orphaned_indices: bool = Body(True, description="清理孤立的索引"),
    dry_run: bool = Body(True, description="是否仅模拟运行"),
    admin_user: Dict[str, Any] = Depends(get_admin_user),
    admin_service: AdminService = Depends(get_admin_service)
):
    """清理系统数据（需要管理员权限）"""
    try:
        result = await admin_service.cleanup_system(
            clean_unindexed=clean_unindexed,
            clean_missing_files=clean_missing_files,
            clean_orphaned_indices=clean_orphaned_indices,
            dry_run=dry_run
        )
        return result
    except Exception as e:
        logger.error(f"系统清理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"系统清理失败: {str(e)}")