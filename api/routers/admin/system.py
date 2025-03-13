# api/routers/admin/system.py
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from api.schemas.common import ErrorResponse
from api.dependencies.services import get_admin_service
from api.dependencies.auth import get_admin_user
from services.admin_service import AdminService

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/status",
    response_model=Dict[str, Any]
)
async def get_system_status(
        admin_user: Dict[str, Any] = Depends(get_admin_user),
        admin_service: AdminService = Depends(get_admin_service)
):
    """获取系统状态信息（需要管理员权限）"""
    try:
        status = await admin_service.get_system_status()
        return status
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


@router.get(
    "/statistics",
    response_model=Dict[str, Any],
    responses={500: {"model": ErrorResponse}}
)
async def get_system_statistics(
        period: str = Query("day", description="统计周期: day, week, month, all"),
        admin_user: Dict[str, Any] = Depends(get_admin_user),
        admin_service: AdminService = Depends(get_admin_service)
):
    """获取系统使用统计（需要管理员权限）"""
    try:
        if period not in ["day", "week", "month", "all"]:
            raise HTTPException(status_code=400, detail=f"不支持的统计周期: {period}")

        statistics = await admin_service.get_system_statistics(period)
        return statistics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取系统统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统统计信息失败: {str(e)}")


@router.get(
    "/logs",
    response_model=Dict[str, Any],
    responses={500: {"model": ErrorResponse}}
)
async def get_system_logs(
        log_type: Optional[str] = Query("all", description="日志类型: all, error, info"),
        limit: int = Query(100, description="返回的日志条数"),
        start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
        end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)"),
        admin_user: Dict[str, Any] = Depends(get_admin_user),
        admin_service: AdminService = Depends(get_admin_service)
):
    """获取系统日志（需要管理员权限）"""
    try:
        if log_type not in ["all", "error", "info", "warning", "debug"]:
            raise HTTPException(status_code=400, detail=f"不支持的日志类型: {log_type}")

        logs = await admin_service.get_system_logs(
            log_type=log_type,
            limit=limit,
            start_date=start_date,
            end_date=end_date
        )

        return {
            "logs": logs,
            "total": len(logs),
            "params": {
                "log_type": log_type,
                "limit": limit,
                "start_date": start_date,
                "end_date": end_date
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取系统日志失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统日志失败: {str(e)}")


@router.post(
    "/config",
    response_model=Dict[str, Any],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def update_system_config(
        config_updates: Dict[str, Any] = Body(..., description="系统配置更新"),
        admin_user: Dict[str, Any] = Depends(get_admin_user),
        admin_service: AdminService = Depends(get_admin_service)
):
    """更新系统配置（需要管理员权限）"""
    try:
        updated_config = await admin_service.update_system_config(config_updates)
        return {
            "success": True,
            "message": "系统配置已更新",
            "config": updated_config
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"更新系统配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新系统配置失败: {str(e)}")