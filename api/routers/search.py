# api/routers/search.py
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Dict, Any, List, Optional

from api.dependencies.services import get_search_service
from services.search_service import SearchService
from api.schemas.search import SearchQuery, SearchResponse

router = APIRouter()


@router.post("/", response_model=SearchResponse)
async def search(
        query: SearchQuery,
        search_service: SearchService = Depends(get_search_service)
):
    """执行搜索"""
    result = search_service.search(
        query=query.query,
        top_k=query.limit,
        filters=query.filters,
        search_type=query.search_type
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/stats")
async def get_search_stats(
        search_service: SearchService = Depends(get_search_service)
):
    """获取搜索服务统计信息"""
    return search_service.get_stats()