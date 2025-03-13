# api/schemas/search.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SearchQuery(BaseModel):
    """搜索查询参数"""
    query: str = Field(..., description="搜索关键词")
    limit: int = Field(10, description="返回的最大结果数")
    search_type: str = Field("hybrid", description="搜索类型: hybrid, vector, keyword")
    filters: Optional[Dict[str, Any]] = Field(None, description="筛选条件")

class SearchResultItem(BaseModel):
    """搜索结果项"""
    text: str = Field(..., description="文本片段")
    score: float = Field(..., description="相似度得分")
    rank: int = Field(..., description="排名")
    metadata: Dict[str, Any] = Field({}, description="元数据")

class SearchResponse(BaseModel):
    """搜索响应"""
    query: str = Field(..., description="搜索关键词")
    results: List[SearchResultItem] = Field([], description="搜索结果列表")
    total: int = Field(0, description="结果总数")
    search_type: str = Field(..., description="使用的搜索类型")
    error: Optional[str] = Field(None, description="错误信息")