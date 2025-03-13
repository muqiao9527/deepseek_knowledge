# api/schemas/common.py
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List


class ErrorResponse(BaseModel):
    """错误响应模型"""
    detail: str = Field(..., description="错误详情")


class SuccessResponse(BaseModel):
    """成功响应模型"""
    success: bool = Field(True, description="是否成功")
    message: str = Field(..., description="成功消息")
    data: Optional[Dict[str, Any]] = Field(None, description="可选的数据")


class PaginationParams(BaseModel):
    """分页参数模型"""
    skip: int = Field(0, description="跳过的记录数")
    limit: int = Field(100, description="返回的最大记录数")


class PaginatedResponse(BaseModel):
    """分页响应基础模型"""
    total: int = Field(..., description="总记录数")
    skip: int = Field(0, description="跳过的记录数")
    limit: int = Field(100, description="返回的最大记录数")
    has_more: bool = Field(..., description="是否有更多数据")


class SortParams(BaseModel):
    """排序参数模型"""
    field: str = Field(..., description="排序字段")
    direction: str = Field("asc", description="排序方向 (asc 或 desc)")


class FilterCondition(BaseModel):
    """过滤条件模型"""
    field: str = Field(..., description="字段名")
    operator: str = Field(..., description="操作符 (eq, ne, gt, lt, gte, lte, in, contains)")
    value: Any = Field(..., description="比较值")


class FilterParams(BaseModel):
    """过滤参数模型"""
    conditions: List[FilterCondition] = Field(..., description="过滤条件列表")
    logic: str = Field("and", description="逻辑连接符 (and 或 or)")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field("ok", description="服务状态")
    version: str = Field(..., description="API版本")
    uptime: float = Field(..., description="服务运行时间（秒）")
    timestamp: str = Field(..., description="当前时间戳")
    components: Dict[str, Dict[str, str]] = Field(..., description="组件状态")