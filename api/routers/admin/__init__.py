# api/routers/admin/__init__.py
from fastapi import APIRouter
from .system import router as system_router
from .index import router as index_router
from .maintenance import router as maintenance_router

# 创建主路由器
router = APIRouter()

# 包含子路由器
router.include_router(system_router, prefix="/system", tags=["admin:system"])
router.include_router(index_router, prefix="/index", tags=["admin:index"])
router.include_router(maintenance_router, prefix="/maintenance", tags=["admin:maintenance"])