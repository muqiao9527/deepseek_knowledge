# api/main.py
import logging
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies.services import get_document_service, get_search_service, get_admin_service
from api.routers import documents, search, users, admin, deepseek
from config.settings import API_HOST, API_PORT

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./data/logs/api.log')
    ]
)

logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="知识库系统API",
    description="支持多种文件格式的知识库管理系统API",
    version="0.1.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求处理时间中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"全局异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {str(exc)}"}
    )


# 注册路由
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

app.include_router(deepseek.router, prefix="/api/deepseek", tags=["deepseek"])


@app.get("/", tags=["root"])
async def read_root():
    """API根路径"""
    return {
        "message": "欢迎使用知识库系统API",
        "version": "0.1.0",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["system"])
async def health_check():
    """健康检查接口"""
    # 检查服务依赖
    document_service = get_document_service()
    search_service = get_search_service()
    admin_service = get_admin_service()

    return {
        "status": "ok",
        "version": "0.1.0",
        "timestamp": time.time(),
        "services": {
            "document_service": "ok",
            "search_service": "ok",
            "admin_service": "ok"
        }
    }


if __name__ == "__main__":
    logger.info(f"启动API服务: {API_HOST}:{API_PORT}")
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)