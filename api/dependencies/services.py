# api/dependencies/services.py

from config.settings import (
    VECTOR_STORE_TYPE, VECTOR_STORE_PATH,
    EMBEDDING_MODEL, DEVICE, EMBEDDING_MODEL_PATH, DEEPSEEK_MODEL_PATH, DEEPSEEK_DEVICE, DEEPSEEK_MODEL_TYPE
)
from services.admin_service import AdminService
from services.deepseek_service import DeepseekService
from services.document_service import DocumentService
from services.search_service import SearchService

# 服务实例缓存
_services_cache = {}


def get_document_service():
    """获取文档服务实例"""
    # 不传递config参数，使用默认值
    return DocumentService()


def get_search_service():
    """获取搜索服务实例"""
    # 构建必要的config
    config = {
        "model_name": EMBEDDING_MODEL,
        "model_path": EMBEDDING_MODEL_PATH,
        "device": DEVICE,
        "store_type": VECTOR_STORE_TYPE,
        "index_path": VECTOR_STORE_PATH
    }

    if "search_service" not in _services_cache:
        _services_cache["search_service"] = SearchService(config)

    return _services_cache["search_service"]


def get_admin_service() -> AdminService:
    """获取管理服务实例"""
    if "admin_service" not in _services_cache:
        config = {
            "document": {
                "upload_dir": "./uploads"
            },
            "search": {
                "model_name": EMBEDDING_MODEL,
                "device": DEVICE,
                "store_type": VECTOR_STORE_TYPE,
                "index_path": VECTOR_STORE_PATH
            }
        }
        _services_cache["admin_service"] = AdminService(config)

    return _services_cache["admin_service"]

def get_deepseek_service() -> DeepseekService:
    """获取Deepseek大模型服务实例"""
    if "deepseek_service" not in _services_cache:
        config = {
            "model_path": DEEPSEEK_MODEL_PATH,
            "device": DEEPSEEK_DEVICE,
            "model_type": DEEPSEEK_MODEL_TYPE
        }
        _services_cache["deepseek_service"] = DeepseekService(config)

    return _services_cache["deepseek_service"]