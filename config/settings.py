# config/settings.py
import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./kb_system.db")

# API配置
API_PORT = int(os.getenv("API_PORT", 8000))
API_HOST = os.getenv("API_HOST", "127.0.0.1")
SECRET_KEY = os.getenv("SECRET_KEY", "development-secret-key")

# 向量存储配置
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", None)  # 本地模型路径，None表示使用在线模型
DEVICE = os.getenv("DEVICE", "cpu")

# Deepseek本地模型配置
DEEPSEEK_MODEL_PATH = os.getenv("DEEPSEEK_MODEL_PATH")
DEEPSEEK_DEVICE = os.getenv("DEEPSEEK_DEVICE", "cuda" if os.path.exists("/dev/nvidia0") else "cpu")
DEEPSEEK_MODEL_TYPE = os.getenv("DEEPSEEK_MODEL_TYPE", "deepseek-llm")
DEEPSEEK_LOAD_8BIT=False
DEEPSEEK_LOAD_4BIT=False