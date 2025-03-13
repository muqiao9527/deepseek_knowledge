# api/dependencies/auth.py
from datetime import datetime, timezone
from typing import Dict, Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import PyJWTError

from config.settings import SECRET_KEY

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 模拟的用户数据存储
# 实际应用中应使用数据库
fake_users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "管理员",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
        "disabled": False,
        "role": "admin",
        "created_at": "2024-01-01T00:00:00"
    }
}

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    从JWT令牌获取当前用户

    Args:
        token: JWT访问令牌

    Returns:
        用户信息

    Raises:
        HTTPException: 如果令牌无效或已过期
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception

        token_exp = payload.get("exp")
        if datetime.now(timezone.utc).timestamp() > token_exp:
            raise credentials_exception

    except PyJWTError:
        raise credentials_exception

    user = fake_users_db.get(username)
    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    获取当前活跃用户

    Args:
        current_user: 当前用户信息

    Returns:
        活跃用户信息

    Raises:
        HTTPException: 如果用户被禁用
    """
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="用户已禁用")
    return current_user


async def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_active_user)) -> Dict[str, Any]:
    """
    验证当前用户是否为管理员

    Args:
        current_user: 当前用户信息

    Returns:
        管理员用户信息

    Raises:
        HTTPException: 如果用户不是管理员
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足，需要管理员权限"
        )
    return current_user