# api/routers/users.py
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, Body, Path, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext

from api.dependencies.auth import get_admin_user
from api.schemas.common import ErrorResponse, SuccessResponse
from api.schemas.user import (
    UserCreate,
    UserResponse,
    UserUpdate,
    UserList,
    Token,
    TokenData
)
from config.settings import SECRET_KEY

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter()

# 密码处理
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT配置
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 模拟用户数据存储
# 实际应用中应使用数据库
# 在fake_users_db初始化中
fake_users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "管理员",
        "hashed_password": pwd_context.hash("password"),
        "disabled": False,
        "role": "admin",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
}

# 在create_user函数中
async def create_user(
        user: UserCreate = Body(...),
        admin_user: dict = Depends(get_admin_user)
):
    """创建新用户（需要管理员权限）"""
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = get_password_hash(user.password)
    user_data = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "disabled": False,
        "role": user.role,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    fake_users_db[user.username] = user_data

    # 返回新用户信息（不包含密码）
    return {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "disabled": False,
        "created_at": user_data["created_at"]
    }


def verify_password(plain_password, hashed_password):
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """获取密码哈希值"""
    return pwd_context.hash(password)


def get_user(db, username: str):
    """根据用户名获取用户信息"""
    if username in db:
        user_dict = db[username]
        return user_dict
    return None


def authenticate_user(db, username: str, password: str):
    """验证用户凭据"""
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建JWT令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    """获取当前活跃用户"""
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_admin_user(current_user: dict = Depends(get_current_active_user)):
    """确保用户是管理员"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """用户登录并获取访问令牌"""
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post(
    "/",
    response_model=UserResponse,
    responses={400: {"model": ErrorResponse}}
)
async def create_user(
        user: UserCreate = Body(...),
        admin_user: dict = Depends(get_admin_user)
):
    """创建新用户（需要管理员权限）"""
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = get_password_hash(user.password)
    user_data = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "disabled": False,
        "role": user.role,
        "created_at": datetime.now().isoformat()
    }

    fake_users_db[user.username] = user_data

    # 返回新用户信息（不包含密码）
    return {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "disabled": False,
        "created_at": user_data["created_at"]
    }


@router.get(
    "/me",
    response_model=UserResponse
)
async def read_users_me(
        current_user: dict = Depends(get_current_active_user)
):
    """获取当前用户信息"""
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "full_name": current_user["full_name"],
        "role": current_user["role"],
        "disabled": current_user["disabled"],
        "created_at": current_user["created_at"]
    }


@router.get(
    "/{username}",
    response_model=UserResponse,
    responses={404: {"model": ErrorResponse}}
)
async def read_user(
        username: str = Path(...),
        current_user: dict = Depends(get_current_active_user)
):
    """获取用户信息"""
    # 普通用户只能查看自己的信息
    if current_user["role"] != "admin" and current_user["username"] != username:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    if username not in fake_users_db:
        raise HTTPException(status_code=404, detail="User not found")

    user = fake_users_db[username]
    return {
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"],
        "disabled": user["disabled"],
        "created_at": user["created_at"]
    }


# 修复update_user函数中的dict方法弃用
async def update_user(
        username: str = Path(...),
        user_update: UserUpdate = Body(...),
        current_user: dict = Depends(get_current_active_user)
):
    """更新用户信息"""
    # 普通用户只能更新自己的信息，管理员可以更新任何人
    if current_user["role"] != "admin" and current_user["username"] != username:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    if username not in fake_users_db:
        raise HTTPException(status_code=404, detail="User not found")

    user = fake_users_db[username]

    # 更新用户数据
    update_data = user_update.model_dump(exclude_unset=True)

    # 如果更新包含密码，则需要哈希处理
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))

    # 更新用户记录
    fake_users_db[username] = {**user, **update_data}

    updated_user = fake_users_db[username]
    return {
        "username": updated_user["username"],
        "email": updated_user["email"],
        "full_name": updated_user["full_name"],
        "role": updated_user["role"],
        "disabled": updated_user["disabled"],
        "created_at": updated_user["created_at"]
    }


@router.delete(
    "/{username}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}}
)
async def delete_user(
        username: str = Path(...),
        admin_user: dict = Depends(get_admin_user)
):
    """删除用户（需要管理员权限）"""
    if username not in fake_users_db:
        raise HTTPException(status_code=404, detail="User not found")

    # 防止删除最后一个管理员账户
    if fake_users_db[username]["role"] == "admin":
        admin_count = sum(1 for user in fake_users_db.values() if user["role"] == "admin")
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete the last admin user")

    del fake_users_db[username]

    return {
        "success": True,
        "message": f"User {username} has been deleted"
    }


@router.get(
    "/",
    response_model=UserList,
    dependencies=[Depends(get_admin_user)]
)
async def list_users(
        skip: int = Query(0),
        limit: int = Query(100)
):
    """获取用户列表（需要管理员权限）"""
    users = list(fake_users_db.values())

    # 应用分页
    paginated_users = users[skip:skip + limit]

    # 转换为响应格式（不包含密码）
    user_list = []
    for user in paginated_users:
        user_list.append({
            "username": user["username"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
            "disabled": user["disabled"],
            "created_at": user["created_at"]
        })

    return {
        "users": user_list,
        "total": len(users),
        "skip": skip,
        "limit": limit
    }


@router.post(
    "/change-password",
    response_model=SuccessResponse,
    responses={400: {"model": ErrorResponse}}
)
async def change_password(
        old_password: str = Body(...),
        new_password: str = Body(...),
        current_user: dict = Depends(get_current_active_user)
):
    """更改当前用户密码"""
    username = current_user["username"]
    user = fake_users_db[username]

    # 验证旧密码
    if not verify_password(old_password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect password")

    # 更新密码
    fake_users_db[username]["hashed_password"] = get_password_hash(new_password)

    return {
        "success": True,
        "message": "Password updated successfully"
    }