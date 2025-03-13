# api/schemas/user.py
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime


class UserBase(BaseModel):
    """用户基础信息模型"""
    username: str = Field(..., min_length=3, max_length=32, description="用户名")
    email: EmailStr = Field(..., description="电子邮箱")
    full_name: Optional[str] = Field(None, description="全名")


class UserCreate(UserBase):
    """用户创建模型"""
    password: str = Field(..., min_length=8, description="用户密码")
    role: str = Field("user", description="用户角色（user 或 admin）")


class UserResponse(UserBase):
    """用户信息响应模型"""
    role: str = Field(..., description="用户角色")
    disabled: bool = Field(False, description="是否禁用")
    created_at: str = Field(..., description="创建时间")
    last_login: Optional[str] = Field(None, description="最后登录时间")
    preferences: Optional[Dict[str, Any]] = Field(None, description="用户偏好设置")


class UserUpdate(BaseModel):
    """用户更新模型"""
    email: Optional[EmailStr] = Field(None, description="电子邮箱")
    full_name: Optional[str] = Field(None, description="全名")
    password: Optional[str] = Field(None, min_length=8, description="用户密码")
    role: Optional[str] = Field(None, description="用户角色")
    disabled: Optional[bool] = Field(None, description="是否禁用")
    preferences: Optional[Dict[str, Any]] = Field(None, description="用户偏好设置")


class UserList(BaseModel):
    """用户列表响应模型"""
    users: List[UserResponse] = Field(..., description="用户列表")
    total: int = Field(..., description="总用户数")
    skip: int = Field(0, description="跳过的用户数")
    limit: int = Field(100, description="限制返回的用户数")


class Token(BaseModel):
    """令牌响应模型"""
    access_token: str = Field(..., description="访问令牌")
    token_type: str = Field("bearer", description="令牌类型")
    expires_in: Optional[int] = Field(None, description="有效期（秒）")
    user_info: Optional[UserResponse] = Field(None, description="用户基本信息")


class TokenData(BaseModel):
    """令牌数据模型"""
    username: Optional[str] = Field(None, description="用户名")
    exp: Optional[datetime] = Field(None, description="过期时间")


class UserPreferences(BaseModel):
    """用户偏好设置模型"""
    theme: Optional[str] = Field("light", description="界面主题（light 或 dark）")
    language: Optional[str] = Field("zh_CN", description="界面语言")
    results_per_page: Optional[int] = Field(10, description="每页显示结果数")
    default_search_mode: Optional[str] = Field("hybrid", description="默认搜索模式")
    notification_enabled: Optional[bool] = Field(True, description="是否启用通知")
    custom_settings: Optional[Dict[str, Any]] = Field(None, description="自定义设置")


class ChangePasswordRequest(BaseModel):
    """修改密码请求模型"""
    current_password: str = Field(..., description="当前密码")
    new_password: str = Field(..., min_length=8, description="新密码")
    confirm_password: str = Field(..., description="确认新密码")