"""
认证模块 — 注册 / 登录 / 登出 / 密码修改 / 用户管理。

依赖 src/user_db.py 获取数据库会话。
Session State 辅助函数适配 Streamlit 的 st.session_state。
"""

import logging
import re
from datetime import datetime
from enum import Enum

import bcrypt
import streamlit as st

from src.user_db import User, UserModel, get_session, init_db, _model_to_user

logger = logging.getLogger(__name__)


# ============================================================
# 错误码定义
# ============================================================

class AuthError(str, Enum):
    USERNAME_TOO_SHORT = "用户名需 3-20 个字符"
    USERNAME_TOO_LONG = "用户名需 3-20 个字符"
    USERNAME_INVALID = "用户名只能包含字母、数字、下划线"
    USERNAME_TAKEN = "用户名已被占用"
    PASSWORD_TOO_SHORT = "密码至少 6 个字符"
    PASSWORD_MISMATCH = "两次输入的密码不一致"
    INVALID_CREDENTIALS = "用户名或密码错误"
    ACCOUNT_DISABLED = "账户已被禁用"
    OLD_PASSWORD_WRONG = "原密码错误"
    NEW_PASSWORD_SAME = "新密码不能与原密码相同"
    DB_ERROR = "数据库异常"


# ============================================================
# 密码工具
# ============================================================

def hash_password(plain: str) -> str:
    """bcrypt 哈希密码。"""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """验证明文密码与哈希是否匹配。"""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ============================================================
# 认证核心函数
# ============================================================

def register_user(
    username: str, password: str, email: str | None = None
) -> tuple[User | None, str | None]:
    """注册新用户。

    Returns:
        (User, None) 成功
        (None, error_message) 失败
    """
    username = username.strip()

    # 用户名长度 3-20
    if len(username) < 3:
        return None, AuthError.USERNAME_TOO_SHORT
    if len(username) > 20:
        return None, AuthError.USERNAME_TOO_LONG

    # 用户名格式：仅字母、数字、下划线
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return None, AuthError.USERNAME_INVALID

    # 密码长度 ≥ 6
    if len(password) < 6:
        return None, AuthError.PASSWORD_TOO_SHORT

    # 邮箱格式（如果提供）
    if email and not re.match(r"^[^@]+@[^@]+\.[^@]+$", email.strip()):
        return None, AuthError.USERNAME_INVALID

    # 查重
    try:
        with get_session() as session:
            existing = (
                session.query(UserModel)
                .filter(UserModel.username == username)
                .first()
            )
            if existing:
                return None, AuthError.USERNAME_TAKEN

            # 创建用户
            new_user = UserModel(
                username=username,
                password_hash=hash_password(password),
                email=email.strip() if email else None,
                role="user",
                is_active=True,
                created_at=datetime.now(),
            )
            session.add(new_user)
            session.flush()  # 获取自增 id
            return _model_to_user(new_user), None
    except Exception as e:
        logger.exception("注册用户失败")
        return None, AuthError.DB_ERROR


def login_user(username: str, password: str) -> tuple[User | None, str | None]:
    """用户登录。

    Returns:
        (User, None) 成功
        (None, error_message) 失败
    """
    username = username.strip()

    try:
        with get_session() as session:
            user = (
                session.query(UserModel)
                .filter(UserModel.username == username)
                .first()
            )

            if not user:
                return None, AuthError.INVALID_CREDENTIALS

            if not user.is_active:
                return None, AuthError.ACCOUNT_DISABLED

            if not verify_password(password, user.password_hash):
                return None, AuthError.INVALID_CREDENTIALS

            # 更新最后登录时间
            user.last_login = datetime.now()
            session.flush()

            return _model_to_user(user), None
    except Exception as e:
        logger.exception("登录失败")
        return None, AuthError.DB_ERROR


def change_password(
    user_id: int, old_password: str, new_password: str
) -> tuple[bool, str | None]:
    """修改密码。

    Returns:
        (True, None) 成功
        (False, error_message) 失败
    """
    if len(new_password) < 6:
        return False, AuthError.PASSWORD_TOO_SHORT

    try:
        with get_session() as session:
            user = session.query(UserModel).filter(UserModel.id == user_id).first()
            if not user:
                return False, AuthError.DB_ERROR

            if not verify_password(old_password, user.password_hash):
                return False, AuthError.OLD_PASSWORD_WRONG

            if old_password == new_password:
                return False, AuthError.NEW_PASSWORD_SAME

            user.password_hash = hash_password(new_password)
            session.flush()
            return True, None
    except Exception as e:
        logger.exception("修改密码失败")
        return False, AuthError.DB_ERROR


# ============================================================
# 用户查询
# ============================================================

def get_user_by_id(user_id: int) -> User | None:
    """按 ID 查用户。"""
    try:
        with get_session() as session:
            user = session.query(UserModel).filter(UserModel.id == user_id).first()
            return _model_to_user(user) if user else None
    except Exception:
        logger.exception("查询用户失败")
        return None


def get_user_by_username(username: str) -> User | None:
    """按用户名查用户。"""
    try:
        with get_session() as session:
            user = (
                session.query(UserModel)
                .filter(UserModel.username == username.strip())
                .first()
            )
            return _model_to_user(user) if user else None
    except Exception:
        logger.exception("查询用户失败")
        return None


# ============================================================
# Admin 管理函数
# ============================================================

def list_users() -> list[User]:
    """列出所有用户（admin 专用）。"""
    try:
        with get_session() as session:
            users = session.query(UserModel).order_by(UserModel.id).all()
            return [_model_to_user(u) for u in users]
    except Exception:
        logger.exception("列出用户失败")
        return []


def toggle_user_active(user_id: int) -> bool:
    """切换用户启用/禁用状态。"""
    try:
        with get_session() as session:
            user = session.query(UserModel).filter(UserModel.id == user_id).first()
            if not user:
                return False
            user.is_active = not user.is_active
            session.flush()
            return True
    except Exception:
        logger.exception("切换用户状态失败")
        return False


def reset_password(user_id: int, new_password: str) -> tuple[bool, str | None]:
    """Admin 强制重置密码。"""
    if len(new_password) < 6:
        return False, AuthError.PASSWORD_TOO_SHORT

    try:
        with get_session() as session:
            user = session.query(UserModel).filter(UserModel.id == user_id).first()
            if not user:
                return False, AuthError.DB_ERROR
            user.password_hash = hash_password(new_password)
            session.flush()
            return True, None
    except Exception:
        logger.exception("重置密码失败")
        return False, AuthError.DB_ERROR


# ============================================================
# Streamlit Session State 辅助函数
# ============================================================

def is_logged_in() -> bool:
    """检查当前是否已登录。"""
    return st.session_state.get("user_id") is not None


def get_current_user() -> dict | None:
    """从 session_state 获取当前用户信息。"""
    if not is_logged_in():
        return None
    return {
        "user_id": st.session_state.user_id,
        "username": st.session_state.username,
        "role": st.session_state.role,
    }


def require_admin() -> bool:
    """检查当前用户是否为 admin。"""
    return st.session_state.get("role") == "admin"


def logout_user():
    """清除所有用户相关 session_state 字段。"""
    keys_to_clear = [
        "user_id",
        "username",
        "role",
        "page",
        "messages",
        "password_change_error",
        "password_change_success",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
