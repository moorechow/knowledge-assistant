"""
用户数据库层 — SQLAlchemy 模型 + SQLite 初始化。

存储引擎: SQLite，文件路径 data/users.db
首次调用 init_db() 自动建表并创建默认 admin 账户。
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

import bcrypt

logger = logging.getLogger(__name__)


def _hash_password(plain: str) -> str:
    """bcrypt 哈希（内部使用，避免循环导入）。"""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

# ---- 项目根路径 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- SQLAlchemy 基类 ----
class Base(DeclarativeBase):
    pass


class UserModel(Base):
    """用户表 ORM 模型。"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, nullable=False, unique=True)
    password_hash = Column(String, nullable=False)
    email = Column(String, nullable=True)
    role = Column(String, nullable=False, default="user")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.now)
    last_login = Column(DateTime, nullable=True)


# ---- 引擎与会话工厂 ----
_engine = None
_SessionFactory = None


def get_engine():
    """创建或返回 SQLite 引擎（懒加载）。"""
    global _engine
    if _engine is None:
        db_path = PROJECT_ROOT / "data" / "users.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},  # Streamlit 多线程必需
            echo=False,
        )
    return _engine


@contextmanager
def get_session():
    """返回 SQLAlchemy Session 上下文管理器。"""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---- 用户纯数据类（不含 password_hash）----

@dataclass
class User:
    """用户数据对象，用于在 auth 层和 web 层传递。"""
    id: int
    username: str
    email: str | None
    role: str
    is_active: bool
    created_at: str
    last_login: str | None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "last_login": self.last_login,
        }


def _model_to_user(m: UserModel) -> User:
    """将 ORM 模型转为纯数据对象。"""
    return User(
        id=m.id,
        username=m.username,
        email=m.email,
        role=m.role,
        is_active=m.is_active,
        created_at=m.created_at.isoformat() if m.created_at else "",
        last_login=m.last_login.isoformat() if m.last_login else None,
    )


# ---- 数据库初始化 ----

def init_db():
    """建表并创建默认 admin 账户（仅当 users 表为空时）。"""
    engine = get_engine()
    Base.metadata.create_all(engine)

    with get_session() as session:
        if session.query(UserModel).count() == 0:
            admin = UserModel(
                username="admin",
                password_hash=_hash_password("admin123"),
                email=None,
                role="admin",
                is_active=True,
                created_at=datetime.now(),
            )
            session.add(admin)
            logger.info("已创建默认 admin 账户（admin / admin123）")
