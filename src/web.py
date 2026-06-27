"""
知识库助手 — Streamlit Web 前端（支持多用户登录）。
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# 确保 src 在路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from src.assistant import KnowledgeAssistant  # noqa: E402
from src.auth import (  # noqa: E402
    AuthError,
    change_password,
    get_current_user,
    is_logged_in,
    list_users,
    login_user,
    logout_user,
    register_user,
    require_admin,
    reset_password,
    toggle_user_active,
)
from src.user_db import init_db  # noqa: E402

# ---- 页面配置 ----
st.set_page_config(
    page_title="知识库助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 全局样式 ----
st.markdown("""
<style>
.stMainBlockContainer { max-width: 1200px; }
.stChatMessage { border-radius: 12px; }
.stSidebar .stMetric { font-size: 13px; }

/* 登录/注册卡片 */
.auth-card {
    background: white;
    border-radius: 16px;
    padding: 40px 32px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    max-width: 420px;
    margin: 60px auto 0 auto;
}
.auth-title {
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 28px;
    color: #1a1a1a;
}
.auth-error {
    background: #fce4e4;
    color: #c62828;
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 14px;
}
.auth-link {
    text-align: center;
    margin-top: 18px;
    font-size: 14px;
    color: #666;
}
.auth-link a {
    color: #1f77b4;
    text-decoration: none;
    font-weight: 500;
}

/* 用户信息栏 */
.user-bar {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 12px;
    padding: 4px 0 8px 0;
}
.user-badge {
    background: #e8f0fe;
    color: #1f77b4;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 500;
}
.admin-badge {
    background: #fff3e0;
    color: #e65100;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ---- 初始化 ----

def init_session_state():
    """初始化 session_state 默认值。"""
    defaults = {
        "page": "login",
        "user_id": None,
        "username": None,
        "role": None,
        "messages": [],
        "pending_files": [],
        "login_error": None,
        "register_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_assistant() -> KnowledgeAssistant:
    """创建 KnowledgeAssistant 实例（每次 rerun 重建）。"""
    return KnowledgeAssistant()


# ---- 页面路由 ----

def render_page():
    """根据 session_state.page 渲染对应页面。"""
    # 初始化数据库
    try:
        init_db()
    except Exception:
        pass

    # 判定默认页面
    if "page" not in st.session_state:
        st.session_state.page = "main" if is_logged_in() else "login"

    page = st.session_state.page

    if page == "login":
        render_login_page()
    elif page == "register":
        render_register_page()
    elif page == "main":
        if not is_logged_in():
            st.session_state.page = "login"
            st.rerun()
        render_main_page()
    else:
        render_login_page()


# ============================================================
# 登录页面
# ============================================================

def render_login_page():
    """渲染登录页面。"""
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">📚 知识库助手 — 登录</div>',
                unsafe_allow_html=True)

    # 错误提示
    if st.session_state.get("login_error"):
        st.markdown(
            f'<div class="auth-error">{st.session_state.login_error}</div>',
            unsafe_allow_html=True,
        )

    username = st.text_input(
        "用户名",
        placeholder="请输入用户名",
        key="login_username",
    )
    password = st.text_input(
        "密码",
        type="password",
        placeholder="请输入密码",
        key="login_password",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        login_btn = st.button(
            "登 录",
            use_container_width=True,
            type="primary",
            disabled=(not username or not password),
        )

    if login_btn:
        user, error = login_user(username, password)
        if user:
            st.session_state.user_id = user.id
            st.session_state.username = user.username
            st.session_state.role = user.role
            st.session_state.login_error = None
            st.session_state.page = "main"
            st.rerun()
        else:
            # 统一显示 "用户名或密码错误"
            if error == AuthError.ACCOUNT_DISABLED:
                st.session_state.login_error = error.value
            else:
                st.session_state.login_error = AuthError.INVALID_CREDENTIALS.value
            st.rerun()

    st.markdown(
        '<div class="auth-link">还没有账户？'
        '<a href="javascript:void(0)" onclick="return false;">'
        '立即注册</a></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("创建新账户", use_container_width=True):
            st.session_state.login_error = None
            st.session_state.page = "register"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 注册页面
# ============================================================

def render_register_page():
    """渲染注册页面。"""
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">✨ 创建新账户</div>',
                unsafe_allow_html=True)

    # 错误提示
    if st.session_state.get("register_error"):
        st.markdown(
            f'<div class="auth-error">{st.session_state.register_error}</div>',
            unsafe_allow_html=True,
        )

    username = st.text_input(
        "用户名",
        placeholder="3-20 位字母/数字/下划线",
        key="reg_username",
    )
    password = st.text_input(
        "密码",
        type="password",
        placeholder="至少 6 位",
        key="reg_password",
    )
    confirm = st.text_input(
        "确认密码",
        type="password",
        placeholder="再次输入密码",
        key="reg_confirm",
    )
    email = st.text_input(
        "邮箱（选填）",
        placeholder="用于找回密码（选填）",
        key="reg_email",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        can_register = bool(username and password and confirm)
        register_btn = st.button(
            "注 册",
            use_container_width=True,
            type="primary",
            disabled=not can_register,
        )

    if register_btn:
        # 前端校验
        if len(username) < 3 or len(username) > 20:
            st.session_state.register_error = AuthError.USERNAME_TOO_SHORT.value
        elif len(password) < 6:
            st.session_state.register_error = AuthError.PASSWORD_TOO_SHORT.value
        elif password != confirm:
            st.session_state.register_error = AuthError.PASSWORD_MISMATCH.value
        else:
            user, error = register_user(
                username, password,
                email=email.strip() if email else None,
            )
            if user:
                # 自动登录
                st.session_state.user_id = user.id
                st.session_state.username = user.username
                st.session_state.role = user.role
                st.session_state.register_error = None
                st.session_state.page = "main"
                st.session_state.messages = []
                st.toast("注册成功！", icon="✅")
                st.rerun()
            else:
                st.session_state.register_error = error
        st.rerun()

    st.markdown(
        '<div class="auth-link">已有账户？'
        '<a href="javascript:void(0)">去登录</a></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("返回登录", use_container_width=True):
            st.session_state.register_error = None
            st.session_state.page = "login"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 主页（知识库助手聊天）
# ============================================================

def render_main_page():
    """渲染知识库助手主页。"""
    assistant = get_assistant()

    # ---- 顶部信息栏 ----
    _render_user_bar()

    st.divider()

    # ---- 侧边栏 ----
    _render_sidebar(assistant)

    # ---- 主聊天区 ----
    st.markdown("### 💬 对话")

    # 渲染历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    if prompt := st.chat_input("输入你的问题，基于知识库回答..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    async def _collect():
                        tokens = []
                        async for token in assistant.chat_stream(prompt):
                            tokens.append(token)
                        return "".join(tokens)

                    response = asyncio.run(_collect())
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    error_msg = f"出错了: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


def _render_user_bar():
    """渲染顶部用户信息栏。"""
    cols = st.columns([5, 2, 1.5, 1.5, 1.5])
    with cols[1]:
        username = st.session_state.username or "用户"
        role = st.session_state.role or "user"
        badge_class = "admin-badge" if role == "admin" else "user-badge"
        role_text = f"👤 {username}" + (" (管理员)" if role == "admin" else "")
        st.markdown(
            f'<div class="{badge_class}" style="text-align:center;">'
            f'{role_text}</div>',
            unsafe_allow_html=True,
        )
    with cols[3]:
        if st.button("🔒 修改密码", use_container_width=True, key="pw_btn_top"):
            st.session_state.show_password_change = True
    with cols[4]:
        if st.button("🚪 登出", use_container_width=True, key="logout_btn"):
            logout_user()
            st.session_state.page = "login"
            st.rerun()


def _render_sidebar(assistant: KnowledgeAssistant):
    """渲染侧边栏。"""
    with st.sidebar:
        st.title("📚 知识库助手")

        # 统计面板
        stats = assistant.stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("知识块", stats["knowledge_chunks"])
            st.metric("长期记忆", stats["long_term_entries"])
        with col2:
            st.metric("对话轮次", stats["session_turns"])
            st.metric("已注册技能", stats["registered_skills"])

        st.divider()

        # 知识库管理
        with st.expander("📂 知识库管理", expanded=False):
            uploaded = st.file_uploader(
                "上传文档（PDF / Markdown / TXT / DOCX）",
                type=["pdf", "md", "txt", "docx"],
                accept_multiple_files=True,
                key="kb_uploader",
            )
            if uploaded:
                progress = st.progress(0, text="准备导入...")
                for i, file in enumerate(uploaded):
                    progress.progress(
                        (i + 1) / len(uploaded),
                        text=f"导入中: {file.name}",
                    )
                    tmp_path = Path(tempfile.gettempdir()) / file.name
                    with open(tmp_path, "wb") as f:
                        f.write(file.getbuffer())
                    try:
                        assistant.ingest_document(str(tmp_path))
                    except Exception as e:
                        st.error(f"导入失败: {file.name} - {e}")
                    finally:
                        if tmp_path.exists():
                            tmp_path.unlink()
                progress.empty()
                st.success(f"已导入 {len(uploaded)} 个文件")
                st.rerun()

        # 记忆管理
        with st.expander("🧠 记忆管理", expanded=False):
            fact = st.text_area(
                "添加事实记忆",
                placeholder="例如：我喜欢用 Python 做 AI 项目",
                key="add_fact",
            )
            if st.button("保存记忆", use_container_width=True):
                if fact.strip():
                    assistant.remember_fact(fact.strip(), importance=0.7)
                    st.success("已保存")
                    st.rerun()

            preference_key = st.text_input(
                "偏好键", placeholder="preferred_language", key="pref_key"
            )
            preference_value = st.text_input(
                "偏好值", placeholder="中文", key="pref_val"
            )
            if st.button("保存偏好", use_container_width=True):
                if preference_key.strip() and preference_value.strip():
                    assistant.remember_preference(
                        preference_key.strip(), preference_value.strip()
                    )
                    st.success("已保存")
                    st.rerun()

        # 会话管理
        with st.expander("⚙️ 会话管理", expanded=False):
            if st.button("清空当前对话", use_container_width=True):
                assistant.clear_conversation()
                st.session_state.messages = []
                st.rerun()
            if st.button("完全重置", use_container_width=True, type="secondary"):
                assistant.reset()
                st.session_state.messages = []
                st.rerun()

        # 修改密码
        with st.expander("🔒 修改密码", expanded=st.session_state.get(
                "show_password_change", False)):
            old_pw = st.text_input(
                "旧密码", type="password", key="pw_old"
            )
            new_pw = st.text_input(
                "新密码（至少 6 位）", type="password", key="pw_new"
            )
            confirm_pw = st.text_input(
                "确认新密码", type="password", key="pw_confirm"
            )
            if st.button("确认修改", use_container_width=True, key="pw_submit"):
                if new_pw != confirm_pw:
                    st.error(AuthError.PASSWORD_MISMATCH.value)
                elif len(new_pw) < 6:
                    st.error(AuthError.PASSWORD_TOO_SHORT.value)
                else:
                    user_id = st.session_state.user_id
                    ok, err = change_password(user_id, old_pw, new_pw)
                    if ok:
                        st.success("密码修改成功")
                        st.session_state.show_password_change = False
                        st.rerun()
                    else:
                        st.error(err)

        # Admin 面板
        if require_admin():
            st.divider()
            with st.expander("👥 用户管理", expanded=False):
                users = list_users()
                if users:
                    # 构建表格数据
                    table_data = []
                    for u in users:
                        status = "✅" if u.is_active else "❌"
                        table_data.append({
                            "ID": u.id,
                            "用户名": u.username,
                            "角色": u.role,
                            "状态": status,
                            "创建时间": u.created_at[:10] if u.created_at else "",
                        })

                    st.dataframe(
                        table_data,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("---")
                    st.caption("用户操作")

                    target_uid = st.number_input(
                        "用户 ID", min_value=1, step=1, key="admin_uid"
                    )

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("启用/禁用", use_container_width=True):
                            toggle_user_active(target_uid)
                            st.rerun()
                    with col_b:
                        new_pw_admin = st.text_input(
                            "新密码",
                            type="password",
                            key="admin_reset_pw",
                            placeholder="至少 6 位",
                        )
                        if st.button("重置密码", use_container_width=True):
                            ok, err = reset_password(target_uid, new_pw_admin)
                            if ok:
                                st.success("密码已重置")
                            else:
                                st.error(err)

        st.divider()
        st.caption(f"👤 {st.session_state.username} ({st.session_state.role})")
        st.caption("模型: DeepSeek chat | Embedding: BGE-zh-v1.5")


# ---- 入口 ----

if __name__ == "__main__":
    init_session_state()
    render_page()
