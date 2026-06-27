"""
知识库助手 — Streamlit Web 前端
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

# ---- 页面配置 ----
st.set_page_config(
    page_title="知识库助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 样式 ----
st.markdown("""
<style>
.stMainBlockContainer { max-width: 1200px; }
.stChatMessage { border-radius: 12px; }
.stSidebar .stMetric { font-size: 13px; }
.upload-section { border: 1px dashed #ccc; border-radius: 12px; padding: 16px; margin: 12px 0; }
.stats-label { font-size: 12px; color: #999; }
</style>
""", unsafe_allow_html=True)


# ---- 初始化 ----

@st.cache_resource
def get_assistant() -> KnowledgeAssistant:
    """全局单例助手，序列化缓存避免重复初始化。"""
    return KnowledgeAssistant()


def init_session_state():
    defaults = {
        "messages": [],
        "pending_files": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()
assistant = get_assistant()


# ---- 侧边栏 ----

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
                    st.error(f"导入失���: {file.name} - {e}")
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
            progress.empty()
            st.success(f"已导入 {len(uploaded)} 个文件")
            st.rerun()

    # 记忆管理
    with st.expander("🧠 记忆管理", expanded=False):
        fact = st.text_area("添加事实记忆", placeholder="例如：我喜欢用 Python 做 AI 项目", key="add_fact")
        if st.button("保存记忆", use_container_width=True):
            if fact.strip():
                assistant.remember_fact(fact.strip(), importance=0.7)
                st.success("已保存")
                st.rerun()

        preference_key = st.text_input("偏好键", placeholder="preferred_language", key="pref_key")
        preference_value = st.text_input("偏好值", placeholder="中文", key="pref_val")
        if st.button("保存偏好", use_container_width=True):
            if preference_key.strip() and preference_value.strip():
                assistant.remember_preference(preference_key.strip(), preference_value.strip())
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

    st.divider()
    st.caption(f"模型: DeepSeek chat | Embedding: BGE-zh-v1.5")


# ---- 主聊天区 ----

st.markdown("### 💬 对话")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("输入你的问题，基于知识库回答..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回复（流式）
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 收集流式 token
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
