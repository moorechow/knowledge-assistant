"""
命令行入口
知识库助手的 CLI 交互界面。
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.config import get_config

# 项目 src 目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)


def setup_logging():
    level = get_config().log_level
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner():
    print(r"""
  _   __                           __              __   __       _
 | | / /                          / /             / /  / _|     (_)
 | |/ /  ___  _   _ ___  _   __ / /__   ___     / /__| |_ ___   _  ___  _ __
 |    \ / _ \| | | / __|| | / // //_/  / _ \   / //_/|  _// _ \ | |/ _ \| '_ \
 | |\  \  __/| |_| \__ \| |/ // ,<    | (_) | / /  _| | | (_) || |  __/| | | |
 \_| \_/\___| \__,_|___/ \___//_/|_|   \___/  \/  (_)_| |_|\___/ |_|\___||_| |_|
    """)


async def interactive_mode(assistant):
    """交互式对话模式。"""
    print_banner()
    print("个人知识库 AI 助手  v0.1.0")
    print("输入 'quit' 或 'exit' 退出，输入 '/help' 查看命令")
    print(f"已加载知识库: {assistant.knowledge_count} 条")
    print(f"已注册 Skill: {assistant.skill_names}")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break
        elif user_input == "/help":
            print("""
  命令列表:
    /stats    - 查看统计信息
    /clear    - 清空当前对话
    /ingest <path> - 导入文档/目录到知识库
    /skills   - 查看已注册的 Skill
    /summarize <text> - 对文本进行摘要
    /help     - 显示此帮助
            """)
            continue
        elif user_input == "/stats":
            stats = assistant.stats()
            print("\n[统计信息]")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            continue
        elif user_input == "/clear":
            assistant.clear_conversation()
            print("[对话已清空]")
            continue
        elif user_input == "/skills":
            print(f"\n已注册 Skill: {assistant.skill_names}")
            continue
        elif user_input.startswith("/ingest "):
            path = user_input[len("/ingest "):].strip()
            try:
                p = Path(path)
                if p.is_dir():
                    assistant.ingest_directory(str(p))
                else:
                    assistant.ingest_document(str(p))
                print(f"[已导入: {path}]")
            except Exception as e:
                print(f"[导入失败: {e}]")
            continue
        elif user_input.startswith("/summarize "):
            text = user_input[len("/summarize "):].strip()
            result = assistant.execute_skill("summarize", text=text)
            if result.success:
                print(f"\n[摘要]\n{result.data['summary']}")
            else:
                print(f"[摘要失败: {result.error}]")
            continue

        # 正常对话
        try:
            print("\n助手: ", end="", flush=True)
            async for token in assistant.chat_stream(user_input):
                print(token, end="", flush=True)
            print()
        except Exception as e:
            logger.exception("对话出错")
            print(f"\n[错误: {e}]")


def main():
    parser = argparse.ArgumentParser(description="个人知识库 AI 助手")
    parser.add_argument(
        "--ingest",
        type=str,
        help="启动时导入指定文档或目录到知识库",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API Key（也可通过 OPENAI_API_KEY 环境变量设置）",
    )
    args = parser.parse_args()

    setup_logging()

    # 如果命令行传入了 API Key，设置到环境变量
    if args.api_key:
        import os
        os.environ["OPENAI_API_KEY"] = args.api_key

    from src.assistant import KnowledgeAssistant

    assistant = KnowledgeAssistant()

    # 启动时导入知识
    if args.ingest:
        p = Path(args.ingest)
        try:
            if p.is_dir():
                assistant.ingest_directory(str(p))
            else:
                assistant.ingest_document(str(p))
        except Exception as e:
            logger.error("启动导入失败: %s", e)

    # 进入交互模式
    asyncio.run(interactive_mode(assistant))


if __name__ == "__main__":
    main()
