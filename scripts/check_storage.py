r"""
存储层自检
================================================================

检查 SQLite 数据库层是否正常工作：
    · 表能不能建出来
    · 外键约束有没有真的开启（这是 SQLite 最著名的坑）
    · CHECK 约束、级联删除是否生效

运行：
    .\.venv\Scripts\python.exe scripts\check_storage.py

什么时候需要跑它？
    · 做完阶段 2（持久化）之后
    · 遇到 "no such table" / "FOREIGN KEY constraint failed" 之类的错误时
    · 怀疑数据库文件损坏时
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from memory_assistant.storage.database import run_self_check  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_self_check())
