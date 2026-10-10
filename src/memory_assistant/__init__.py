r"""
memory_assistant —— 有记忆的私人助理

包结构说明（这是 Python 的 src 布局，也是业界推荐的写法）：

    src/memory_assistant/
    ├─ __init__.py          ← 你正在看的文件，标志"这是一个包"
    ├─ config.py            ← 配置（已提供）
    ├─ llm.py               ← 模型客户端 + 假模型（已提供）
    ├─ storage/             ← SQLite 持久化层（已提供）
    │  ├─ database.py       ←   连接 / 建表 / 事务 / schema
    │  └─ repositories.py   ←   sessions / messages / summaries 仓储
    ├─ memory/              ← 记忆模块（你的主战场）
    │  ├─ short_term.py     ← 短期记忆（第 3 周，已给出规格和测试）
    │  ├─ summary.py        ← 中期摘要（第 4 周）
    │  ├─ long_term.py      ← 长期向量记忆（第 5 周）
    │  └─ profile.py        ← 用户画像（第 10 周）
    ├─ tools/               ← 工具（第 7 周）
    └─ cli.py               ← 命令行入口（第 4 周）

为了方便使用，这里把常用的东西"提到"包的顶层，
这样外面可以写 `from memory_assistant import Config`，
而不必写 `from memory_assistant.config import Config`。

⚠️ 注意：随着你后面添加模块，如果这里 import 了还不存在的文件，程序会报错。
   所以下面被注释掉的导入，是你实现到那一步之后再取消注释的。
"""

from memory_assistant.config import PROJECT_ROOT, Config
from memory_assistant.embeddings import (
    EmbeddingProvider,
    FakeEmbeddings,
    LocalEmbeddings,
    create_embeddings,
)
from memory_assistant.llm import (
    ChatModel,
    DeepSeekClient,
    FakeLLM,
    Usage,
    create_llm,
    estimate_messages_tokens,
    estimate_tokens,
)
from memory_assistant.storage import Database, Store, utc_now_iso
from memory_assistant.tools import (
    Tool,
    ToolRegistry,
    calculate,
    create_default_tools,
    create_learning_tools,
    create_memory_tools,
    get_current_time,
)

# 第 3 周实现 ShortTermMemory 之后，取消下面这行的注释：
# from memory_assistant.memory import ShortTermMemory

# __all__ 声明"这个包对外提供哪些名字"。
# 好处：别人写 from memory_assistant import * 时，只会导入这里列出的东西，
#       而不会把内部使用的变量也一起导出去。
__all__ = [
    "PROJECT_ROOT",
    "Config",
    "ChatModel",
    "DeepSeekClient",
    "FakeLLM",
    "Usage",
    "create_llm",
    "estimate_tokens",
    "estimate_messages_tokens",
    "Database",
    "Store",
    "utc_now_iso",
    "EmbeddingProvider",
    "FakeEmbeddings",
    "LocalEmbeddings",
    "create_embeddings",
    "Tool",
    "ToolRegistry",
    "calculate",
    "get_current_time",
    "create_default_tools",
    "create_learning_tools",
    "create_memory_tools",
]

__version__ = "0.1.0"
