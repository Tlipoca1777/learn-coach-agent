r"""
记忆模块
================================================================

四层记忆设计：

    short_term.py   短期记忆 —— 最近几轮的原文
    summary.py      中期记忆 —— 更早对话的压缩摘要   （第 4 周）
    long_term.py    长期记忆 —— 抽取的事实 + 向量检索 （第 5 周）
    profile.py      用户画像 —— 结构化的人物属性     （第 10 周）

四者组合起来，由 memory/router.py（第 6 周）决定每轮注入哪些、各占多少 token。

短期的规格和测试已下发；摘要的规格和测试也已下发。
long_term / profile 是后面几周的内容，实现之后再往这里加导出。
"""

from memory_assistant.memory.extraction import ExtractedFact, FactExtractor
from memory_assistant.memory.long_term import LongTermMemory
from memory_assistant.memory.short_term import ShortTermMemory
from memory_assistant.memory.summary import SummaryMemory

__all__ = [
    "ShortTermMemory",
    "SummaryMemory",
    "LongTermMemory",
    "ExtractedFact",
    "FactExtractor",
]
