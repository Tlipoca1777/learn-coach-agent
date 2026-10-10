# 有记忆的私人助理 Agent

> 一个会**真正记住你**的 AI 助手。
>
> 和直接用 ChatGPT 的区别：它有一套自己实现的**四层记忆系统**，
> 记得你刚才说了什么、几周前聊过什么、你是一个什么样的人，也知道什么时候该忘。

---

## 当前进度

| 单元 | 内容 | 测试 | 状态 |
|---|---|---|---|
| **D1** | `ShortTermMemory`（滑动窗口 + token 预算） | 19 | ✅ 已完成 |
| **D2** | `SummaryMemory`（中期递归摘要） | 42 | ✅ 已完成 |
| **D3** | `LongTermMemory`（集合、事实写入与读取） | 基础用例通过 | ✅ 已完成 |
| **D4** | `LongTermMemory`（三因子评分、去重与失效） | 51 | ✅ 已完成 |
| **D5** | `FactExtractor`（结构化输出 + 健壮解析） | 57 | ✅ 已完成 |
| D6 | 完整记忆链路集成（`python -m memory_assistant`） | — | ⏳ 待验证端到端流程 |
| D7+ | 工具调用 / LangGraph / 学习教练业务 / 部署 | — | ⏳ |

当前测试基线（2026-10-10）：

- 全量测试：**266 passed，4 skipped**（共 270 条）
- D1 短期记忆：**19/19 通过**
- D2 中期摘要：**42/42 通过**
- D3 长期记忆基础层：集合初始化、事实写入、按 ID 读取和基础检索已完成
- D4 长期记忆增强层：三因子评分、时间衰减、命中加权、去重、失效与删除已完成（51/51）
- D5 事实抽取：prompt 构造、健壮 JSON 解析、置信度过滤、批内去重与异常保护已完成（57/57）
- 已就绪基础设施：SQLite 存储 **39/39**、对话引擎 **38/38**、向量层 **20/20**（另有 4 条真实模型慢测默认跳过）
- 当前单元测试全绿；完整记忆链路和 CLI 端到端行为仍需 D6 验证

已就绪的基础设施：

- 配置管理、DeepSeek 客户端（含离线假模型 `FakeLLM`）、环境自检脚本
- SQLite 持久化层（`Database` + `Store`）—— 39 个测试
- 向量层（本地中文 BGE 模型 + 假向量模型）—— 20 个快速测试
- 对话引擎 + 命令行界面（依赖注入设计，可脱离记忆模块单独测试）—— 38 个测试

> 向量模型的选型有实测数据支撑，见 [记忆架构设计 · 第 12 节](docs/记忆架构设计.md#12-向量模型选型实测记录)

---

## 这个项目解决什么问题

大模型本身**没有记忆**。它是无状态的：每次请求你都得把之前的对话重新发一遍。

于是产生两个致命问题：

| 问题 | 后果 |
|---|---|
| 历史全量塞回去 | token 爆炸、成本飙升、最终超出上下文窗口 |
| 只塞最近几轮 | 忘了用户是谁、忘了重要偏好，"每次都像在跟陌生人说话" |

主流产品的做法是简单粗暴地「保留最近 N 轮」，信息损失很大。
本项目做了更细的分层：**该记住的记住，该压缩的压缩，该忘的忘掉。**

---

## 核心设计：四层记忆

| 层次 | 存什么 | 技术方案 | 解决的问题 |
|---|---|---|---|
| **短期** | 最近几轮原文 | 按 token 预算裁剪的滑动窗口 | 对话连贯性 |
| **中期** | 更早对话的摘要 | LLM 递归增量压缩 | 上下文窗口限制 |
| **长期** | 抽取出的结构化事实 | 向量检索 + 时间衰减 | 跨会话记住细节 |
| **画像** | 姓名/职业/偏好/禁忌 | 结构化 JSON，增量合并 | 稳定、便宜、可控 |

**不只是"存进去"，还包括：**

- **抽取**：从对话里自动识别值得记住的事实（`subject-predicate-object` 三元组）
- **去重**：相似事实合并，累加命中次数而不是重复存储
- **冲突消解**：用户改主意时（"我戒咖啡了"）让模型裁决，旧事实标记失效而非删除
- **遗忘**：时间衰减 + 低频淘汰 + TTL，防止向量库无限膨胀
- **配额**：显式的 prompt token 预算表，保证输入永不失控

---

## Token 预算分配

每次请求的输入 token 都受这张表约束，超出配额就按优先级裁剪：

| 组成部分 | 预算占比 | 超预算时的处理 |
|---|---|---|
| 系统提示词 | 10% | 固定，不可压缩 |
| 用户画像 | 10% | 字段按重要性排序，从低到高砍 |
| 长期记忆 | 20% | 按综合分数取 top-k |
| 中期摘要 | 20% | 过长时按 token 上限截断并加省略号 |
| 近期对话 | 40% | 从最老的开始丢弃 |

---

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.12 |
| 大模型 | DeepSeek（OpenAI 兼容接口） |
| 编排 | LangGraph（StateGraph + Checkpointer + Store） |
| 结构化输出 | pydantic |
| 向量库 | Chroma |
| 关系存储 | SQLite |
| 后端 | FastAPI（SSE 流式） |
| 前端 | Streamlit |
| 部署 | Docker + docker-compose |
| 测试 | pytest |

---

## 快速开始

```powershell
# 1. 进入项目目录
cd D:\Agent开发

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps   # 把本项目装成可导入的包

# 3. 配置密钥
Copy-Item .env.example .env
#    然后用编辑器打开 .env，填入你的 DEEPSEEK_API_KEY
#    申请地址：https://platform.deepseek.com/api_keys
#    默认模型是 deepseek-flash；旧的 deepseek-chat 兼容名目前也会路由到 flash。
#    如需确认实际路由，运行环境自检，或查看 DeepSeek 后台的用量记录。

# 4. 环境自检（以后遇到任何问题，第一件事就是跑它）
.\.venv\Scripts\python.exe scripts\check_env.py

# 5. 第一次调用大模型
.\.venv\Scripts\python.exe scripts\step0_hello_llm.py

# 6. 开始对话
.\.venv\Scripts\python.exe scripts\step1_chat_cli.py

# 7. 运行测试
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

> ⚠️ 所有 `python` 命令都要用 `.\.venv\Scripts\python.exe`，不要用全局的 `python`。
> 否则会提示找不到 `openai` 模块。

---

## 目录结构

```
learn-coach-agent/
├─ README.md
├─ requirements.txt
├─ pyproject.toml               ← 打包配置 + pytest 配置
├─ conftest.py                  ← pytest 全局配置（修 Windows 中文乱码）
├─ .env.example                 ← 配置模板，复制成 .env 后填真实值
├─ .gitignore / .gitattributes
├─ docs/                        ← 项目文档
│  ├─ 记忆架构设计.md            ← 四层记忆完整设计（含向量模型选型实测）
│  └─ 业务设计.md                ← 学习教练业务设计
├─ scripts/
│  ├─ check_env.py              ← 环境自检（遇到问题先跑它）
│  ├─ check_storage.py          ← 数据库层自检
│  ├─ check_embeddings.py       ← 向量层自检（含中文语义检索效果验证）
│  ├─ step0_hello_llm.py        ← 最小可运行示例：跑通 API 调用
│  ├─ step1_chat_cli.py         ← 多轮对话 + 流式输出
│  └─ step2_budget_experiment.py← 三种历史策略的 token 对比实验
├─ src/memory_assistant/
│  ├─ config.py                 ← 配置集中管理
│  ├─ llm.py                    ← 模型客户端 + 离线假模型 FakeLLM
│  ├─ embeddings.py             ← 向量化层：本地中文 BGE + 假向量
│  ├─ storage/                  ← SQLite 持久化层
│  │  ├─ database.py            ←   连接 / 建表 / 事务 / schema
│  │  └─ repositories.py        ←   sessions / messages / summaries 仓储
│  ├─ memory/                   ← 四层记忆
│  │  ├─ short_term.py          ←   短期：滑动窗口 + token 预算
│  │  ├─ summary.py             ←   中期：递归增量摘要
│  │  ├─ long_term.py           ←   长期：向量库 + 三因子评分 + 去重/失效
│  │  └─ extraction.py          ←   事实抽取（结构化输出 + 健壮解析）
│  ├─ engine.py                 ← 对话引擎：把四层记忆串起来
│  ├─ cli.py / __main__.py      ← 命令行界面
│  └─ __init__.py               ← 包入口
├─ tests/                       ← 共 270 个测试
│  ├─ test_short_term.py        ←   19
│  ├─ test_summary.py           ←   42
│  ├─ test_long_term.py         ←   51
│  ├─ test_extraction.py        ←   57
│  ├─ test_engine.py            ←   38
│  ├─ test_storage.py           ←   39
│  └─ test_embeddings.py        ←   24（其中 4 条需设置 RUN_SLOW_TESTS=1）
└─ data/                        ← 本地数据（.gitignore 排除）
   └─ models/                   ←   向量模型缓存（约 90MB）
```

---

## 内置命令（CLI）

| 命令 | 作用 |
|---|---|
| `/exit` | 退出 |
| `/status` | 查看会话状态、消息数和 token 占用 |
| `/memory` | 查看当前注入 prompt 的记忆 |
| `/help` | 显示命令帮助 |
| `/forget` | 清空当前会话消息、摘要和长期事实 |
| `--fake` | 启动离线假模型模式（集成入口会提示尚未实现的记忆模块） |

---

## 设计取舍（面试重点）

> 这部分是面试官最感兴趣的内容。每做一个决策，就回来补一条。

- **为什么按 token 预算裁剪，而不是按消息条数？**
  因为中英文单条消息长度差异极大，按条数会让预算完全失控。实测数据待补（阶段 1）。

- **为什么长期记忆存"抽取的事实"而不是"原始对话"？**
  原始对话检索噪声大、命中率低、还费 token。结构化事实可以精确匹配、去重、做冲突消解。
  代价是抽取会引入错误和额外成本，需要 pydantic 校验兜底。

- **为什么记忆检索要做成工具，而不是每次无脑注入？**
  无脑注入会让每轮都付出固定 token 成本，且大部分时候用不上。
  做成工具后模型按需调用，省 token；代价是模型可能"忘记去回忆"，需要优化工具描述。

- **为什么先用自研记忆，后来才重构到 LangGraph？**
  先手写才能真正理解 checkpointer 和 store 在解决什么问题。
  重构后获得了断点恢复、流式、图可视化等能力，但自研的去重和冲突消解逻辑仍然保留。

（后续持续补充）

---

## 评测数据

### 阶段 1：token 预算裁剪实验

用 `scripts/step2_budget_experiment.py` 生成一个长度分布极不均匀的对话
（日常短对话 + 用户粘贴的长文档），对比三种历史策略：

| 策略 | 消息数 | 输入 token | 占预算 | 结果 |
|---|---|---|---|---|
| ① 全量历史（不裁剪） | 81 | 6787 | 170% | ⚠️ 超预算 |
| ② 朴素条数截断（最近 10 条） | 11 | 4528 | 113% | ⚠️ 超预算 528 |
| ③ **token 预算裁剪（本项目）** | 6 | **2278** | **57%** | ✅ 不超预算 |

场景：80 条消息，预算 4000 token，其中含 2 份各约 2000 token 的长文档。

**结论**：
- 相比全量历史，**节省 66.4% 输入 token**（6787 → 2278）
- 朴素截断「看起来只留了 10 条很克制」，实际留了 4528 token，**超预算 13%**
  —— 因为「条数」和「token 数」之间没有换算关系
- 我们的方案代价是**预算利用率只有 57%**：规格要求「装不下就停」，
  遇到超长消息时会放弃更早的对话。这个代价可接受，因为短期记忆只负责连贯性，
  用户身份信息由画像和长期记忆层负责

> ⏳ 上方数字来自一份参考实现的基准跑分。等你实现 `ShortTermMemory` 后，
> 用自己的代码跑一遍并把结果替换进来（数值应基本一致）。

### 阶段 7：记忆效果评测（待做）

| 实验 | 指标 | 无记忆 | 仅短期 | 四层记忆 |
|---|---|---|---|---|
| 跨会话事实问答 | 准确率 | 待测 | 待测 | 待测 |
| 平均输入 token | tokens/请求 | 待测 | 待测 | 待测 |
| 单次请求成本 | 元 | 待测 | 待测 | 待测 |

---

## 许可证

MIT
