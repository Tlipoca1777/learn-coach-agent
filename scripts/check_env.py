r"""
环境自检脚本
================================================================

用途：一键检查你的开发环境是否正常。

**任何时候遇到"跑不起来"，第一件事就是运行它。**

    .\.venv\Scripts\python.exe scripts\check_env.py

它会依次检查 8 项，最后告诉你哪里有问题、怎么修。

--------------------------------------------------------------------
为什么要写这样一个脚本？
--------------------------------------------------------------------
因为初学者卡住的时候，最难的不是"解决问题"，
而是"根本不知道是哪里出了问题"。

是虚拟环境没激活？是包没装？是 Key 没配？是网络不通？
没有自检脚本，你只能瞎猜。
有了它，30 秒就能定位。

这个习惯在真实工作中叫"可观测性"，
一个成熟的项目一定有类似的 health check 接口。
"""

import sys
from pathlib import Path

# 让本脚本无论从哪里运行都能找到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 检查结果都记在这里，最后统一打印
results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    """记录一项检查结果，并立即打印。"""
    results.append((name, passed, detail))
    icon = "✅" if passed else "❌"
    print(f"{icon} {name}")
    if detail:
        # 缩进显示详情，多行的话每行都缩进
        for line in detail.splitlines():
            print(f"      {line}")
    return passed


# ====================================================================
# 检查 1：Python 版本
# ====================================================================
def check_python_version() -> bool:
    version = sys.version_info
    detail = f"Python {version.major}.{version.minor}.{version.micro}\n解释器：{sys.executable}"

    if version < (3, 10):
        return check(
            "Python 版本",
            False,
            detail + "\n本项目需要 Python 3.10 以上（因为用到了 X | Y 类型语法）。",
        )
    return check("Python 版本", True, detail)


# ====================================================================
# 检查 2：用的是不是虚拟环境
# ====================================================================
def check_venv() -> bool:
    """
    这是新手最容易踩的坑：用的是全局 Python，而不是 .venv 里的。
    表现是"我明明 pip install 了，怎么还是 ModuleNotFoundError"。
    """
    in_venv = sys.prefix != sys.base_prefix
    expected = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

    if not in_venv:
        return check(
            "虚拟环境",
            False,
            "当前用的不是虚拟环境！\n"
            "请改用这条命令运行：\n"
            r"    .\.venv\Scripts\python.exe scripts\check_env.py",
        )

    detail = f"虚拟环境位置：{sys.prefix}"
    if expected.exists():
        detail += "\n（正确，就是项目里的 .venv）"
    return check("虚拟环境", True, detail)


# ====================================================================
# 检查 3：依赖包是否装好
# ====================================================================
def check_dependencies() -> bool:
    required = ["openai", "dotenv", "rich", "pytest", "pydantic"]
    missing = []

    for module_name in required:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        return check(
            "依赖包",
            False,
            f"缺少：{', '.join(missing)}\n"
            "请运行：\n"
            r"    .\.venv\Scripts\python.exe -m pip install -r requirements.txt",
        )

    # 显示 openai 的版本，因为大版本升级可能带来不兼容
    import openai

    return check(
        "依赖包",
        True,
        f"openai / python-dotenv / rich / pytest / pydantic 都已安装\n"
        f"openai 版本：{getattr(openai, '__version__', '未知')}",
    )


# ====================================================================
# 检查 4：自己的包能不能 import
# ====================================================================
def check_own_package() -> bool:
    try:
        import memory_assistant
        from memory_assistant import Config, FakeLLM, estimate_tokens  # noqa: F401
    except ImportError as error:
        return check(
            "项目包导入",
            False,
            f"无法导入 memory_assistant：{error}\n"
            "可能原因：忘了做可编辑安装。请运行：\n"
            r"    .\.venv\Scripts\python.exe -m pip install -e . --no-deps",
        )

    count = estimate_tokens("你好，世界")
    return check(
        "项目包导入",
        True,
        f"memory_assistant 版本 {memory_assistant.__version__}\n"
        f"token 估算功能正常（'你好，世界' ≈ {count} tokens）",
    )


# ====================================================================
# 检查 5：存储层（SQLite）
# ====================================================================
def check_storage() -> bool:
    """
    检查数据库层。

    这里刻意只做"最关键的三件事"，不重复跑完整自检
    （完整版在 scripts/check_storage.py）：
        1. 表能不能建出来
        2. 外键有没有真的开启  ← SQLite 最著名的坑
        3. 级联删除有没有生效  ← 外键没开的话这里会挂
    """
    try:
        from memory_assistant.storage import Database, Store
    except ImportError as error:
        return check(
            "存储层",
            False,
            f"无法导入 storage 模块：{error}\n"
            "这通常说明包没装好，请运行：\n"
            r"    .\.venv\Scripts\python.exe -m pip install -e . --no-deps",
        )

    try:
        # 用内存数据库检查，不碰你的真实数据文件
        db = Database(":memory:")
        db.initialize()
    except Exception as error:
        return check("存储层", False, f"建表失败：{type(error).__name__}: {error}")

    problems = []
    if not db.foreign_keys_enabled():
        problems.append("外键约束没有开启（PRAGMA foreign_keys 没生效）")

    store = Store(db)
    sid = store.sessions.create()
    store.messages.append(sid, "user", "测试消息")
    store.delete_session(sid)
    if store.db.count("messages") != 0:
        problems.append("级联删除没生效：删掉会话之后消息还留着（会产生孤儿数据）")

    table_count = len(db.table_names())
    db.close()

    if problems:
        return check("存储层", False, "\n".join(problems))

    return check(
        "存储层",
        True,
        f"SQLite 正常，{table_count} 张表；外键约束与级联删除均已生效",
    )


# ====================================================================
# 检查 6：向量化层（长期记忆的基础）
# ====================================================================
def check_embeddings() -> bool:
    """
    检查向量化层。

    ⚠️ 这里**刻意不下载模型**（那要 90MB、20 多秒），
       只检查"配置对不对、依赖全不全、缓存有没有"。
       真正的下载和效果验证交给 scripts/check_embeddings.py。
    """
    try:
        from memory_assistant.config import Config
        from memory_assistant.embeddings import create_embeddings
    except ImportError as error:
        return check("向量化层", False, f"无法导入 embeddings 模块：{error}")

    # ---- 1. 假向量模型必须能用（这是测试和离线开发的基础）----
    try:
        fake = create_embeddings(fake=True, dimension=32)
        vector = fake.embed_query("测试")
    except Exception as error:
        return check("向量化层", False, f"假向量模型异常：{type(error).__name__}: {error}")

    if len(vector) != 32:
        return check("向量化层", False, f"假向量模型维度不对：期望 32，实际 {len(vector)}")

    # ---- 2. fastembed 装了吗 ----
    try:
        from fastembed import TextEmbedding
    except ImportError as error:
        return check(
            "向量化层",
            False,
            f"fastembed 没装：{error}\n"
            "请运行：\n"
            r"    .\.venv\Scripts\python.exe -m pip install -r requirements.txt",
        )

    # ---- 3. 配置的模型在支持列表里吗 ----
    config = Config.from_env(require_key=False)
    supported = {str(info.get("model", "")) for info in TextEmbedding.list_supported_models()}

    if config.embedding_model not in supported:
        return check(
            "向量化层",
            False,
            f"配置的模型 {config.embedding_model} 不在 fastembed 的支持列表里。\n"
            "中文场景建议用 BAAI/bge-small-zh-v1.5\n"
            "（改 .env 里的 EMBEDDING_MODEL 即可）",
        )

    # ---- 4. 模型缓存状态 ----
    cache_dir = config.embedding_cache_dir
    cached_files = list(cache_dir.rglob("*.onnx")) if cache_dir.exists() else []
    if cached_files:
        cache_status = "已就绪 ✅"
    else:
        cache_status = "尚未下载（首次使用时会自动下载约 90MB）"

    endpoint_text = config.hf_endpoint or "官方站（需要能直连 huggingface.co）"
    if config.hf_endpoint == "https://hf-mirror.com":
        endpoint_text += "（国内推荐）"

    return check(
        "向量化层",
        True,
        f"向量模型  ：{config.embedding_model}\n"
        f"HF 端点   ：{endpoint_text}\n"
        f"缓存目录  ：{cache_dir}\n"
        f"模型缓存  ：{cache_status}",
    )


# ====================================================================
# 检查 7：配置能不能加载
# ====================================================================
def check_config() -> tuple[bool, str]:
    """返回 (是否通过, api_key)。"""
    # ⚠️ 这里传 require_key=False，
    #    因为"没有 Key"是检查 6 的事，不该让配置加载本身失败。
    from memory_assistant import Config

    try:
        config = Config.from_env(require_key=False)
    except Exception as error:
        check("配置加载", False, f"{type(error).__name__}: {error}")
        return False, ""

    check("配置加载", True, config.describe())

    # 顺带检查 .env 文件是否存在
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("      ⚠️ 提示：没有找到 .env 文件，当前用的是默认值。")
        print("         请运行：Copy-Item .env.example .env")
    elif not config.api_key:
        print("      ⚠️ .env 文件存在，但 DEEPSEEK_API_KEY 是空的。")

    return True, config.api_key


# ====================================================================
# 检查 8：真实调用一次 API（可选）
# ====================================================================
def check_api_call(api_key: str) -> bool:
    if not api_key:
        return check(
            "真实 API 调用",
            False,
            "跳过：没有配置 API Key。\n"
            "请打开 .env 填入 DEEPSEEK_API_KEY 后再运行本脚本。\n"
            "申请地址：https://platform.deepseek.com/api_keys",
        )

    print("⏳ 正在调用 DeepSeek API（可能要几秒钟）...")

    from memory_assistant import Config, create_llm

    config = Config.from_env()

    try:
        llm = create_llm(config)
        # 用最简单的问题，把 max_tokens 压到最小，省钱
        answer = llm.chat(
            [{"role": "user", "content": "只回答数字：1+1 等于几？"}],
            max_tokens=16,
            temperature=0.0,
        )
    except Exception as error:
        error_name = type(error).__name__
        hint = ""

        # 根据错误类型给出针对性的建议 —— 这是这个脚本最有价值的地方
        if "Authentication" in error_name or "401" in str(error):
            hint = (
                "\n这通常是 API Key 的问题：\n"
                "  - Key 填错了，或者复制时多了空格 / 引号\n"
                "  - Key 已被删除或重置\n"
                "  - .env 里 DEEPSEEK_API_KEY= 后面要直接写 Key，不要加引号"
            )
        elif "InsufficientBalance" in str(error) or "402" in str(error):
            hint = "\n账户余额不足，请到 DeepSeek 平台充值。"
        elif "RateLimit" in error_name or "429" in str(error):
            hint = "\n请求过于频繁，稍等一会儿再试。"
        elif "Connection" in error_name or "Timeout" in error_name or "timeout" in str(error):
            hint = (
                "\n网络问题：\n"
                "  - 检查能否访问 https://api.deepseek.com\n"
                "  - 如果用了代理，确认代理配置正确"
            )

        return check("真实 API 调用", False, f"{error_name}: {error}{hint}")

    return check(
        "真实 API 调用",
        True,
        f"模型回答：{answer.strip()}\n本次用量：{llm.last_usage}",
    )


# ====================================================================
# 主流程
# ====================================================================
def main() -> int:
    print("=" * 64)
    print(" 环境自检")
    print("=" * 64)
    print()

    check_python_version()
    check_venv()

    # 后面几项依赖前面成功，失败就早点退出，避免报一堆无关的错
    if not check_dependencies():
        return finish()
    if not check_own_package():
        return finish()
    if not check_storage():
        return finish()
    if not check_embeddings():
        return finish()

    config_ok, api_key = check_config()
    if not config_ok:
        return finish()

    check_api_call(api_key)

    return finish()


def finish() -> int:
    """打印汇总，并返回退出码（0 = 全部通过）。"""
    print()
    print("=" * 64)

    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    if failed == 0:
        print(f" 全部通过（{passed}/{total}）—— 环境没问题，可以开始干活了！")
        print("=" * 64)
        return 0

    print(f" 结果：{passed}/{total} 项通过，{failed} 项有问题")
    print()
    print(" 有问题的项目：")
    for name, ok, _ in results:
        if not ok:
            print(f"   ❌ {name}")
    print()
    print(" 照着上面的提示修，修完再跑一次本脚本。")
    print("=" * 64)
    return 1


if __name__ == "__main__":
    sys.exit(main())
