r"""
向量层自检
================================================================

检查向量模型能不能正常工作，并且**用真实数据验证中文语义检索有效**。

运行：
    .\.venv\Scripts\python.exe scripts\check_embeddings.py

首次运行会下载模型（约 90MB，走镜像大约 20~30 秒），之后就快了。

--------------------------------------------------------------------
这个脚本为什么值得单独存在？
--------------------------------------------------------------------
因为"模型能加载"和"模型能用"是两件事。

一个模型可以：
    · 正常加载，不报任何错
    · 维度也对，512 维
    · 但**检索结果完全是错的**（Chroma 默认的英文模型就是这样）

所以我在这里做的是**效果验证**，不只是冒烟测试：
用自然语言改写的方式提问，看正确的那条能不能排到第一。
如果不能，说明这个模型不能用于中文检索 —— 再往下做也是白做。

这就是 docs/记忆架构设计.md 第 12 节记录的实测方法的可执行版本。
"""

import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from memory_assistant.config import Config  # noqa: E402
from memory_assistant.embeddings import create_embeddings  # noqa: E402

# 候选语料：4 条互不相关的用户事实
CORPUS = [
    "我喜欢喝手冲咖啡，不加糖",
    "我养了一只橘猫叫豆豆",
    "我对花生过敏，点菜要注意",
    "相对论描述了时空的几何结构",
]

# 测试用例：(查询, 期望命中的句子, 说明)
# 关键在于这些查询都做了**语义改写**，和候选句几乎没有字面重合 ——
# 这正是英文模型会挂掉、中文模型才能过的场景。
CASES = [
    ("我平时喝什么咖啡？", "我喜欢喝手冲咖啡，不加糖", "语义改写：用『喝什么』问『喜欢喝』"),
    ("我家猫咪叫什么名字？", "我养了一只橘猫叫豆豆", "语义改写：『猫咪』vs『橘猫』"),
    ("我吃花生会怎么样？", "我对花生过敏，点菜要注意", "有字面重合（花生）"),
]

# 一条无关查询，用来检查"区分度"
IRRELEVANT_QUERY = "请解释一下深度学习中的注意力机制"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def main() -> int:
    print("=" * 70)
    print(" 向量层自检")
    print("=" * 70)
    print()

    # ---------- 1. 配置 ----------
    config = Config.from_env(require_key=False)
    cache_dir = config.embedding_cache_dir
    cached = list(cache_dir.rglob("*.onnx")) if cache_dir.exists() else []

    print(f"向量模型   : {config.embedding_model}")
    print(f"HF 端点    : {config.hf_endpoint or '官方站（需要能直连 huggingface.co）'}")
    print(f"缓存目录   : {cache_dir}")
    print(f"缓存状态   : {'已就绪' if cached else '尚未下载'}")
    print()

    if not cached:
        print("⏳ 首次运行需要下载模型（约 90MB）。")
        print("   如果卡在这里不动，多半是 HF 端点不通 —— ")
        print("   请确认 .env 里的 HF_ENDPOINT=https://hf-mirror.com")
        print()

    # ---------- 2. 加载模型 ----------
    print("正在加载向量模型...")
    t0 = time.time()
    try:
        embedder = create_embeddings(config)
    except Exception as error:
        print()
        print(f"❌ 加载失败：{type(error).__name__}: {error}")
        print()
        print("常见原因：")
        print("  · 网络不通，模型下载失败 → 检查 .env 里的 HF_ENDPOINT")
        print("  · 磁盘空间不足（需要约 100MB）")
        print("  · fastembed 没装 → pip install -r requirements.txt")
        return 1

    elapsed = time.time() - t0
    print(f"✅ 加载成功，耗时 {elapsed:.1f}s")
    print()

    # ---------- 3. 向量化 ----------
    t1 = time.time()
    vectors = embedder.embed_documents(CORPUS)
    embed_ms = (time.time() - t1) * 1000

    print(f"向量维度   : {embedder.dimension}")
    print(f"向量化性能 : {len(CORPUS)} 条耗时 {embed_ms:.1f}ms")
    print()

    # ---------- 4. 语义检索测试 ----------
    print("=" * 70)
    print(" 中文语义检索测试（这才是关键）")
    print("=" * 70)
    print()

    passed = 0
    relevant_scores: list[float] = []

    for query, expected, note in CASES:
        query_vector = embedder.embed_query(query)
        scored = sorted(
            ((cosine(query_vector, v), text) for v, text in zip(vectors, CORPUS)),
            reverse=True,
        )
        top_score, top_text = scored[0]
        ok = top_text == expected
        passed += int(ok)
        relevant_scores.append(top_score)

        print(f"{'✅' if ok else '❌'} {query}")
        print(f"   （{note}）")
        for rank, (score, text) in enumerate(scored, 1):
            mark = "  ← 命中" if text == expected else ""
            print(f"     {rank}. {score:.4f}  {text}{mark}")
        if not ok:
            print(f"   ⚠️ 期望命中『{expected}』，实际命中了『{top_text}』")
        print()

    # ---------- 5. 区分度测试 ----------
    print("=" * 70)
    print(" 区分度测试")
    print("=" * 70)
    print()

    irrelevant_vector = embedder.embed_query(IRRELEVANT_QUERY)
    irrelevant_top = max(cosine(irrelevant_vector, v) for v in vectors)
    min_relevant = min(relevant_scores)

    print(f"无关查询：{IRRELEVANT_QUERY}")
    print(f"  它的最高相似度      : {irrelevant_top:.4f}")
    print(f"  相关查询的最低相似度: {min_relevant:.4f}")

    has_margin = min_relevant > irrelevant_top
    if has_margin:
        print(f"  ✅ 相关 > 无关，差值 {min_relevant - irrelevant_top:+.4f}，有区分度")
    else:
        print(f"  ❌ 相关 ≤ 无关，说明模型无法区分 —— 检索会变成随机抽签")
    print()

    # ---------- 6. 判定 ----------
    print("=" * 70)
    print(" 判定")
    print("=" * 70)
    print()

    all_ok = passed == len(CASES) and has_margin

    if all_ok:
        print(f"  ✅ 全部通过（语义检索 {passed}/{len(CASES)}，区分度正常）")
        print()
        print("  向量层没问题，可以实现 memory/long_term.py 了。")
        print("=" * 70)
        return 0

    print(f"  ❌ 语义检索通过 {passed}/{len(CASES)}，区分度 {'正常' if has_margin else '异常'}")
    print()
    print("  排查建议：")
    print("    1. 确认 .env 里 EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5")
    print("       （换成英文模型就会复现『咖啡查询命中猫』这种结果）")
    print("    2. 删掉 data/models 让它重新下载")
    print("    3. 把上面的输出发给我")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())
