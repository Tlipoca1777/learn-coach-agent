r"""
第 0 步：跑通你的第一次大模型调用
================================================================

这是整个项目的地基。

跑通这一个文件，你就理解了 90% 的 AI 应用在干什么：
    把一段文字（我们叫它 prompt / 提示词）
    通过 HTTP 发给大模型
    大模型返回一段文字。

后面所有的「记忆」「工具调用」「Agent 编排」，
本质上都只是在研究一件事：**怎么组织发过去的那段文字**。
所以这一步千万不要跳过。

--------------------------------------------------------------------
运行前准备（只需要做一次）
--------------------------------------------------------------------
1. 把 .env.example 复制成 .env
       Copy-Item .env.example .env
2. 用记事本 / VS Code 打开 .env，在 DEEPSEEK_API_KEY= 后面填上你的真实 Key
       地址：https://platform.deepseek.com/api_keys

--------------------------------------------------------------------
运行命令（在 D:\Agent开发 目录下执行）
--------------------------------------------------------------------
    .\.venv\Scripts\python.exe scripts\step0_hello_llm.py

预期：屏幕上打印出一段 AI 写的自我介绍，以及本次消耗的 token 数。
"""

# ====================================================================
# 第 1 部分：导入需要的工具（库）
# ====================================================================
# Python 的哲学是「不要重复造轮子」。
# 别人写好了一大堆功能，打包成「库」，我们用 import 拿过来用。

import os   # os = operating system，用来读环境变量
import sys  # sys = system，用来处理解释器相关的事情

# Windows 的命令行有时候会用 GBK 编码，打印中文会乱码或直接报错。
# 下面这三行强制把输出编码改成 UTF-8，是 Windows 上写中文程序的必备操作。
# hasattr 是「有没有这个属性」的意思，防止在某些环境下没有 reconfigure 而报错。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# from ... import ... 的意思是「从这个库里，只拿我需要的那个东西」
from dotenv import load_dotenv          # 从 .env 文件读取配置
from openai import OpenAI               # DeepSeek 兼容 OpenAI 的接口，所以能用这个官方库


# ====================================================================
# 第 2 部分：读取配置
# ====================================================================

# load_dotenv() 会去找当前目录（及其上层目录）里的 .env 文件，
# 把里面的 KEY=VALUE 一行行读出来，放进「环境变量」里。
# 好处：密钥不会出现在代码里，代码可以放心分享给别人 / 传到 GitHub。
load_dotenv()

# os.getenv("名字") 就是「去环境变量里把这个名字对应的值取出来」。
# 第二个参数是「如果没找到，就用这个默认值」。
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")          # 没默认值，没填就必须报错
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# -------- 友好的错误提示 --------
# 初学者最常见的问题就是「忘了填 Key」，所以我们提前检查，给出人话提示，
# 而不是让程序抛出一大堆看不懂的报错。
# `if not API_KEY` 的意思是「如果 API_KEY 是空的（空字符串、None 都算）」。
if not API_KEY:
    print("=" * 60)
    print("[错误] 没有找到 DEEPSEEK_API_KEY")
    print("=" * 60)
    print("请按下面两步操作：")
    print("  1. 在本目录执行：Copy-Item .env.example .env")
    print("  2. 打开 .env，把 DEEPSEEK_API_KEY= 后面填上你的真实 Key")
    print("     申请地址：https://platform.deepseek.com/api_keys")
    print()
    # sys.exit(1) 表示「程序到此为止，退出」，1 是一个约定俗成的「失败」代码。
    sys.exit(1)


# ====================================================================
# 第 3 部分：创建「客户端」
# ====================================================================
# 客户端（client）你可以理解成一个「打电话用的手机」。
# 创建一次，后面所有请求都用它来打。
# 注意：这里用的是「关键字参数」写法（api_key=..., base_url=...），
# 好处是一眼就能看出每个值是什么意思，不容易传错位置。

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ====================================================================
# 第 4 部分：组织消息（这是最关键的一步）
# ====================================================================
# 大模型的输入是一个「消息列表」，每个消息是一个字典（dict）。
# 字典的写法是 {"键": 值, "键": 值}，可以理解成一张"标签卡片"。
#
# 有三个角色（role）：
#   "system"    —— 系统提示词。设定 AI 的身份、性格、规矩。用户看不到。
#   "user"      —— 用户说的话。
#   "assistant" —— AI 之前说过的话。
#
# 为什么是「列表」而不是「一句话」？
# 因为大模型本身没有记忆！它是「无状态」的。
# 你每次都得把之前的对话全部重新发一遍，它才知道上下文。
# 这就是为什么后面我们要专门做「记忆管理」——
# 对话越长，这个列表越长，token 花得越多，最后会超出模型的上限。
#
# ⭐ 请记住这句话，它是你整个项目的核心动机：
#    大模型没有记忆，所谓记忆，都是我们自己把历史重新塞回去。

messages = [
    {
        "role": "system",
        "content": (
            "你是一位耐心的 Python 老师，擅长用生活化的比喻解释技术概念。"
            "回答请控制在 200 字以内。"
        ),
    },
    {
        "role": "user",
        "content": "用一句话解释什么是大模型的『上下文窗口』。",
    },
]


# ====================================================================
# 第 5 部分：发起请求
# ====================================================================
# try / except 是「异常处理」：试着做某件事，如果失败了就走 except 分支。
# 网络请求随时可能失败（断网、欠费、Key 过期），所以必须处理。

print("正在请求 DeepSeek...\n")

try:
    # create() 就是「发一次请求」，返回值里装着模型的回答。
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,   # 随机性：0 最稳定，1 最放飞
        max_tokens=MAX_TOKENS,     # 最多回答多长
    )
except Exception as error:
    # f"..." 是 f-string，可以把变量直接塞进字符串里，非常常用。
    print(f"[请求失败] {type(error).__name__}: {error}")
    print()
    print("常见原因：")
    print("  - API Key 填错了 / 已失效 / 余额不足")
    print("  - 网络无法访问 api.deepseek.com")
    sys.exit(1)


# ====================================================================
# 第 6 部分：把结果取出来打印
# ====================================================================
# response 是一个嵌套很深的对象。取值的路径是固定的，背下来就行：
#     response.choices[0].message.content
# 读法：从所有候选回答里取第 1 个 -> 取它的 message -> 取 message 的 content 文本
answer = response.choices[0].message.content

print("-" * 60)
print("AI 的回答：")
print("-" * 60)
print(answer)
print("-" * 60)


# ====================================================================
# 第 7 部分：看看花了多少钱（重要！养成成本意识）
# ====================================================================
# usage 里记录了本次消耗的 token 数。
# token 是模型计费的最小单位，大致规律：1 个汉字 ≈ 0.6~1 个 token，
# 1 个英文单词 ≈ 1.3 个 token。
#
# 为什么现在就要关心这个？
# 因为你的项目主题是「记忆」，而记忆最大的敌人就是 token 无限增长。
# 面试时如果你能说「我的记忆模块把每次请求的 token 控制在 X 以内」，
# 这就是实打实的工程能力，而不是只会调 API。

usage = response.usage
print()
print("本次消耗：")
print(f"  输入 prompt_tokens    : {usage.prompt_tokens}")
print(f"  输出 completion_tokens: {usage.completion_tokens}")
print(f"  合计 total_tokens     : {usage.total_tokens}")
print()
print("单价请到 https://api-docs.deepseek.com/zh-cn/quick_start/pricing 查看，")
print("用 单价 × 本次 token 数 / 1000000 就能算出这次花了多少钱。")
print()
print("[OK] 第 0 步完成！你已经跑通了 AI 应用的地基。")
