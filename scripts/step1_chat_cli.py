r"""
第 1 步：一个能对话的命令行助手（带流式输出 + 消息历史）
================================================================

第 0 步是「问一句，答一句」。
第 1 步我们要加两个真实产品必备的能力：

  【新能力 1】多轮对话
      把每一轮问答都追加进 messages 列表，下一轮再整个发回去。
      这就是「记忆」最原始、最粗糙的形态。
      ⭐ 跑起来后请亲自体会它的两个致命问题：
         (1) 关掉程序再打开，它就把你忘了 —— 记忆没有持久化。
         (2) 聊久了 messages 越来越长，token 越来越多 —— 迟早撑爆上下文。
      这两个问题，就是你这个项目后面要解决的。现在先亲眼看到它们。

  【新能力 2】流式输出（streaming）
      不用等模型把整段话写完，而是写好一个字就吐一个字。
      这是所有 AI 聊天产品的用户体验基础。
      实现方式：把 stream=True，然后 for 循环去读一个个「碎片」。

--------------------------------------------------------------------
运行命令（在 D:\Agent开发 目录下）
--------------------------------------------------------------------
    .\.venv\Scripts\python.exe scripts\step1_chat_cli.py

内置命令（在对话中输入）：
    /exit    退出
    /reset   清空历史（体验一下"失忆"）
    /history 查看当前发出去的完整消息列表（重点看它怎么变长）
    /tokens  查看累计消耗的 token
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
if not API_KEY:
    print("[错误] 没有找到 DEEPSEEK_API_KEY，请先配置 .env 文件。")
    print("       参考 scripts/step0_hello_llm.py 顶部的说明。")
    sys.exit(1)

client = OpenAI(
    api_key=API_KEY,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("LLM_MODEL", "deepseek-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))


# ====================================================================
# 消息列表：整个程序的「记忆」
# ====================================================================
# 这里我们把它单独拎出来，因为「AI 的记忆 = 这个列表的内容」。
# 后面的项目里，我们会把它换成一个真正的记忆模块。
messages = [
    {
        "role": "system",
        "content": "你是一个简洁、友好的中文助手。回答尽量控制在 150 字以内。",
    }
]

# 累计用量统计
total_prompt_tokens = 0
total_completion_tokens = 0


def ask_stream(user_input: str) -> str:
    """
    发一次请求，流式打印回答，并返回完整回答文本。

    关于函数（def）：
      - def 是 define 的缩写，用来定义函数。
      - 括号里的 user_input: str 是「参数」，: str 是类型提示（告诉别人这是字符串）。
      - -> str 是「返回值类型提示」：这个函数最后会返回一个字符串。
      - 类型提示不会影响运行，它的作用是让你和编辑器都能看懂代码，强烈建议养成习惯。
      - 函数最后一行的 return 决定返回值。
    """

    # 把用户这句话追加到历史里。
    # .append(...) 是列表的方法，作用是「在末尾添加一个元素」。
    messages.append({"role": "user", "content": user_input})

    # stream=True：不要一次性给我全部结果，而是一小块一小块给我
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    # 用一个字符串变量，把吐出来的碎片一点点拼起来
    answer_parts = []

    # for ... in ... 是循环。
    # 这里的 stream 是一个「迭代器」，每循环一次就拿到一个小碎片。
    for chunk in stream:
        # 有个坑：最后一个 chunk 只带用量信息，它的 choices 是空列表。
        # 所以我们先判断一下，避免下标越界报错。
        if not chunk.choices:
            # 有些兼容接口会把用量放在最后一个 chunk 的 usage 字段里
            if getattr(chunk, "usage", None):
                usage = chunk.usage
                # global 表示「我要修改的是外面那个全局变量」，不是新建局部变量
                global total_prompt_tokens, total_completion_tokens
                total_prompt_tokens += usage.prompt_tokens
                total_completion_tokens += usage.completion_tokens
            continue

        # 取出这个碎片里的文字
        piece = chunk.choices[0].delta.content
        # 有时候碎片是空的（比如只带了角色信息），要过滤掉
        if piece:
            answer_parts.append(piece)
            # end="" 表示打印完不换行；flush=True 表示立刻显示，不要缓冲
            print(piece, end="", flush=True)

    # "".join(列表) 把字符串列表拼成一个完整字符串，这是 Python 里的惯用写法
    answer = "".join(answer_parts)

    # 把 AI 的回答也追加进历史，这样下一轮它才知道自己刚才说过什么
    messages.append({"role": "assistant", "content": answer})

    return answer


def main() -> None:
    """程序主入口。习惯上我们把主要流程放在 main() 里。"""
    print("=" * 64)
    print(" 有记忆的私人助理 · 第 1 步：能对话的命令行助手")
    print("=" * 64)
    print("命令：/exit 退出　/reset 清空历史　/history 看消息列表　/tokens 看用量")
    print()

    while True:
        # input() 会暂停程序，等用户输入一行文字并回车，返回输入的内容。
        # .strip() 去掉首尾的空格，防止用户手滑多打了空格。
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            # 用户按了 Ctrl+C 或 Ctrl+Z
            print("\n再见！")
            break

        # not user_input 等价于「是空字符串」，直接跳过这轮
        if not user_input:
            continue

        # -------- 处理内置命令 --------
        if user_input == "/exit":
            print("再见！")
            break

        if user_input == "/reset":
            # 小心！这里要让列表「原地清空」，而不是重新赋值。
            # 用 messages.clear()，不用 messages = [...]。
            # 因为我们已经在函数里引用了这个列表对象，重新赋值会让引用失效。
            messages.clear()
            messages.append(
                {"role": "system", "content": "你是一个简洁、友好的中文助手。"}
            )
            print("[已清空历史，现在它不认识你了 —— 试着问它『我刚才说了什么』]\n")
            continue

        if user_input == "/history":
            print(f"\n[当前消息列表，共 {len(messages)} 条]")
            for index, message in enumerate(messages):
                # 只显示前 60 个字符，避免刷屏
                preview = message["content"][:60].replace("\n", " ")
                print(f"  {index}. [{message['role']}] {preview}...")
            print()
            continue

        if user_input == "/tokens":
            print(
                f"\n[累计用量] 输入 {total_prompt_tokens} / 输出 {total_completion_tokens}"
                f" / 合计 {total_prompt_tokens + total_completion_tokens} tokens\n"
            )
            continue

        # -------- 正常对话 --------
        print("AI > ", end="", flush=True)
        try:
            ask_stream(user_input)
        except Exception as error:
            print(f"\n[请求失败] {type(error).__name__}: {error}\n")
            # 请求失败时，把刚加进去的用户消息撤掉，保持历史干净
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            continue
        print("\n")


# ====================================================================
# 这个 if 是一个约定俗成的写法，意思是：
#   「只有直接运行这个文件时才执行 main()；
#     如果它被别人 import 走，就不要自动执行。」
# 记住它就行，几乎每个 Python 项目的入口文件都有这两行。
# ====================================================================
if __name__ == "__main__":
    main()
