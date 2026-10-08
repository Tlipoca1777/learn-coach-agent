r"""
让 `python -m memory_assistant` 能直接启动。

Python 的规矩：一个包想要支持 `python -m 包名` 运行，
就必须在包里放一个名叫 `__main__.py` 的文件。
解释器会把它的内容当成主程序执行。

所以这个文件只做一件事：把控制权交给 cli.main()。
真正的逻辑都在 cli.py 和 engine.py 里。
"""

import sys

from memory_assistant.cli import main

if __name__ == "__main__":
    sys.exit(main())
