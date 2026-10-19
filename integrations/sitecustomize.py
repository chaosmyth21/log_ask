"""LabMemory 采集钩子（可选安装到解释器 site-packages）。

放在 site-packages 里之后，**任何** python 进程启动都会执行到这里：
包括你在 IDE 里按 F5 调试、包括 `python train.py`、包括 pytest。

激活条件（只在满足时才动手）：
  当前工作目录或其上级（最多 7 层）存在 `.labmemory/` 目录。
也就是说：只有你主动在项目里跑过 `python main.py hook --install .` 的项目才会被采集，
别的 Python 程序完全不受影响。

这一切都包在 try/except 里：钩子绝不允许影响用户程序。
"""

import os
import sys


def _boot():
    try:
        home = os.environ.get("LABMEMORY_HOME")
        if home and home not in sys.path:
            sys.path.insert(0, home)          # 让 site-packages 里也能找到 labmemory
        from labmemory.capture import install_hook
        install_hook()
    except Exception:
        return


_boot()
