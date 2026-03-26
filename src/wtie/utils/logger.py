import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

"""轻量日志与终端输出工具。

该模块用于将运行事件写入文本日志文件，并提供简单的耗时展示、进度输出与标准输出刷新工具；
不负责日志分级管理、日志轮转、多进程/多线程并发安全或 `logging` 标准库兼容配置。

核心公开对象
------------
1. Logger
    面向文件的简单日志记录器。
2. display_time_from_seconds
    将秒数格式化为可读字符串。
3. simple_progressbar
    在终端打印单行进度百分比。
4. sysout
    不换行输出并立即刷新。

Examples
--------
>>> logger = Logger("./run.log", overwrite=True, ignore_git_info=True)
>>> logger.write("开始训练")
>>> display_time_from_seconds(3661)
'days:hours:min:sec - 0:1:1:1'
"""

# TODO: look at python logging module


##############################
# LOGGER
##############################


class Logger:
    """将文本事件写入日志文件，并可同步输出到终端。

    该类在首次创建日志文件时会写入初始化信息，包括系统信息、可选 Git 信息和 Conda 信息。
    日志目录必须已存在，否则会触发断言错误。

    Attributes
    ----------
    file : pathlib.Path
        日志文件路径对象。
    name : str
        日志文件名（不含目录）。
    directory : str
        日志文件所在目录（POSIX 风格字符串）。
    """

    def __init__(
        self, file_path: str, overwrite: bool = False, ignore_git_info: bool = True, hook_message: bool = False
    ):
        """初始化日志器并按需创建日志文件。

        Parameters
        ----------
        file_path : str
            日志文件路径。其父目录必须已存在。
        overwrite : bool, default=False
            若为 True 且日志文件已存在，则先删除旧文件再重新初始化。
        ignore_git_info : bool, default=True
            是否跳过 Git 仓库信息写入。
            为 False 时，会执行 `git remote show origin` 与 `git rev-parse HEAD`。
        hook_message : bool, default=False
            当日志文件已存在且未覆盖时，是否追加一条 "Hooking logger ..." 消息。

        Raises
        ------
        AssertionError
            当 `file_path` 的父目录不存在时触发。
        AttributeError
            在不支持 `os.uname()` 的平台上，首次创建日志文件时可能触发。
        OSError
            创建、删除或写入日志文件时可能触发。
        """

        self.file = Path(file_path)

        self.name = self.file.name
        self.directory = self.file.parent.as_posix()

        assert self.file.parent.is_dir(), "Directory %s should be created first." % self.directory
        # assert self.file.parent.exists()

        if self.file.exists() and overwrite:
            self.file.unlink()

        if not self.file.exists():
            # create file
            _datetime = str(datetime.now().strftime("%H:%M:%S on %A, %B the %dth, %Y"))
            self.file.write_text(("INITIALIZATION...\nCreation of the log at %s." % _datetime))

            # sys info
            # _system = str(os.uname())
            # self.write(("System information: %s" % _system))

            # git info
            if ignore_git_info:
                self.write("Git information: ignored")
            else:
                # Can be slow and requires to enter login credentials
                # everytime if user did not exchange ssh keys with the server.
                _git_info = subprocess.getoutput("git remote show origin")
                _git_rev = subprocess.getoutput("git rev-parse HEAD")
                self.write(("Git repository: \n%s" % _git_info))
                self.write(("current git commit revision number: %s\n") % _git_rev)

            # conda info
            _conda_info = subprocess.getoutput("conda info")
            self.write(("Conda information: %s") % _conda_info)

        else:
            _datetime = str(datetime.now().strftime("%H:%M:%S on %A, %B the %dth, %Y"))
            if hook_message:
                self.write(("Hooking logger at %s." % _datetime))

    def write(self, text: str, highlight: bool = False) -> None:
        """写入一条日志消息，并同步打印到标准输出。

        Parameters
        ----------
        text : str
            待写入的日志文本。
        highlight : bool, default=False
            若为 True，则终端输出使用青色加粗 ANSI 转义序列；写入文件的文本不带颜色。

        Returns
        -------
        None
            该方法原地执行，无返回值。

        Raises
        ------
        OSError
            打开或追加写入日志文件失败时触发。

        Examples
        --------
        >>> logger = Logger("./run.log", overwrite=True)
        >>> logger.write("epoch=1")
        >>> logger.write("warning", highlight=True)
        """
        if highlight:
            text_p = bcolors.BOLD + bcolors.OKCYAN + text + bcolors.ENDC
        else:
            text_p = text
        print(text_p)
        with self.file.open("a") as f:
            f.write("\n" + text)


class bcolors:
    """终端 ANSI 颜色与样式常量集合。"""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


##############################################
# UTILS
##############################################
def print_time_from_seconds(seconds: int):
    """将秒数转换为天/时/分/秒并直接打印。

    Parameters
    ----------
    seconds : int
        时长，单位为 s。内部会先执行 `int(seconds)`。

    Returns
    -------
    None
        该函数直接向标准输出打印两行文本，无返回值。

    Examples
    --------
    >>> print_time_from_seconds(3661)
    DAYS:HOURS:MIN:SEC
    0:1:1:1
    """
    sec = timedelta(seconds=int(seconds))
    d = datetime(1, 1, 1) + sec

    print("DAYS:HOURS:MIN:SEC")
    print("%d:%d:%d:%d" % (d.day - 1, d.hour, d.minute, d.second))


def display_time_from_seconds(seconds: int, lower: bool = True):
    """将秒数格式化为天/时/分/秒字符串。

    Parameters
    ----------
    seconds : int
        时长，单位为 s。内部会先执行 `int(seconds)`。
    lower : bool, default=True
        是否将返回字符串整体转为小写。

    Returns
    -------
    str
        形如 ``"DAYS:HOURS:MIN:SEC - d:h:m:s"`` 的格式化字符串。

    Examples
    --------
    >>> display_time_from_seconds(3661)
    'days:hours:min:sec - 0:1:1:1'
    >>> display_time_from_seconds(3661, lower=False)
    'DAYS:HOURS:MIN:SEC - 0:1:1:1'
    """
    sec = timedelta(seconds=int(seconds))
    d = datetime(1, 1, 1) + sec

    _str = "DAYS:HOURS:MIN:SEC - %d:%d:%d:%d" % (d.day - 1, d.hour, d.minute, d.second)

    if lower:
        _str = _str.lower()

    return _str


def simple_progressbar(count: int, total: int, refresh_rate: int = 1):
    """按给定刷新频率输出单行进度百分比。

    Parameters
    ----------
    count : int
        当前计数。
    total : int
        总计数。用于计算 `100 * (count / total)`。
    refresh_rate : int, default=1
        刷新步长。仅当 `count % refresh_rate == 0` 时输出。

    Returns
    -------
    None
        该函数直接写入标准输出并刷新，无返回值。

    Raises
    ------
    ZeroDivisionError
        当 `total == 0` 且满足刷新条件时触发。

    Examples
    --------
    >>> simple_progressbar(count=50, total=200, refresh_rate=10)
    """
    if count % refresh_rate == 0:
        perc = 100 * (count / total)
        sys.stdout.write("\r%.1f%% completed" % perc)
        sys.stdout.flush()


def sysout(message: str):
    """向标准输出写入消息并立即刷新，不自动换行。

    Parameters
    ----------
    message : str
        待输出消息。内部会执行 `str(message)` 后写出。

    Returns
    -------
    None
        该函数直接写入标准输出并刷新，无返回值。

    Examples
    --------
    >>> sysout("loading...")
    """
    sys.stdout.write(str(message))
    sys.stdout.flush()
