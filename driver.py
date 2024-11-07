import argparse
import sys

class Inhibitor:
    """
    Prevent OS sleep/hibernate in windows
    """
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        pass

    def __enter__(self):
        if sys.platform == "win32":
            import ctypes
            print("Preventing Windows from going to sleep")
            ctypes.windll.kernel32.SetThreadExecutionState(
                Inhibitor.ES_CONTINUOUS | \
                Inhibitor.ES_SYSTEM_REQUIRED)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.platform == "win32":
            import ctypes
            print("Allowing Windows to go to sleep")
            ctypes.windll.kernel32.SetThreadExecutionState(
                Inhibitor.ES_CONTINUOUS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("调用一些乱七八糟的方法")

    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='列出目标或执行目标')

    # 创建 list 命令的解析器
    parser_list = subparsers.add_parser('list', help='列出所有项目')

    # 创建 run 命令的解析器
    parser_run = subparsers.add_parser('run', help='运行项目')
    parser_run.add_argument('items', metavar='item', type=str, nargs='+',
                            help='要运行的项目')

    # 解析命令行参数
    args = parser.parse_args()
    
    # todo
