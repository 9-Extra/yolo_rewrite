import argparse
import sys
from typing import Sequence

from scheduler import Target
from config import Config
import scheduler

config = Config()


@Target()
def _yolo_train():
    import yolo_train
    yolo_train.train(config)


@Target(_yolo_train)
def _yolo_val():
    import yolo_train
    yolo_train.val(config)


@Target(_yolo_train)
def _safe_val():
    import safe_val
    data_paths = [
        "run/preprocess/drone_train.h5",
        "run/preprocess/drone_val.h5",
        "run/preprocess/drone_test.h5",
        "run/preprocess/drone_test_with_bird.h5",
        "run/preprocess/drone_test_with_coco.h5"
    ]
    safe_val.main(config, data_paths)


@Target(_yolo_train)
def _vos_finetune():
    import vos_finetune
    vos_finetune.vos_finetune_val(config)


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

import search # 导入以注册search相关target

def _run_targets(names: Sequence[str]):
    targets: list[Target] = []
    unknown_targets = []
    for name in names:
        t = scheduler.get_target_by_name(name)
        if t is not None:
            targets.append(t)
        else:
            unknown_targets.append(name)

    if len(unknown_targets) != 0:
        raise RuntimeError(f"存在未知目标: {unknown_targets}")

    print(f"Run targets {names}")
    with Inhibitor():
        for t in targets:
            scheduler.run_target(t.func)

    print("Done!")


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

    if args.command == 'list':
        for name in scheduler.get_target_names():
            print(name)
    elif args.command == 'run':
        if not args.items:
            print('错误：run 命令需要至少一个项目名称。')
        else:
            _run_targets(args.items)
    else:
        parser.print_help()
