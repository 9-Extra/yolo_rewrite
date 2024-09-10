import argparse
import sys
from typing import Sequence

import safe
from dataset.h5Dataset import H5DatasetYolo
from safe.attack import FSGMAttack, PDGAttack
from scheduler import Target
from schedules.schedule import Config
import yolo_train
import search
import scheduler

config = Config()


@Target()
def target_train():
    yolo_train.main(config)


@Target(target_train)
def target_collect_result():
    network = config.trained_yolo_network
    network.to(config.device, non_blocking=True)

    val_dataset = H5DatasetYolo(config.file_detected_base_dataset)
    print("样本数：", len(val_dataset))
    result_dataset = search.collect_stats_and_feature(network, val_dataset)
    result_dataset.save(config.file_detected_dataset)


@Target(target_collect_result)
def target_search_layer():
    print("开始尝试每一层")

    network = config.trained_yolo_network
    network.to(config.device, non_blocking=True)

    attackers = [
        FSGMAttack(0.08),
        PDGAttack(0.006, 20),
        # *(PDGAttack(e, 10) for e in (0.005, 0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1))
    ]

    train_dataset = H5DatasetYolo(config.file_train_dataset)  # 从训练集前向传播过程中抽取特征，用于训练OOD检测用

    feature_name_set = search.ExtractAll().get_name_set(network)
    print(f"共提取{len(feature_name_set)}层的特征")

    safe.safe_method.extract_features_h5(network, feature_name_set, train_dataset, attackers,
                                         config.h5_extract_features)

    for attacker in attackers:
        search.search_single_layers(config.detected_result_dataset,
                                    config.extract_features_database,
                                    attacker.name,
                                    f"run/summary/single_layer_search_{attacker.name}.csv",
                                    15,
                                    config.device)


@Target(target_collect_result)
def search_combine_layer():
    print("尝试策略")

    attackers = [FSGMAttack(e) for e in (0.05, 0.06, 0.07, 0.08, 0.09, 0.1)]

    name_set_list = [*[{"backbone.inner.25"} for _ in range(3)],
                     *[{"backbone.inner.25", "backbone.inner.21"} for _ in range(3)],
                     *[{"backbone.inner.25", "backbone.inner.21", "backbone.inner.23.norm"} for _
                       in range(3)],
                     ]

    for attacker in attackers:
        search.search_multi_layers(
            name_set_list,
            config.detected_result_dataset,
            config.extract_features_database,
            attacker.name,
            f"run/summary/multi_layer_search_{attacker.name}.csv",
            15,
            config.device
        )


@Target(target_collect_result)
def mult_epsilon_compare():
    network = config.trained_yolo_network
    network.to(config.device, non_blocking=True)

    train_dataset = H5DatasetYolo(config.file_train_dataset)

    feature_name_set = {"backbone.inner.25"}
    name_set_list = [feature_name_set] * 3

    attackers = [
        # (FSGMAttack(e) for e in (0.05, 0.06, 0.07, 0.08, 0.09, 0.1)),
        # *(PDGAttack(e, 20) for e in (0.005, 0.006, 0.007)),
        *(PDGAttack(e, 10) for e in (0.005, 0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1))
    ]


    safe.safe_method.extract_features_h5(network, feature_name_set, train_dataset, attackers,
                                             config.h5_extract_features)

    for attacker in attackers:
        @Target(name=f"mult_epsilon_compare.val_{attacker.name}")
        def val():
            search.search_multi_layers(name_set_list,
                                       config.detected_result_dataset,
                                       config.extract_features_database,
                                       attacker.name,
                                       f"run/summary/multi_layer_search_{attacker.name}.csv",
                                       30,
                                       config.device)

        scheduler.run_target(val)


# def configure_inject(binder: inject.Binder):
#     binder.bind_to_constructor(yolo.Network.Yolo, config.trained_yolo_network)
#     binder.bind(AttackMethod, config.attack_method)


class WindowsInhibitor:
    """Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx
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
                WindowsInhibitor.ES_CONTINUOUS | \
                WindowsInhibitor.ES_SYSTEM_REQUIRED)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.platform == "win32":
            import ctypes
            print("Allowing Windows to go to sleep")
            ctypes.windll.kernel32.SetThreadExecutionState(
                WindowsInhibitor.ES_CONTINUOUS)


def _run_targets(names: Sequence[str]):
    targets = []
    unknown_targets = []
    for name in args.items:
        target_func = scheduler.get_target_by_name(name)
        if target_func is None:
            unknown_targets.append(name)
        else:
            targets.append(target_func)

    if len(unknown_targets) != 0:
        raise RuntimeError(f"存在未知目标: {unknown_targets}")

    print(f"Run targets {names}")
    with WindowsInhibitor():
        for t in targets:
            scheduler.run_target(t)

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
