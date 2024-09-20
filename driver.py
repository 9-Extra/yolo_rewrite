import argparse
import os.path
import sys
from typing import Sequence

from safe import safe_method
from safe.safe_method import ExtractFeatureDatabase
import safe.safe_method
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


@Target(_yolo_train)
def target_collect_result():
    if os.path.exists(config.file_detected_dataset):
        return

    from dataset.h5Dataset import H5DatasetYolo
    import search

    network = config.trained_yolo_network
    network.to(config.device, non_blocking=True)

    val_dataset = H5DatasetYolo(config.file_detected_base_dataset)
    print("样本数：", len(val_dataset))
    result_dataset = search.collect_stats_and_feature(network, val_dataset)
    result_dataset.save(config.file_detected_dataset)


@Target(target_collect_result)
def target_search_layer():
    from safe.attack import FSGMAttack, PDGAttack
    from dataset.h5Dataset import H5DatasetYolo
    import search
    import safe

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

    safe_method.extract_features_h5(network, feature_name_set, train_dataset, attackers,
                                         config.h5_extract_features)

    name_set_list = [{layer_name} for layer_name in feature_name_set]  # 所有的单层
    for attacker in attackers:
        search.search_layers(
            name_set_list,
            config.detected_result_dataset,
            ExtractFeatureDatabase(config.h5_extract_features),
            attacker.name,
            f"run/summary/single_layer_search_{attacker.name}.csv",
            15,
            config.device
        )


@Target(target_collect_result)
def search_combine_layer():
    from safe.attack import FSGMAttack, PDGAttack
    import search

    print("尝试策略")

    attackers = [FSGMAttack(e) for e in (0.05, 0.06, 0.07, 0.08, 0.09, 0.1)]

    name_set_list = [*[{"backbone.inner.25"} for _ in range(3)],
                     *[{"backbone.inner.25", "backbone.inner.21"} for _ in range(3)],
                     *[{"backbone.inner.25", "backbone.inner.21", "backbone.inner.23.norm"} for _
                       in range(3)],
                     ]

    for attacker in attackers:
        search.search_layers(
            name_set_list,
            config.detected_result_dataset,
            ExtractFeatureDatabase(config.h5_extract_features),
            attacker.name,
            f"run/summary/multi_layer_search_{attacker.name}.csv",
            15,
            config.device
        )


@Target(target_collect_result)
def mult_epsilon_compare():
    from safe.attack import FSGMAttack, PDGAttack
    from dataset.h5Dataset import H5DatasetYolo
    import safe, search

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
    
    safe_method.extract_features_h5(network, feature_name_set, train_dataset, attackers,
                                         config.h5_extract_features)

    for attacker in attackers:
        @Target(name=f"mult_epsilon_compare.val_{attacker.name}")
        def val():
            search.search_layers(name_set_list,
                                 config.detected_result_dataset,
                                 ExtractFeatureDatabase(config.h5_extract_features),
                                 attacker.name,
                                 f"run/summary/multi_layer_search_{attacker.name}.csv",
                                 30,
                                 config.device)

        scheduler.run_target(val)


class WindowsInhibitor:
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
    for name in names:
        t = scheduler.get_target_by_name(name)
        if t is not None:
            targets.append(t)
        else:
            unknown_targets.append(name)

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
    else:
        parser.print_help()
