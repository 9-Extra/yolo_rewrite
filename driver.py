import os
import atexit
from collections.abc import Callable
from typing import Optional, Literal

import torch

from dataset.h5Dataset import H5DatasetYolo
from schedules.schedule import Config
import preprocess
import train
import search

TargetFunction = Callable[[], None]

config = Config()


class _Context:
    state: dict[Literal["running", "completed"], set[str]]
    targets: dict[TargetFunction, "Target"]

    def __init__(self):
        self.targets = {}
        self._load_state()
        atexit.register(self._dump_state)

    def register_target(self, func: TargetFunction, target: "Target"):
        assert target.name not in self.targets
        self.targets[func] = target

    def _dump_state(self):
        os.makedirs(os.path.dirname(config.file_state_record), exist_ok=True)
        open(config.file_state_record, "w").write(repr(self.state))

    def _load_state(self):
        if os.path.isfile(config.file_state_record):
            self.state = eval(open(config.file_state_record, "r").read())
        else:
            self.state = {"running": set(), "completed": set()}

    def on_target_start(self, target: "Target"):
        self.state["running"].add(target.name)
        self._dump_state()

    def on_target_complete(self, target: "Target"):
        self.state["running"].remove(target.name)
        self.state["completed"].add(target.name)
        self._dump_state()


_context = _Context()


class Target:
    name: str
    dependence: list[TargetFunction]

    def __init__(self, dependence: Optional[list[TargetFunction]] = None, name: Optional[str] = None):
        self.dependence = [] if dependence is None else dependence
        self.name = name

    def __call__(self, func: TargetFunction):
        if self.name is None:
            self.name = func.__name__

        global _context
        _context.register_target(func, self)

        return func


def _dependency_resolution(target: TargetFunction) -> list[TargetFunction]:
    global _context
    visit_list = []
    visited = set()
    to_visit = [target]

    while len(to_visit) != 0:
        to_visit_next = []
        for target in to_visit:
            if target not in visited:
                visited.add(target)
                visit_list.append(target)
                to_visit_next.extend(_context.targets[target].dependence)
        to_visit = to_visit_next

    visit_list.reverse()  # 执行顺序与遍历顺序相反
    return visit_list


def run_target(target: TargetFunction):
    global _context
    execute = _dependency_resolution(target)

    for func in execute:
        target = _context.targets[func]
        if target.name not in _context.state["completed"]:
            _context.on_target_start(target)
            print(f"Running Target: {target.name}")
            func()
            _context.on_target_complete(target)
        else:
            print(f"Completed Target: {target.name}, Skip!")


def virtual_run_target(target: TargetFunction):
    global _context
    execute = _dependency_resolution(target)

    for func in execute:
        print(f"Run {_context.targets[func].name}")


@Target()
def target_preprocess_train_dataset():
    preprocess.main(config.file_train_dataset, config.raw_train_dataset)


@Target([target_preprocess_train_dataset])
def target_train():
    train.main(config)


@Target([target_train, target_preprocess_train_dataset])
def target_extract_features():
    network = config.trained_yolo_network
    network.to(config.device, non_blocking=True)

    train_dataset = H5DatasetYolo(config.file_train_dataset)  # 从训练集前向传播过程中抽取特征，用于训练OOD检测用
    search.extract_features(network, train_dataset, config.attack_method, config.mlp_epsilon, config.dir_extract_features)


@Target()
def target_preprocess_result_val_dataset():
    preprocess.main(config.file_detected_base_dataset, config.raw_detected_base_dataset)


@Target([target_train, target_preprocess_result_val_dataset])
def target_collect_result():
    network = config.trained_yolo_network
    network.to(config.device, non_blocking=True)

    val_dataset = H5DatasetYolo(config.file_detected_base_dataset)
    print("样本数：", len(val_dataset))
    result_dataset = search.collect_stats(network, val_dataset)
    result_dataset.save(config.file_detected_dataset)


@Target([target_collect_result, target_extract_features])
def target_search_layer():
    print("开始尝试每一层")
    search.search_single_layers(config.detected_result_dataset,
                                config.dir_extract_features,
                                config.file_single_layer_search_summary,
                                10,
                                config.device)


@Target([target_collect_result, target_extract_features])
def search_combine_layer():
    print("尝试策略")
    name_set_list = [*[{"backbone.inner.25.cv3.norm"} for _ in range(3)],
                     *[{"backbone.inner.25.cv3.norm", "backbone.inner.21.cv3.norm"} for _ in range(3)],
                     *[{"backbone.inner.25.cv3.norm", "backbone.inner.21.cv3.norm", "backbone.inner.29.cv3.norm"} for _
                       in range(3)],
                     ]
    search.search_multi_layers(name_set_list,
                               config.detected_result_dataset,
                               config.dir_extract_features,
                               config.file_multi_layer_search_summary,
                               15,
                               config.device)


@Target([target_collect_result, target_extract_features])
def train_mlp():
    name_set = {"backbone.inner.25.cv3.norm"}
    mlp_network, mlp_acc = (
        search.train_mlp_from_features_dir(name_set,
                                           config.detected_result_dataset,
                                           config.dir_extract_features, config.mlp_epoch, config.device))

    auroc, fpr95 = search.compute_auroc_fpr95(mlp_network, config.detected_result_dataset)
    print(f"{mlp_acc=:%} {auroc=} {fpr95=}")

    torch.save(mlp_network.to_static_dict(), config.file_mlp_weight)


if __name__ == '__main__':
    print("Start")
    run_target(train_mlp)
    print("Done!")
