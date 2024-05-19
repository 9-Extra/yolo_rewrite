import os
import typing

import torch

from yolo.Network import Yolo
from safe.safe_method import MLP


def load_network(weight_path: str, load_ood_evaluator=False) \
        -> tuple[Yolo, typing.Any, typing.Optional[list[str]]]:
    state_dict: dict = torch.load(weight_path)
    num_class = state_dict["num_class"]
    network = Yolo(num_class)
    network.load_state_dict(state_dict["network"])

    print(f"成功从{os.path.abspath(weight_path)}加载模型权重")
    ood_evaluator = None
    if load_ood_evaluator:
        try:
            ood_evaluator = MLP.from_static_dict(state_dict["ood_evaluator"])
            print("成功加载OOD计算模块")
        except ValueError:
            raise RuntimeError("ood_evaluator信息不存在")

    if "label_names" in state_dict:
        label_names = state_dict["label_names"]
        print("获取标签名称：", label_names)
    else:
        print("获取标签失败，自动生成标签")
        label_names = list(str(i + 1) for i in range(num_class))

    return network, ood_evaluator, label_names


def load_checkpoint(weight_path: str, device: torch.device) -> tuple[Yolo, torch.optim.Optimizer]:
    state_dict = torch.load(weight_path, map_location=device)
    num_class = state_dict["num_class"]
    # start_epoch = state_dict["epoch"]
    network = Yolo(num_class)
    network.to(device)
    network.load_state_dict(state_dict["network"])
    opt = torch.optim.Adam(network.parameters())
    opt.load_state_dict(state_dict["optimizer"])

    return network, opt