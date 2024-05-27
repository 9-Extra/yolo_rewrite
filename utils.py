import os
import typing

import cv2
import numpy
import torch

from yolo.Network import Yolo
from safe.safe_method import MLP


def letterbox_fixed_size(im, new_shape: tuple[int, int], color=(114, 114, 114), scaleup=False):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    height, width = im.shape[:2]  # current shape [height, width]
    # Scale ratio (new / old)
    r = min(new_shape[0] / height, new_shape[1] / width)
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = round(width * r), round(height * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    ratio = new_unpad[0] / width, new_unpad[1] / height

    if (width, height) != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh // 2, dh - (dh // 2)  # divide padding into 2 parts
    left, right = dw // 2, dw - (dw // 2)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (top, left)


def letterbox_stride(im: numpy.ndarray, stride: int = 32, color=(114, 114, 114)):
    height, width, channel = im.shape
    target_width = width + stride - 1 - (width - 1) % stride
    target_height = height + stride - 1 - (height - 1) % stride
    if target_width == width and target_height == height:
        return im  # skip

    result = numpy.full_like(im, color, shape=(target_height, target_width, channel), subok=False)
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    result[top:top + height, left: left + width] = im

    return result


def crop_image(img: numpy.ndarray, bbox, size: tuple[int, int]):
    x, y, w, h = bbox
    img = img[y:y + h, x:x + w, :]
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img


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
