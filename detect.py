import os

import numpy

import yolo
import torch

from preprocess import letterbox
import cv2

from yolo.non_max_suppression import non_max_suppression
from dataset import h5Dataset


def display(img: numpy.ndarray, objs, label_names):
    for obj in objs:
        x1, y1, x2, y2, conf, ood_score, cls = [x.item() for x in obj.cpu()]
        x1, y1, x2, y2, cls = round(x1), round(y1), round(x2), round(y2), int(cls)
        # print(x1, y1, x2, y2, cls, conf)

        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.putText(img, label_names[cls] + f"{conf:%}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                          cv2.LINE_AA)
        img = cv2.putText(img, f"{ood_score}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                          cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)


def detect(network: yolo.Network.NetWork, images: list[str], label_names, device: torch.device):
    with torch.no_grad():
        network.eval().to(device, non_blocking=True)
        network.detect.output_odd_feature = True

        for img in images:

            ori_img = letterbox(cv2.imread(img), (640, 640))[0]
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV_FULL).transpose(2, 0, 1)[numpy.newaxis, ...]
            img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255

            output, ood_feature = network(img)
            output = network.detect.inference_post_process(output, ood_feature)

            output = non_max_suppression(output)

            display(ori_img, output, label_names)
    pass


def main(weight_dir, img_dir):
    device = torch.device("cuda")

    state_dict: dict = torch.load(weight_dir)
    num_class = state_dict["num_class"]
    network = yolo.Network.NetWork(num_class)
    network.load_state_dict(state_dict["network"])

    print(f"成功从{os.path.abspath(weight_dir)}加载模型权重")
    if "label_names" in state_dict:
        label_names = state_dict["label_names"]
        print("获取标签名称：", label_names)
    else:
        print("获取标签失败，自动生成标签")
        label_names = list(str(i + 1) for i in range(num_class))

    network.eval().to(device, non_blocking=True)

    images = [os.path.join(img_dir, x) for x in os.listdir(img_dir)]

    detect(network, images, label_names, device)

    pass


if __name__ == '__main__':
    main("weight/yolo_final.pth", r"G:\datasets\BirdVsDrone\Drones")
