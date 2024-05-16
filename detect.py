import os
import typing

import numpy

import yolo
import torch

from preprocess import letterbox
import cv2

from yolo.non_max_suppression import non_max_suppression
from dataset import h5Dataset


def display(img: numpy.ndarray, objs, ood_scores, label_names):
    # objs: numpy.ndarray = objs.numpy(force=True)
    for obj, ood_score in zip(objs, ood_scores):
        bbox, conf, origin_bbox, layer_id, cls = numpy.split(obj, [4, 5, 9, 10])
        x1, y1, x2, y2 = [int(p) for p in bbox]
        cls = int(cls.item())
        conf = conf.item()
        # print(x1, y1, x2, y2, cls, conf)

        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.putText(img, label_names[cls] + f"{conf:%}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                          cv2.LINE_AA)
        img = cv2.putText(img, f"{ood_score.item():.3}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                           cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)


def detect(network: yolo.Network.NetWork,
           ood_evaluators,
           images: list[str],
           label_names,
           device: torch.device
           ):
    with torch.no_grad():
        network.eval().to(device, non_blocking=True)
        ood_evaluators.to(device, non_blocking=True)

        for img in images:

            ori_img = letterbox(cv2.imread(img), (640, 640))[0]
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV_FULL).transpose(2, 0, 1)[numpy.newaxis, ...]
            img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255

            extract_features = {}
            output = network(img, extract_features)
            output = network.detect.inference_post_process(output)
            # print(extract_features.keys())

            output = non_max_suppression(output)

            scores = ood_evaluators.score(extract_features, output)

            for img, output, score in zip([ori_img], output, scores):
                # if output.shape[0] != 0:
                display(ori_img, output, score, label_names)
    pass


def main(weight_path, img_dir):
    device = torch.device("cuda")

    network, ood_evaluators, label_names = yolo.Network.load_network(weight_path, load_ood_evaluator=True)

    network.eval().to(device, non_blocking=True)

    images = [os.path.join(img_dir, x) for x in os.listdir(img_dir)]

    detect(network, ood_evaluators, images, label_names, device)

    pass


if __name__ == '__main__':
    main("weight/yolo_final_full_20.pth", r"G:\datasets\BirdVsDrone\Drones")
