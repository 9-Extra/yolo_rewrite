import os

import numpy

import utils
import yolo
import torch
import cv2

from dataset.DroneDataset import DroneTestDataset
from dataset.RawDataset import RawDataset
from safe.safe_method import MLP
from safe.FeatureExtract import FeatureExtract
from yolo.non_max_suppression import non_max_suppression


def display(img: numpy.ndarray, objs, is_ood: numpy.ndarray, label_names):
    # objs: numpy.ndarray = objs.numpy(force=True)
    for obj, ood in zip(objs, is_ood):
        bbox, conf, origin_bbox, layer_id, cls = numpy.split(obj, [4, 5, 9, 10])
        x1, y1, x2, y2 = [int(p) for p in bbox]
        cls = int(cls.item())
        conf = conf.item()
        # print(x1, y1, x2, y2, cls, conf)

        if not ood:
            color = (0, 255, 0)
            text = f"{label_names[cls]}:{conf:%}"
        else:
            color = (255, 0, 0)
            text = "OOD"

        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        img = cv2.putText(img, text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                          color,
                          2,
                          cv2.LINE_AA)

    cv2.imshow('image', img)
    cv2.waitKey(0)


@torch.no_grad()
def detect(network: yolo.Network.Yolo,
           ood_evaluator: MLP,
           images: list[str],
           label_names,
           threshold,
           device: torch.device
           ):
    extractor = FeatureExtract(ood_evaluator.feature_name_set)
    extractor.attach(network)
    network.eval().to(device, non_blocking=True)
    ood_evaluator.eval().to(device, non_blocking=True)

    for img in images:

        ori_img = utils.letterbox_fixed_size(cv2.imread(img), (640, 640))[0]
        # ori_img = utils.letterbox_stride(cv2.imread(img), 32)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV_FULL).transpose(2, 0, 1)[numpy.newaxis, ...]
        img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255

        extractor.ready()
        output = network(img)
        output = network.detect.inference_post_process(output)

        output = non_max_suppression(output)

        scores = ood_evaluator.score(extractor.get_features(), output)

        for img, output, score in zip([ori_img], output, scores):
            # if output.shape[0] != 0:

            is_ood = score < threshold
            display(img, output, is_ood, label_names)

    extractor.detach()

    pass


def main(weight_path, raw_dataset: RawDataset):
    device = torch.device("cuda")

    network, _, label_names = utils.load_network(weight_path, load_ood_evaluator=False)
    ood_evaluator = MLP.from_static_dict(torch.load("mlp.pth"))
    ood_evaluator.eval().to(device, non_blocking=True)
    network.eval().to(device, non_blocking=True)

    images = []
    for item in raw_dataset.items:
        if len(item.objs) != 0:
            images.append(item.img)

    threshold = 0.5

    detect(network, ood_evaluator, images, label_names, threshold, device)

    pass


if __name__ == '__main__':
    # main("weight/yolo_final_full.pth", r"G:\datasets\BirdVsDrone\Drones")
    main("weight/yolo_final_full.pth", DroneTestDataset(r"G:\datasets\DroneTestDataset"))
