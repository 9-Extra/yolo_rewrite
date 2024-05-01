import os

import numpy

import yolo
import torch

from preprocess import letterbox
import cv2

from yolo.non_max_suppression import non_max_suppression


def display(img: numpy.ndarray, objs, label_names):
    for obj in objs:
        x1, y1, x2, y2, conf, cls = [x.item() for x in obj.cpu()]
        x1, y1, x2, y2, cls = round(x1), round(y1), round(x2), round(y2), int(cls)
        # print(x1, y1, x2, y2, cls, conf)

        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.putText(img, label_names[cls] + f"{conf:%}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                          cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)


def detect(network: yolo.Network.NetWork, images: list[str], label_names, device: torch.device):
    with torch.no_grad():
        network.eval().to(device, non_blocking=True)
        network.detect.output_odd_feature = True

        for img in images:
            # ori_img = cv2.imread(img) / 255
            # cv2.imshow('image', ori_img)
            # cv2.imshow('diff1', numpy.fabs(numpy.diff(ori_img, axis=0, append=0)))
            # cv2.imshow('diff2', numpy.fabs(numpy.diff(ori_img, axis=1, append=0)))
            # cv2.waitKey(0)

            ori_img, ratio, (top, left) = letterbox(cv2.imread(img), (640, 640))
            h, w, _ = ori_img.shape
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[numpy.newaxis, ...]
            img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255

            output, ood_feature = network(img)
            output = network.detect.inference_post_process(output)

            output = non_max_suppression(output)

            display(ori_img, output, label_names)
    pass


def main(weight_dir, img_dir):
    device = torch.device("cuda")

    network = yolo.Network.NetWork(1)
    network.load_state_dict(torch.load(weight_dir))
    network.eval().to(device, non_blocking=True)

    label_names = ["drone", "bird"]

    detect(network, [os.path.join(img_dir, x) for x in os.listdir(img_dir)], label_names, device)

    pass


if __name__ == '__main__':
    main("weight/yolo_2.pth", r"G:\datasets\BirdVsDrone\Drones")
