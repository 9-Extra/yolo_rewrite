import os

import numpy
import torchvision

import yolo
import torch

from dataset.h5Dataset import ObjectRecord
from preprocess import letterbox
import cv2


def display(img: numpy.ndarray, objs, label_names):
    for obj in objs:
        x1, y1, x2, y2, conf, cls = [x.item() for x in obj.cpu()]
        x1, y1, x2, y2, cls = round(x1), round(y1), round(x2), round(y2), int(cls)
        print(x1, y1, x2, y2, cls, conf)

        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.putText(img, label_names[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                          cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)


def box_iou(box1, box2):
    # box1 = [x1, y1, x2, y2, area]
    # box2 = [x1, y1, x2, y2, area]
    # 右边界点中小的一个是重叠部分的右边，左边界点中大的一个是重叠部分的左边
    # 下边界点中小的一个是重叠部分的下边，上边界点中大的一个是重叠部分的上边
    left = torch.max(box1[..., 0], box2[..., 0])
    right = torch.min(box1[..., 2], box2[..., 2])
    top = torch.max(box1[..., 1], box2[..., 1])
    bottom = torch.min(box1[..., 3], box2[..., 3])

    width = torch.clamp(right - left, min=0)
    height = torch.clamp(bottom - top, min=0)

    overlap = width * height

    iou = overlap / (box1[..., 4] + box2[..., 4] - overlap)

    return iou


def non_max_suppression(prediction: torch.Tensor, conf_threshold=0.25, iou_threshold=0.45):
    # prediction = [batch, num_anchors,  center_x center_y width height + conf + num_classes]
    num_classes = prediction.shape[-1] - 5
    prediction = prediction[prediction[..., 4] > conf_threshold]
    xy, wh, conf, class_prediction = prediction.split([2, 2, 1, num_classes], dim=-1)

    class_prediction = class_prediction.argmax(dim=-1, keepdims=True)
    area = wh.prod(dim=-1, keepdims=True)
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    prediction = torch.cat([x1y1, x2y2, area, conf, class_prediction.float()], -1)

    score_order = conf[:, 0].argsort(descending=True)

    keep = []
    while score_order.shape[0] > 0:
        # 取出conf最高的一个作为预测结果
        best = score_order[0]
        score_order = score_order[1:]
        keep.append(best)

        iou = box_iou(prediction[best], prediction[score_order])

        score_order = score_order[iou < iou_threshold]  # 过滤掉剩下的框中与当前选出的框iou过大的

    return prediction[keep, :][:, [0, 1, 2, 3, 5, 6]]


def detect(network: yolo.Network.NetWork, images: list[str], label_names, device: torch.device):
    network.eval().to(device, non_blocking=True)

    for img in images:
        ori_img, ratio, (top, left) = letterbox(cv2.imread(img), [640, 640])
        h, w, _ = ori_img.shape
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[numpy.newaxis, ...]
        img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255

        output = network(img)
        output = network.detect.inference_post_process(output)

        output = non_max_suppression(output)

        display(ori_img, output, label_names)

    pass


def main(weight_dir, img_dir):
    device = torch.device("cuda")

    network = yolo.Network.NetWork(80)
    network.load_state_dict(torch.load(weight_dir))
    network.eval().to(device, non_blocking=True)

    label_names = ObjectRecord.load("dataset/cocos/obj_record.pkl").label_names

    detect(network, [os.path.join(img_dir, x) for x in os.listdir(img_dir)], label_names, device)

    pass


if __name__ == '__main__':
    main("weight/yolo.pth", r"D:\迅雷下载\train2017")
