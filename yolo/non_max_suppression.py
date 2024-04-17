import torch
import torchvision


def box_iou(box1, box2):
    left = torch.max(box1[..., 0], box2[..., 0])
    right = torch.min(box1[..., 2], box2[..., 2])
    top = torch.max(box1[..., 1], box2[..., 1])
    bottom = torch.min(box1[..., 3], box2[..., 3])

    width = torch.clamp(right - left, min=0)
    height = torch.clamp(bottom - top, min=0)
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    overlap = width * height

    iou = overlap / (box1_area + box2_area - overlap)

    return iou


def _box_iou_with_area(box1, box2):
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


# def non_max_suppression(prediction: torch.Tensor, conf_threshold=0.25, iou_threshold=0.45):
#     # prediction = [batch, num_anchors,  center_x center_y width height + conf + num_classes]
#     num_classes = prediction.shape[-1] - 5
#     prediction = prediction[prediction[..., 4] > conf_threshold]
#     xy, wh, conf, class_prediction = prediction.split([2, 2, 1, num_classes], dim=-1)
#
#     class_prediction = class_prediction.argmax(dim=-1, keepdims=True)
#     area = wh.prod(dim=-1, keepdims=True)
#     x1y1 = xy - wh / 2
#     x2y2 = xy + wh / 2
#     prediction = torch.cat([x1y1, x2y2, area, conf, class_prediction.float()], -1)
#
#     score_order = conf[:, 0].argsort(descending=True)
#
#     keep = []
#     while score_order.shape[0] > 0:
#         # 取出conf最高的一个作为预测结果
#         best = score_order[0]
#         score_order = score_order[1:]
#         keep.append(best)
#
#         iou = _box_iou_with_area(prediction[best], prediction[score_order])
#
#         score_order = score_order[iou < iou_threshold]  # 过滤掉剩下的框中与当前选出的框iou过大的
#
#     return prediction[keep, :][:, [0, 1, 2, 3, 5, 6]]
def non_max_suppression(prediction: torch.Tensor, conf_threshold=0.25, iou_threshold=0.45):
    num_classes = prediction.shape[-1] - 5
    conf = prediction[..., 4] * torch.max(prediction[..., 5:], dim=-1)[0]
    prediction[..., 4] = conf
    prediction = prediction[conf > conf_threshold]
    xy, wh, conf, _ = prediction.tensor_split([2, 4, 5], dim=-1)

    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    boxes = torch.cat([x1y1, x2y2], -1)
    keep = torchvision.ops.nms(boxes, conf.squeeze(1), iou_threshold)

    prediction = prediction[keep]
    xy, wh, conf, class_prediction = prediction.split([2, 2, 1, num_classes], dim=-1)
    value, class_prediction = torch.max(class_prediction, dim=-1, keepdim=True)
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    prediction = torch.cat([x1y1, x2y2, value * conf, class_prediction.float()], -1)

    return prediction