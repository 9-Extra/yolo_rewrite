import torch
import torchvision


def non_max_suppression(batch_prediction: torch.Tensor, conf_threshold=0.25, iou_threshold=0.45):
    # [x y w h conf origin_bbox cls]

    num_classes = batch_prediction.shape[-1] - 10

    results = []
    for prediction in batch_prediction:
        prediction = prediction[prediction[..., 4] > conf_threshold]

        xy, wh, conf, _ = prediction.tensor_split([2, 4, 5], dim=-1)

        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2
        boxes = torch.cat([x1y1, x2y2], -1)
        keep = torchvision.ops.nms(boxes, conf.squeeze(1), iou_threshold)

        prediction = prediction[keep]

        xy, wh, conf, origin_bbox, layer_id, class_prediction = prediction.split([2, 2, 1, 4, 1, num_classes], dim=-1)
        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2

        class_prediction = torch.argmax(class_prediction, dim=-1, keepdim=True)
        prediction = torch.cat([x1y1, x2y2, conf, origin_bbox, layer_id, class_prediction.float()], -1)

        results.append(prediction)

    return results
