import torch
import torchvision


def non_max_suppression(
        batch_prediction: torch.Tensor,
        num_class: int,
        num_extra_data: int = 0,
        conf_threshold=0.25,
        iou_threshold=0.45
) -> list[torch.Tensor]:
    """
    非极大值抑制
    :return: 因为batch中每张图像包含的检测结果数不同，使用list放置每张图的结果
    """
    # [x y w h conf origin_bbox layer_id cls extra]
    assert batch_prediction.shape[-1] == 10 + num_class + num_extra_data

    results = []
    for prediction in batch_prediction:
        prediction = prediction[prediction[..., 4] > conf_threshold]

        xy, wh, conf, _ = prediction.tensor_split([2, 4, 5], dim=-1)

        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2
        boxes = torch.cat([x1y1, x2y2], -1)
        keep = torchvision.ops.nms(boxes, conf.squeeze(1), iou_threshold)

        prediction = prediction[keep]

        xy, wh, conf, origin_bbox, layer_id, class_logit, extra_data = prediction.split([2, 2, 1, 4, 1, num_class, num_extra_data], dim=-1)
        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2

        cls = torch.argmax(class_logit, dim=-1, keepdim=True)
        prediction = torch.cat([x1y1, x2y2, conf, origin_bbox, layer_id, cls.float(), extra_data], -1)

        results.append(prediction)

    # 输出最后一维的格式为[true_bbox 0: 4, conf 4, origin_bbox 5: 9, layer_id 9, cls 10, ...]
    return results
