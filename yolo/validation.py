import cv2
import numpy
import torch
from sklearn import metrics


def box_iou(box1, box2):
    left_top = numpy.maximum(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    right_bottom = numpy.minimum(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    width_height = numpy.maximum(right_bottom - left_top, 0)

    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    overlap = numpy.prod(width_height, axis=-1)

    iou = overlap / (box1_area[:, None] + box2_area - overlap)

    return iou


def display(img: numpy.ndarray, objs, gt, label_names):
    for obj in objs:
        x1, y1, x2, y2, conf, cls = [x.item() for x in obj.cpu()]
        x1, y1, x2, y2, cls = round(x1), round(y1), round(x2), round(y2), int(cls)
        # print(x1, y1, x2, y2, cls, conf)

        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.putText(img, label_names[cls] + f"{conf:%}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                          cv2.LINE_AA)
    for obj in gt:
        cls, x1, y1, x2, y2 = [x.item() for x in obj.cpu()]
        x1, y1, x2, y2, cls = round(x1), round(y1), round(x2), round(y2), int(cls)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        img = cv2.putText(img, label_names[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)


def smooth(y, f=0.05):
    """Applies box filter smoothing to array `y` with fraction `f`, yielding a smoothed array."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode="valid")  # y-smoothed


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = numpy.concatenate(([0.0], recall, [1.0]))
    mpre = numpy.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = numpy.flip(numpy.maximum.accumulate(numpy.flip(mpre)))

    # Integrate area under curve
    method = "continuous"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = numpy.trapz(numpy.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = numpy.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = numpy.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(stat, eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.

    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    assert len(stat) != 0 and stat[0].any(), "必须包含正样本"

    tp, conf, ood_score, pred_cls, target_cls = stat
    # 从大到小排序
    i = numpy.argsort(-conf)
    tp, conf, ood_score, pred_cls = tp[i], conf[i], ood_score[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = numpy.zeros((nc, tp.shape[1])), numpy.zeros((nc, 1000)), numpy.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i: numpy.ndarray = pred_cls == c  # type: ignore  # 所有当前类型的预测结果
        n_l = nt[ci]  # 目标数
        n_p = i.sum()  # 预测结果数
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)  # 对于此类型的每一个检测结果，不同iou阈值下的FP样本数
        tpc = tp[i].cumsum(0)  # TP样本数，通过累加和计算不同iou阈值下的正确数量

        # Recall
        recall = tpc / (n_l + eps)  # recall curve，在不同iou阈值下的recall曲线
        # 插个值作为输出结果，只使用最宽松的iou阈值（0.5）
        # 原始曲线为recall[:, 0]-conf[i]，在不同的置信度下的recall值的曲线
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve，在不同iou阈值下的准确率
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # 统计OOD检测结果
    # print("总样本数：", len(conf))
    detect_mask = conf != 0
    # print("真正检测结果数：", numpy.count_nonzero(detect_mask))
    detect_ood = ood_score[detect_mask]
    detect_gt = tp[detect_mask, 0]  # 真正的目标
    # print("检测结果中正确的目标数：", numpy.count_nonzero(detect_gt))
    assert detect_ood.shape[0] == detect_gt.shape[0]

    # 使用ood_score计算auroc
    fpr, tpr, threshold = metrics.roc_curve(detect_gt, detect_ood)
    auroc = metrics.auc(fpr, tpr)
    tpr95_index = numpy.where(tpr > 0.95)[0][0]
    fpr95 = fpr[tpr95_index]
    thr = threshold[tpr95_index]

    # 使用conf计算auroc
    fpr, tpr, threshold = metrics.roc_curve(detect_gt, conf[detect_mask])
    conf_auroc = metrics.auc(fpr, tpr)
    tpr95_index = numpy.where(tpr > 0.95)[0][0]
    conf_fpr95 = fpr[tpr95_index]
    conf_thr = threshold[tpr95_index]

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # type: ignore # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, auroc, fpr95, thr, conf_auroc, conf_fpr95, conf_thr


def process_batch(detections, labels, iouv):
    """
    为每一个检测结果尝试匹配一个正确的标签
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = numpy.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)
    # torchvision.ops.box_iou()
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 10]  # 类型是否匹配
    # iou和correct_class都表示了每个检测结果和每个正确结果的配对
    for i in range(len(iouv)):
        # 计算当前iou阈值下正确配对的下标，包含多个正确的配对
        x = numpy.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0] != 0:  # 如果存在正确配对
            # matches包含配对信息和此配对的iou
            matches = numpy.concatenate((numpy.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detect, iou]
            if x[0].shape[0] > 1:  # 如果配对总数多于一个，需要过滤掉多对一或一对多的情况
                # 将所有配对以iou从高到低排序
                matches = matches[matches[:, 2].argsort()[::-1]]
                # 对于每个检测结果，如果对应了多个真实包围盒，只保留iou最高的一个
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                # 反之，对于每个包围盒，如果对应了多个检测结果，只保留iou最高的一个
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            # 将正确匹配结果对应的correct设置为True
            correct[matches[:, 1].astype(int), i] = True

    # correct中的每一行为一个检测结果
    # 如果其中存在True说明匹配上了，是TP
    # 如果全为False说明是FP，误识别
    # 没有检测到的正确结果实际上漏掉了没有统计

    return correct


def bbox_ratio2pixel(batched_bbox: numpy.ndarray, img_shape: torch.Size) -> numpy.ndarray:
    """
    :param batched_bbox: H5DatasetYolo输出的bbox后4列
    :param img_shape: 图像大小
    :return: 以像素为大小[x1, y1, x2, y2]格式的bbox，用于检验
    """
    img_h, img_w = img_shape[2:]
    gain = numpy.array([img_w, img_h, img_w, img_h], dtype=numpy.float32)

    center_x, center_y, w, h = batched_bbox[:, 0], batched_bbox[:, 1], batched_bbox[:, 2], batched_bbox[:, 3]
    x1 = center_x - w / 2
    y1 = center_y - h / 2
    x2 = center_x + w / 2
    y2 = center_y + h / 2
    return numpy.stack([x1, y1, x2, y2], -1) * gain


def match_nms_prediction(
        prediction: list[torch.Tensor],
        target: numpy.ndarray,
        img_shape: torch.Size,
        ood_score_pos: int = 4
        ):
    """
    将经过NMS后的预测结果与目标进行匹配。此函数并不在计算损失时使用，计算损失时另一套特殊的方法
    :param prediction: non_max_suppression()或者Yolo.inference()的输出
    :param target: H5DatasetYolo输出的targets，迭代时输出元组的第[1]个
    :param img_shape: 图像大小
    :param ood_score_pos: ood_score在prediction张量最后一维的下标，默认使用置信度
    :return:
    """
    iouv = numpy.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.size

    stats = []

    batched_labels = numpy.concatenate((target[:, 1:2], bbox_ratio2pixel(target[:, 2:], img_shape)), axis=-1)

    for i, batch_p in enumerate(prediction):  # 遍历每一张图的结果
        batch_p = batch_p.numpy(force=True)
        # 取得对应batch的正确label
        labels = batched_labels[target[:, 0] == i]

        # 检测结果实际上分三类：正确匹配的正样本，没有被匹配的正样本，误识别的负样本
        # 在进行OOD检测时需要区分这三种样本

        nl, npr = labels.shape[0], batch_p.shape[0]  # number of labels, predictions

        if npr == 0:
            # 没有预测任何东西
            if nl != 0:  # 但是实际上有东西
                correct = numpy.zeros([nl, niou], dtype=bool)  # 全错
                # 没有被匹配的正样本
                stats.append((correct, *numpy.zeros([3, nl]), labels[:, 0]))
        else:
            if nl != 0:  # 实际上也有东西，这个时候才需要进行判断
                # 可能产生三种样本
                correct = process_batch(batch_p, labels, iouv)
            else:
                # 误识别的负样本
                correct = numpy.zeros([npr, niou], dtype=bool)  # 全错

            conf = batch_p[:, 4]
            cls = batch_p[:, 10]
            ood_score = batch_p[:, ood_score_pos]

            stats.append((correct, conf, ood_score, cls, labels[:, 0]))  # (correct, conf, ood_score, pcls, tcls)

    return stats