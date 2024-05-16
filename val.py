import cv2
import numpy
import sklearn

import yolo
import torch
from torch.utils.data import DataLoader
from dataset.h5Dataset import H5DatasetYolo
from rich.progress import track
from sklearn import metrics

from yolo.non_max_suppression import non_max_suppression


def box_iou(box1, box2):
    left_top = numpy.maximum(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    right_bottom = numpy.minimum(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    width_height = numpy.maximum(right_bottom - left_top, 0)

    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    overlap = numpy.prod(width_height, axis=-1)

    iou = overlap / (box1_area + box2_area - overlap)

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
    tp, conf, ood_score, pred_cls, target_cls = stat
    # 从大到小排序
    i = numpy.argsort(-conf)
    tp, conf, ood_score, pred_cls = tp[i], ood_score[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = numpy.zeros((nc, tp.shape[1])), numpy.zeros((nc, 1000)), numpy.zeros((nc, 1000))
    auroc = numpy.zeros(nc)
    fpr95 = numpy.zeros(nc)
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

        fpr = fpc[:, 0] / n_p
        tpr = tpc[:, 0] / n_p
        auroc[ci] = metrics.auc(fpr, tpr)
        # fpr95[ci] = fpr[numpy.where(tpr > 0.95)] 计算上还有问题


    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, ood_score, auroc, fpr95


def process_batch(detections, labels, iouv):
    """
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
    correct_class = labels[:, 0:1] == detections[:, 10]
    for i in range(len(iouv)):
        x = numpy.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = numpy.concatenate((numpy.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def val(network: torch.nn.Module, ood_evaluators, train_loader: DataLoader):
    device = next(network.parameters()).device

    network.eval()

    iouv = numpy.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.size
    fp = 0.0
    stats = []

    for i, (img, target) in enumerate(track(train_loader)):
        img_h, img_w = img.shape[2:]

        img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
        center_x, center_y, w, h = target[:, 2], target[:, 3], target[:, 4], target[:, 5]
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2
        target[:, 2:] = numpy.stack([x1, y1, x2, y2], -1) * numpy.array([img_w, img_h, img_w, img_h],
                                                                        dtype=numpy.float32)
        # target = torch.from_numpy(target).to(device)

        extract_features = {}
        output = network(img, extract_features)
        output = network.detect.inference_post_process(output)
        output = non_max_suppression(output)
        ood_scores = ood_evaluators.score(extract_features, output)

        for i, (pred, ood_score) in enumerate(zip(output, ood_scores)):
            pred = pred.numpy(force=True)
            ood_score = ood_score.numpy(force=True)

            # 取得对应batch的正确label
            labels = target[target[:, 0] == i, 1:]

            # TP：正确检测结果
            # FP：目标没有被检测到或者类型错误
            # FN：没有目标的地方打上了框
            # Precision：所有真正的目标中被正确识别的概率 (TP / 目标总数)
            # Recall：检测结果正确的概率 (TP / 检测结果数)

            # 检测步骤为先将预测出的包围盒与GT包围盒进行匹配
            # 对每个预测结果匹配IOU最大的包围盒，只要IOU超过阈值且类型一致就算是TP，如果类型不一致为记为FP
            # 如果预测结果没有匹配上GT，记为FN
            # 如果有GT没有匹配预测结果，记为FP

            # 检测步骤为先将预测出的包围盒与GT包围盒进行匹配
            # 记录目标数和检测结果数
            # 对每个预测结果匹配IOU最大的包围盒，只要IOU超过阈值且类型一致就可能是TP
            # 如果预测结果没有匹配上GT，记为FN，为误识别，记录其OOD score
            # 对每个可能是TP的结构，记录其置信度值，OOD score

            # 对于一个只有负样本的数据集，如何检验神经网络的效能

            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions

            if npr == 0:
                # 没有预测任何东西
                if nl != 0:  # 但是实际上有东西
                    correct = numpy.zeros([npr, niou], dtype=bool)  # 全错
                    stats.append((correct, *numpy.zeros([3, npr]), labels[:, 0]))
            else:
                # 预测出了东西
                if nl != 0:  # 实际上也有东西，这个时候才需要进行判断
                    correct = process_batch(pred, labels, iouv)
                else:
                    correct = numpy.zeros([npr, niou], dtype=bool)  # 全错
                conf = pred[:, 4]
                # ood_score = pred[:, 5]
                cls = pred[:, 10]
                stats.append((correct, conf, ood_score, cls, labels[:, 0]))  # (correct, conf, pcls, tcls)
        pass

    stats = [numpy.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ood_score, auroc, fpr95 = ap_per_class(stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map, auroc = p.mean(), r.mean(), ap50.mean(), ap.mean(), auroc.mean()
    else:
        mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0
        auroc = 0.0

    # Print results
    print(f"准确率 = {mp:.2%}, 召回率 = {mr:.2%}, map50 = {map50:.2%}, map = {map:.2%}")
    print(f"AUROC = {auroc:.2%}")
    pass


def main(weight_path: str, data_path: str):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"正在验证网络{weight_path}， 使用数据集{data_path}")

    network, ood_evaluators, label_names = yolo.Network.load_network(weight_path, load_ood_evaluator=True)
    ood_evaluators.to(device, non_blocking=True)
    network.eval().to(device, non_blocking=True)

    dataset = H5DatasetYolo(data_path)
    print("样本数：", len(dataset))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True,
                            collate_fn=H5DatasetYolo.collate_fn)

    with torch.no_grad():
        val(network, ood_evaluators, dataloader)

    pass


if __name__ == '__main__':
    main("weight/yolo_final_full_20.pth", "preprocess/pure_drone_val_200.h5")
    # main("weight/yolo_drone_with_bird.pth", "preprocess/drone_val")
