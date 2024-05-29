import cv2
import numpy

import utils
import torch
from torch.utils.data import DataLoader, Dataset
from dataset.h5Dataset import H5DatasetYolo
from rich.progress import track
from sklearn import metrics
from matplotlib import pyplot

from safe.FeatureExtract import FeatureExtract
from safe.safe_method import MLP, peek_relative_feature_single_batch
from yolo.non_max_suppression import non_max_suppression


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
    print("总样本数：", len(conf))
    detect_mask = conf != 0
    print("真正检测结果数：", numpy.count_nonzero(detect_mask))
    detect_ood = ood_score[detect_mask]
    detect_gt = tp[detect_mask, 0]  # 真正的目标
    print("检测结果中正确的目标数：", numpy.count_nonzero(detect_gt))
    assert detect_ood.shape[0] == detect_gt.shape[0]
    # TP数
    fpr, tpr, _ = metrics.roc_curve(detect_gt, detect_ood)
    # print(f"{fpr=}\n{tpr=}\n{threshold=}")
    pyplot.figure("ROC")
    pyplot.plot(fpr, tpr)
    pyplot.show()
    auroc = metrics.auc(fpr, tpr)
    fpr95 = fpr[numpy.where(tpr > 0.95)[0][0]]

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # type: ignore # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, auroc, fpr95


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


@torch.no_grad()
def collect_stats(network: torch.nn.Module, ood_evaluator: MLP, val_dataset: Dataset):
    device = next(network.parameters()).device
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=H5DatasetYolo.collate_fn)

    network.eval()
    feature_extractor = FeatureExtract(ood_evaluator.feature_name_set)
    feature_extractor.attach(network)

    ood_evaluator.eval()

    iouv = numpy.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.size
    stats = []

    for i, (img, target) in enumerate(track(dataloader)):
        img_h, img_w = img.shape[2:]  # noqa

        img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
        center_x, center_y, w, h = target[:, 2], target[:, 3], target[:, 4], target[:, 5]
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2
        target[:, 2:] = numpy.stack([x1, y1, x2, y2], -1) * numpy.array([img_w, img_h, img_w, img_h],
                                                                        dtype=numpy.float32)
        # target = torch.from_numpy(target).to(device)

        feature_extractor.ready()
        output = network(img)
        output = network.detect.inference_post_process(output)
        output = non_max_suppression(output)

        for i, batch_p in enumerate(output):
            predictions = batch_p.numpy(force=True)
            # 取得对应batch的正确label
            labels = target[target[:, 0] == i, 1:]

            # 检测结果实际上分三类：正确匹配的正样本，没有被匹配的正样本，误识别的负样本
            # 在进行OOD检测时需要区分这三种样本

            nl, npr = labels.shape[0], predictions.shape[0]  # number of labels, predictions

            if npr == 0:
                # 没有预测任何东西
                if nl != 0:  # 但是实际上有东西
                    correct = numpy.zeros([nl, niou], dtype=bool)  # 全错
                    # 没有被匹配的正样本
                    # stats.append((correct, *numpy.zeros([3, nl]), labels[:, 0]))
            else:
                # 预测出了东西
                if nl != 0:  # 实际上也有东西，这个时候才需要进行判断
                    # 可能产生三种样本
                    correct = process_batch(predictions, labels, iouv)
                else:
                    # 误识别的负样本
                    correct = numpy.zeros([npr, niou], dtype=bool)  # 全错

                conf = predictions[:, 4]
                cls = predictions[:, 10]

                # 计算ood_score
                feature = peek_relative_feature_single_batch(feature_extractor.get_features(), batch_p, i)
                ood_score = ood_evaluator(feature).numpy(force=True)
                assert ood_score.shape[0] == npr

                stats.append((correct, conf, ood_score, cls, labels[:, 0]))  # (correct, conf, pcls, tcls)
        pass

    feature_extractor.detach()

    stats = [numpy.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    return stats


def val(network: torch.nn.Module, ood_evaluator: MLP, val_dataset: Dataset):
    stats = collect_stats(network, ood_evaluator, val_dataset)

    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, auroc, fpr95 = ap_per_class(stats)
        ap50, ap95, ap = ap[:, 0], ap[:, -1], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map95, map, auroc = p.mean(), r.mean(), ap50.mean(), ap95.mean(), ap.mean(), auroc
    else:
        mp, mr, map50, ap50, map95, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        auroc = 0.0

    # Print results
    print(f"准确率 = {mp:.2%}, 召回率 = {mr:.2%}, map50 = {map50:.2%}, map95 = {map95:.2%}, map = {map:.2%}")
    print(f"AUROC = {auroc:.2%}")

    pass


def main(weight_path: str, data_path: str):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"正在验证网络{weight_path}， 使用数据集{data_path}")

    network, ood_evaluator, label_names = utils.load_network(weight_path, load_ood_evaluator=True)
    # ood_evaluator = MLP.from_static_dict(torch.load("mlp.pth"))
    ood_evaluator.to(device, non_blocking=True)
    network.eval().to(device, non_blocking=True)
    print("提取特征层：", ood_evaluator.feature_name_set)

    dataset = H5DatasetYolo(data_path)

    val(network, ood_evaluator, dataset)

    pass


if __name__ == '__main__':
    main("weight/yolo_final_full.pth", "preprocess/pure_drone_full_val.h5")
    # main("weight/yolo_drone_with_bird.pth", "preprocess/drone_val")
