import cv2
import numpy
import torchvision.ops

import yolo
import torch
from torch.utils.data import DataLoader
from dataset.h5Dataset import H5Dataset
import tqdm

from yolo.non_max_suppression import non_max_suppression


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


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
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
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = numpy.zeros((nc, tp.shape[1])), numpy.zeros((nc, 1000)), numpy.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = numpy.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = torchvision.ops.box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def val(network: torch.nn.Module, train_loader: DataLoader):
    device = next(network.parameters()).device

    network.eval()

    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0
    fp = 0.0
    stats = []

    for i, (img, target) in enumerate(tqdm.tqdm(train_loader)):
        img_h, img_w = img.shape[2:]

        img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
        center_x, center_y, w, h = target[:, 2], target[:, 3], target[:, 4], target[:, 5]
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2
        target[:, 2:] = numpy.stack([x1, y1, x2, y2], -1) * numpy.array([img_w, img_h, img_w, img_h], dtype=numpy.float32)
        target = torch.from_numpy(target).to(device)

        output = network(img)
        output = network.detect.inference_post_process(output)
        for i, pred in enumerate(output):
            pred = non_max_suppression(pred)

            labels = target[target[:, 0] == i, 1:]

            # ori_img = cv2.cvtColor(img[i].numpy(force=True).transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            # display(ori_img, pred, labels, train_loader.dataset.obj_record.label_names)

            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions

            correct = torch.zeros([npr, niou], dtype=torch.bool, device=device)  # init

            if npr == 0:
                # 没有预测任何东西
                if nl != 0:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
            else:
                # Evaluate
                if nl != 0:
                    correct = process_batch(pred, labels, iouv)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
        pass

    stats = [torch.cat(x, 0).numpy(force=True) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map, fp = p.mean(), r.mean(), ap50.mean(), ap.mean(), fp.mean()

    # Print results
    print(f"准确率 = {mp}, 召回率 = {mr} 误识别 = {fp}, map50 = {map50}, map = {map}")
    pass


def main(weight_path: str, data_path: str):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    network = yolo.Network.NetWork(2)
    network.load_state_dict(torch.load(weight_path))
    network.eval().to(device, non_blocking=True)

    dataset = H5Dataset(data_path)
    print("样本数：", len(dataset))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True,
                            collate_fn=H5Dataset.collate_fn)

    with torch.no_grad():
        val(network, dataloader)

    pass


if __name__ == '__main__':
    main("weight/yolo.pth", "preprocess/drone_val")
    main("weight/yolo_drone_with_bird.pth", "preprocess/drone_val")
