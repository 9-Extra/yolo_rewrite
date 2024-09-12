import os
from typing import Sequence

import numpy
import pandas

import torch
from rich.progress import track
from torch.utils.data import Dataset, DataLoader

import yolo.Network
from yolo.Network import Yolo
from dataset.h5Dataset import H5DatasetYolo

from schedules.schedule import Config
from yolo.validation import ap_per_class, process_batch


def val(network: Yolo, val_dataset: Dataset):
    stats = collect_stats(network, val_dataset)

    tp, fp, p, r, f1, ap, auroc, fpr95, threshold, conf_auroc, conf_fpr95, conf_thr = ap_per_class(stats)
    ap50, ap95, ap = ap[:, 0], ap[:, -1], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map95, map, auroc = p.mean(), r.mean(), ap50.mean(), ap95.mean(), ap.mean(), auroc

    # Print results
    print(f"map50 = {map50:.2%}, map = {map:.2%}, 召回率 = {mr:.2%}")
    print(f"AUROC = {auroc:.2} FPR95={fpr95:.2}, threshold={threshold:.4}")

    return map50, map, mr, auroc, fpr95, conf_auroc, conf_fpr95

    pass


def main(config: Config, data_paths: Sequence[str]):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    collect = []
    network: Yolo = Yolo(config.num_class)
    network.eval().to(device, non_blocking=True)

    network.load_state_dict(torch.load("run/weight/yolo.pth", weights_only=True)["state_dict"])

    for path in data_paths:
        print(f"正在使用数据集{path}验证网络")
        dataset = H5DatasetYolo(path)
        result = val(network, dataset)
        collect.append((path, "original", *result))
    pass

    network.load_state_dict(torch.load("run/weight/vos_yolo.pth", weights_only=True))

    for path in data_paths:
        print(f"正在使用数据集{path}验证网络")
        dataset = H5DatasetYolo(path)
        result = val(network, dataset)
        collect.append((path, "vos", *result))
    pass

    collect = pandas.DataFrame(collect,
                               columns=['dataset', 'weight', 'map50', 'map95', 'recall', 'auroc', 'fpr95', 'conf_auroc',
                                        'conf_fpr95'])

    os.makedirs("run/summary", exist_ok=True)
    collect.to_csv("run/summary/vos_test_result.csv")
    print(collect.to_latex(index=False))


if __name__ == '__main__':
    config = Config()
    data_paths = [
        "run/preprocess/drone_test.h5",
        "run/preprocess/drone_test_with_bird.h5",
        "run/preprocess/drone_test_with_coco.h5"]
    main(config, data_paths)


@torch.no_grad()
def collect_stats(network: yolo.Network.Yolo, val_dataset: Dataset):
    device = network.device
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=H5DatasetYolo.collate_fn)

    network.eval()

    iouv = numpy.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.size
    stats = []

    for img, target in track(dataloader):
        img_h, img_w = img.shape[2:]  # noqa

        img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
        center_x, center_y, w, h = target[:, 2], target[:, 3], target[:, 4], target[:, 5]
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2
        target[:, 2:] = numpy.stack([x1, y1, x2, y2], -1) * numpy.array([img_w, img_h, img_w, img_h],
                                                                        dtype=numpy.float32)

        prediction = network.inference(img)

        for i, batch_p in enumerate(prediction):  # 遍历每一张图的结果
            batch_p = batch_p.numpy(force=True)
            # 取得对应batch的正确label
            labels = target[target[:, 0] == i, 1:]

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

                stats.append((correct, conf, conf, cls, labels[:, 0]))  # (correct, conf, pcls, tcls)
        pass

    pass

    stats = [numpy.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    return stats
