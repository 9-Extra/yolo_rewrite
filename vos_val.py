import os
from typing import Sequence

import pandas

import torch
from torch.utils.data import Dataset

from yolo.Network import Yolo
from dataset.h5Dataset import H5DatasetYolo

from schedules.schedule import Config
from yolo.validation import ap_per_class, collect_stats


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
