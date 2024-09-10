import os
from typing import Sequence

import pandas

import torch
from torch.utils.data import Dataset

import safe
import scheduler
import search
from yolo.Network import Yolo
from dataset.h5Dataset import H5DatasetYolo

from driver import target_collect_result
from safe.attack import FSGMAttack
from safe.mlp import MLP
from schedules.schedule import Config
from yolo.validation import ap_per_class, collect_stats_with_mlp


def val(network: Yolo, ood_evaluator: MLP, val_dataset: Dataset):
    stats = collect_stats_with_mlp(network, ood_evaluator, val_dataset)

    if len(stats) != 0 and stats[0].any():
        tp, fp, p, r, f1, ap, auroc, fpr95, threshold, conf_auroc, conf_fpr95, conf_thr = ap_per_class(stats)
        ap50, ap95, ap = ap[:, 0], ap[:, -1], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map95, map, auroc = p.mean(), r.mean(), ap50.mean(), ap95.mean(), ap.mean(), auroc
    else:
        mp, mr, map50, ap50, map95, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        auroc, fpr95, threshold, conf_auroc, conf_fpr95, conf_thr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Print results
    print(f"map50 = {map50:.2%}, map = {map:.2%}, 召回率 = {mr:.2%}")
    print(f"AUROC = {auroc:.2} FPR95={fpr95:.2}, threshold={threshold:.4}")

    return map50, map, mr, auroc, fpr95, conf_auroc, conf_fpr95

    pass


def main(config: Config, data_paths: Sequence[str]):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    network: Yolo = Yolo(config.num_class)
    network.load_state_dict(torch.load("run/weight/yolo.pth", weights_only=True)['state_dict'])
    network.eval().to(device, non_blocking=True)

    feature_name_set = {'backbone.inner.25'}

    layer_name_list = []
    for layer in network.layer_order:
        if layer in feature_name_set:
            layer_name_list.append(layer)

    attackers = [
        FSGMAttack(0.08),
        # *(PDGAttack(e, 20) for e in (0.005, 0.006, 0.007)),
        # PDGAttack(0.006, 20)
    ]

    collect = []  # attacker_name,

    safe.safe_method.extract_features_h5(
        network,
        feature_name_set,
        H5DatasetYolo(config.file_train_dataset),
        attackers,
        config.h5_extract_features
    )
    for attacker in attackers:
        mlp, _ = search.train_mlp_from_features(
            layer_name_list,
            config.extract_features_database,
            attacker.name,
            40,
            config.device
        )

        mlp.eval().to(device, non_blocking=True)
        print("提取特征层：", mlp.layer_name_list)
        print("总特征长度：", mlp.in_dim)

        for path in data_paths:
            print(f"正在使用数据集{path}，对抗攻击方法{attacker.name}验证网络")
            dataset = H5DatasetYolo(path)
            result = val(network, mlp, dataset)
            collect.append((path, attacker.name, *result))
        pass

    collect = pandas.DataFrame(collect, columns=['dataset', 'attacker', 'map50', 'map95', 'recall', 'auroc', 'fpr95', 'conf_auroc', 'conf_fpr95'])
    os.makedirs("run/summary", exist_ok=True )
    collect.to_csv("run/summary/test_result.csv")
    print(collect.to_latex(index=False))


if __name__ == '__main__':
    config = Config()
    scheduler.run_target(target_collect_result)
    data_paths = [
        "run/preprocess/drone_test.h5",
        "run/preprocess/drone_test_with_bird.h5",
        "run/preprocess/drone_test_with_coco.h5"
    ]
    main(config, data_paths)
