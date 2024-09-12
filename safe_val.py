import os
from typing import Sequence

import numpy
import pandas

import torch
from rich.progress import track
from torch.utils.data import Dataset, DataLoader

import safe
import search
import yolo.Network
from safe.FeatureExtract import FeatureExtract
from safe.safe_method import feature_roi_flatten
from yolo.Network import Yolo
from dataset.h5Dataset import H5DatasetYolo

from safe.attack import FSGMAttack
from safe.mlp import MLP
from schedules.schedule import Config
from yolo.validation import ap_per_class, process_batch


@torch.no_grad()
def collect_stats_with_mlp(network: yolo.Network.Yolo, mlp: MLP, val_dataset: Dataset):
    device = network.device
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=H5DatasetYolo.collate_fn)

    network.eval()
    feature_extractor = FeatureExtract(set(mlp.layer_name_list))
    feature_extractor.attach(network)

    mlp.eval()

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

        feature_extractor.ready()
        prediction = network.inference(img)

        if sum(p.shape[0] for p in prediction) != 0:
            feature_dict = feature_roi_flatten(feature_extractor.get_features(), prediction)
            feature = torch.cat([feature_dict[layer] for layer in mlp.layer_name_list], dim=1) # 按层序拼接
            ood_scores = mlp(feature).numpy(force=True)
            del feature_dict, feature
        else:
            ood_scores = torch.empty(0, device=device)

        assert sum(p.shape[0] for p in prediction) == ood_scores.shape[0], "有多少个检测结果就有多少个ood_score"

        offset = 0 # 因为ood_scores是所有图的ood检测结果拼接起来的，每张图的结果数又不一样，需要使用offset访问
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

                ood_score = ood_scores[offset: offset + npr]
                stats.append((correct, conf, ood_score, cls, labels[:, 0]))  # (correct, conf, pcls, tcls)

            offset += npr # 偏移结果数
        pass

        assert offset == ood_scores.shape[0]
    pass

    feature_extractor.detach()

    stats = [numpy.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    return stats

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
    # print(f"map50 = {map50:.2%}, map = {map:.2%}, 召回率 = {mr:.2%}")
    # print(f"AUROC = {auroc:.2} FPR95={fpr95:.2}, threshold={threshold:.4}")

    return map50, map, mr, auroc, fpr95, conf_auroc, conf_fpr95

    pass


def main(config: Config, data_paths: Sequence[str]):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    network: Yolo = Yolo(config.num_class)
    network.load_state_dict(torch.load("run/weight/yolo.pth", weights_only=True))
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

    safe.safe_method.extract_features_h5(
        network,
        feature_name_set,
        H5DatasetYolo(config.file_train_dataset),
        attackers,
        config.h5_extract_features
    )

    collect = []  # 收集结果
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
    print(collect.to_markdown())
    os.makedirs("run/summary", exist_ok=True )
    collect.to_csv("run/summary/test_result.csv")
    print(collect.to_latex(index=False))


if __name__ == '__main__':
    config = Config()
    data_paths = [
        "run/preprocess/drone_train.h5",
        "run/preprocess/drone_val.h5",
        "run/preprocess/drone_test.h5",
        "run/preprocess/drone_test_with_bird.h5",
        "run/preprocess/drone_test_with_coco.h5"
    ]
    main(config, data_paths)
