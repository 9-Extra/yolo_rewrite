import os
from typing import Sequence

import numpy
import pandas

import torch
from rich.progress import track
from torch.utils.data import Dataset, DataLoader

import safe
from safe.feature_cache import ExtractFeatureDatabase
import safe.safe_method
import yolo.Network
from safe.FeatureExtract import FeatureExtract
from safe.safe_method import feature_roi_flatten
from yolo.Network import Yolo
from dataset.h5Dataset import H5DatasetYolo

from safe.attack import FSGMAttack
from safe.mlp import MLP
from config import Config
from yolo.validation import match_nms_prediction, ap_per_class


@torch.no_grad()
def val_with_mlp(network: yolo.Network.Yolo, mlp: MLP, val_dataset: Dataset):
    device = network.device
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=H5DatasetYolo.collate_fn)

    network.eval()
    feature_extractor = FeatureExtract(set(mlp.layer_name_list))
    feature_extractor.attach(network)

    mlp.eval()

    stats = []
    for img, target in track(dataloader):
        feature_extractor.ready()
        img = torch.from_numpy(img).to(network.device, non_blocking=True).float() / 255

        prediction = network.inference(img)

        if sum(p.shape[0] for p in prediction) != 0:
            feature_dict = feature_roi_flatten(feature_extractor.get_features(), prediction)
            feature = torch.cat([feature_dict[layer] for layer in mlp.layer_name_list], dim=1) # 按层序拼接
            ood_scores = mlp(feature)
            del feature_dict, feature
        else:
            ood_scores = torch.empty(0, device=device)

        # 将ood_scores加入prediction中
        index = 0 # 因为ood_scores是所有图的ood检测结果拼接起来的，每张图的结果数又不一样，需要使用offset访问
        ood_score_pos = prediction[0].shape[-1] if len(prediction) != 0 else 0 # 将ood_score放在prediction的最后一个
        for i in range(len(prediction)):
            npr = prediction[i].shape[0]
            s = ood_scores[index: index + npr].unsqueeze_(-1)
            prediction[i] = torch.cat((prediction[i], s) , dim=-1)
            index += npr

        assert index == ood_scores.shape[0], "有多少个检测结果就有多少个ood_score"

        stats.extend(match_nms_prediction(prediction, target, img.shape, ood_score_pos))

    feature_extractor.detach()

    stats = [numpy.concatenate(x, 0) for x in zip(*stats)]  # 合并

    # 生成指标
    tp, fp, p, r, f1, ap, auroc, fpr95, threshold, conf_auroc, conf_fpr95, conf_thr = ap_per_class(stats)
    ap50, ap95 = ap[:, 0], ap[:, -1]  # AP@0.5, AP@0.5:0.95
    mr, map50, map95 = r.mean(), ap50.mean(), ap95.mean()

    summary = dict(map50=map50, map95=map95, recall=mr, auroc=auroc, fpr95=fpr95, conf_auroc=conf_auroc, conf_fpr95=conf_fpr95)

    return summary


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

    summary_table = pandas.DataFrame(columns=['dataset', 'attacker', 'map50', 'map95', 'recall', 'auroc', 'fpr95', 'conf_auroc', 'conf_fpr95'])  # 收集结果
    for attacker in attackers:
        mlp, _ = safe.safe_method.train_mlp_from_features(
            layer_name_list,
            ExtractFeatureDatabase(config.h5_extract_features),
            attacker.name,
            40,
            config.device
        )

        mlp.eval().to(device, non_blocking=True)
        print("提取特征层：", mlp.layer_name_list)
        print("总特征长度：", mlp.in_dim)

        summary_table = []
        for path in data_paths:
            print(f"正在使用数据集{path}，对抗攻击方法{attacker.name}验证网络")
            summary = {"dataset": path, "attacker": attacker.name}

            dataset = H5DatasetYolo(path)
            summary.update(val_with_mlp(network, mlp, dataset))

            summary_table.append(summary)
        pass

    summary_table = pandas.DataFrame(summary_table)

    print(summary_table.to_markdown())
    os.makedirs("run/summary", exist_ok=True )
    summary_table.to_csv("run/summary/test_result.csv")
    print(summary_table.to_latex(index=False))


if __name__ == '__main__':
    config = Config()
    data_paths = [
        # "run/preprocess/drone_train.h5",
        "run/preprocess/drone_val.h5",
        "run/preprocess/drone_test.h5",
        "run/preprocess/drone_test_with_bird.h5",
        "run/preprocess/drone_test_with_coco.h5"
    ]
    main(config, data_paths)
