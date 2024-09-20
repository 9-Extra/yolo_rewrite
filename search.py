import csv
import os
import re

import numpy
import torch
import torchvision
from rich.progress import track
from sklearn import metrics

from torch.utils.data import DataLoader, TensorDataset

from dataset.h5Dataset import H5DatasetYolo
from safe.FeatureExtract import FeatureExtract
from safe.safe_method import ExtractFeatureDatabase
from safe.mlp import MLP, train_mlp
from yolo.validation import process_batch
from yolo.Network import FeatureExporter, BackBone, Conv, FeatureConcat, Yolo


class ExtractAll:

    def __init__(self):
        self.pattern = re.compile("detect|bottlenecks|loss_func")

    def get_name_set(self, network: torch.nn.Module):
        names = set()
        for name, layer in network.named_modules():
            if self.filter(name, layer):
                names.add(name)

        return names

    def filter(self, name: str, layer: torch.nn.Module) -> bool:
        if self.pattern.search(name) is None:
            if not isinstance(layer, (
                    FeatureExporter, FeatureConcat, BackBone, Conv, torch.nn.Identity, torch.nn.MaxPool2d)):
                if name != "backbone.inner" and name != "":
                    return True
        return False


class DetectedDataset:
    """
    检测模型在验证集上运行后检测结果和特征，用于进一步的OOD检测
    """
    tp: numpy.ndarray
    conf: numpy.ndarray
    ood_features: dict[str, torch.Tensor]

    def __init__(self, tp: numpy.ndarray, conf: numpy.ndarray, ood_features: dict[str, torch.Tensor]):
        self.tp = tp
        self.conf = conf
        self.ood_features = ood_features

    def save(self, path):
        torch.save({
            "tp": self.tp,
            "conf": self.conf,
            "ood_features": self.ood_features
        }, path)

    @staticmethod
    def load(path):
        data = torch.load(path, weights_only=False)
        return DetectedDataset(data["tp"], data["conf"], data["ood_features"])


@torch.no_grad()
def collect_stats_and_feature(network: Yolo, val_dataset: H5DatasetYolo):
    device = next(network.parameters()).device
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=H5DatasetYolo.collate_fn)

    network.eval()
    feature_extractor = FeatureExtract(ExtractAll().get_name_set(network))
    feature_extractor.attach(network)

    iouv = numpy.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.size
    stats = []
    ood_feature_collect = {}

    for img, target in track(dataloader):
        img_h, img_w = img.shape[2:]  # type: ignore

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
        output = network.inference(img)

        for i, batch_p in enumerate(output):  # 对于每一张图像上的结果
            predictions = batch_p.numpy(force=True)
            # 取得对应batch的正确label
            labels = target[target[:, 0] == i, 1:]

            # 检测结果实际上分三类：正确匹配的正样本，没有被匹配的正样本，误识别的负样本
            # 在进行OOD检测时需要区分这三种样本

            nl, npr = labels.shape[0], predictions.shape[0]  # number of labels, predictions

            if npr == 0:
                # 没有预测任何东西，不考虑
                pass
            else:
                # 预测出了东西
                if nl != 0:  # 实际上也有东西，这个时候才需要进行判断
                    # 可能产生三种样本
                    correct = process_batch(predictions, labels, iouv)
                else:
                    # 误识别的负样本
                    correct = numpy.zeros([npr, niou], dtype=bool)  # 全错

                conf = predictions[:, 4]

                batch_bbox = torch.empty([batch_p.shape[0], 5], dtype=torch.float32, device=device)
                batch_bbox[..., 0] = i
                batch_bbox[..., 1:] = batch_p[..., 5: 9]

                for name, feature in feature_extractor.get_features().items():
                    b, c, h, w = feature.shape

                    relative_feature = torchvision.ops.roi_align(feature, batch_bbox, (2, 2), w) # type: ignore
                    ood_feature_collect.setdefault(name, []).append(relative_feature.flatten(start_dim=1))  # 保留第0维

                stats.append((correct, conf))
        pass

    tp, conf = [numpy.concatenate(x, 0) for x in zip(*stats)]
    ood_feature_collect = {k: torch.cat(v) for k, v in ood_feature_collect.items()}

    result_num = tp.shape[0]
    assert all(result_num == v.shape[0] for v in ood_feature_collect.values())

    # 统计OOD检测结果
    print("真正检测结果数：", result_num)
    print("检测结果中正确的目标数：", numpy.count_nonzero(tp[:, 0]))
    # # TP数
    # fpr, tpr, _ = metrics.roc_curve(detect_gt, detect_ood)

    return DetectedDataset(tp[:, 0], conf, ood_feature_collect)


def compute_auroc_fpr95(mlp: MLP, feature_data: DetectedDataset):
    # 需要保证feature_dict中顺序与network.named_modules的顺序一致
    # feature_dict为每一层已经经过roi_align的特征，包含 样本数 个batch
    collected = []
    name_set = mlp.layer_name_list
    for name in feature_data.ood_features.keys():
        if name in name_set:
            collected.append(feature_data.ood_features[name])

    assert len(collected) == len(name_set)

    collected = torch.cat(collected, dim=1)  # 拼接来自不同层的特征
    assert collected.shape[0] == feature_data.tp.shape[0]  # 样本数一致

    mlp.eval()
    with torch.no_grad():
        ood_score = mlp(collected).numpy(force=True)

    fpr, tpr, _ = metrics.roc_curve(feature_data.tp, ood_score)
    # pyplot.figure("ROC")
    # pyplot.plot(fpr, tpr)
    # pyplot.show()
    auroc = metrics.auc(fpr, tpr)
    fpr95 = fpr[numpy.where(tpr > 0.95)[0][0]].item()

    return auroc, fpr95


def train_mlp_from_features(
        layer_name_list: list[str],
        feature_database: ExtractFeatureDatabase,
        attacker_name: str,
        epoch: int,
        device: torch.device
):
    assert len(layer_name_list) != 0, "搞啥呢"

    neg, pos = feature_database.query_features(attacker_name, layer_name_list)

    x = torch.from_numpy(numpy.concatenate((pos, neg))).to(device, non_blocking=True)
    y = torch.zeros(x.shape[0], device=device, dtype=torch.float32)
    y[0:x.shape[0] // 2] = 1  # 前一半为正样本
    del pos, neg

    dataset = TensorDataset(x, y)
    feature_dim = x.shape[1]

    mlp = MLP(feature_dim, layer_name_list)
    # mlp = torch.compile(mlp, backend="cudagraphs", fullgraph=True, disable=False)
    mlp_acc = train_mlp(mlp, dataset, 64, epoch, device)

    return mlp, mlp_acc


def search_layers(name_set_list: list[set],
                  feature_data: DetectedDataset,
                  feature_database: ExtractFeatureDatabase,
                  attacker_name: str,
                  summary_path: str,
                  epoch: int,
                  device: torch.device):
    """
    搜索指定的特征策略
    """
    layer_order = list(feature_data.ood_features.keys())

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(["name_set", "mlp_acc", "auroc", "fpr95"])
        for name_set in name_set_list:
            # train
            name_list = []
            for layer in layer_order:
                if layer in name_set:
                    name_list.append(layer)
            assert len(name_set) == len(name_list)

            mlp_network, mlp_acc = train_mlp_from_features(
                name_list,
                feature_database,
                attacker_name,
                epoch,
                device
            )
            # val
            auroc, fpr95 = compute_auroc_fpr95(mlp_network, feature_data)

            print(f"{mlp_acc=:%} {auroc=} {fpr95=}")
            spamwriter.writerow([str(name_set), f"{mlp_acc:%}", auroc, fpr95])
