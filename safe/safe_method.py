from collections import defaultdict
import csv
import os
import re
from typing import Sequence

import numpy
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, track
from sklearn import metrics
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision

from dataset.h5Dataset import H5DatasetYolo
from safe.FeatureExtract import FeatureExtract
from safe.attack import Attacker
from safe.feature_cache import DetectedDataset, ExtractFeatureDatabase, FeatureSaver
from safe.mlp import MLP, train_mlp
from yolo.network import BackBone, Conv, FeatureConcat, FeatureExporter, Yolo
from yolo.validation import match_nms_prediction_fp_only, process_batch



def feature_roi_flatten(
        feature_dict: dict[str, torch.Tensor],
        prediction: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    对提取出的特征进行roi
    :param feature_dict: 使用FeatureExtractor提取的完整特征
    :param prediction: 模型inference结果，已经进行过NMS
    :return: 隐藏层名称到其特征的映射，特征为二维（样本数，特征长度）
    """
    device = prediction[0].device

    out_dict: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    # 遍历每一个batch的输出
    for batch_id, batch_p in enumerate(prediction):
        if batch_p.shape[0] == 0:
            continue

        origin_bbox = batch_p[..., 5: 9]

        batch_bbox = torch.empty([batch_p.shape[0], 5], dtype=torch.float32, device=device)
        batch_bbox[..., 0] = batch_id
        batch_bbox[..., 1:] = origin_bbox

        for name, feature in feature_dict.items():
            b, c, h, w = feature.shape

            relative_feature = torchvision.ops.roi_align(feature, batch_bbox, (2, 2), w)
            out_dict[name].append(relative_feature.flatten(start_dim=1))  # 保留第0维

    result: dict[str, torch.Tensor] = {}
    for k, v in out_dict.items():
        result[k] = torch.cat(v)

    return result


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
    ood_feature_collect = defaultdict(list)

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
                    ood_feature_collect[name].append(relative_feature.flatten(start_dim=1).cpu())  # 保留第0维，移到内存节省显存

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
    
    collected = collected.to(next(mlp.parameters()).device, non_blocking=True)

    mlp.eval()
    with torch.inference_mode():
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
    mlp_acc = train_mlp(mlp, dataset, 256, epoch, device)

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
            spamwriter.writerow([str(name_set), mlp_acc, auroc, fpr95])


def extract_features_h5(network: Yolo,
                        feature_name_set: set,
                        dataset: str,
                        attackers: Sequence[Attacker],
                        h5_file_name: str):

    saver = FeatureSaver(h5_file_name)
    new_attackers = []
    for attacker in attackers:
        c = saver.is_completed(attacker.name, feature_name_set)
        print(f"数据集{attacker.name}" + ("完整" if c else "缺失"))
        if not c:
            saver.delete_group(attacker.name)
            new_attackers.append(attacker)

    if len(new_attackers) == 0:
        print("所有特征已缓存，跳过")
        return

    attackers = new_attackers
    print(f"正在生成{len(attackers)}组特征：", [attacker.name for attacker in attackers])

    train_loader = DataLoader(H5DatasetYolo(dataset), batch_size=8, shuffle=True, num_workers=0, pin_memory=True,
                              collate_fn=H5DatasetYolo.collate_fn)

    device = next(network.parameters()).device
    network.eval()

    feature_extractor = FeatureExtract(feature_name_set)

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
    ) as process:
        for i, (img, target) in enumerate(process.track(train_loader, len(train_loader), description="收集特征")):
            img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
            target_tensor = torch.from_numpy(target).to(device, non_blocking=True)

            attack_samples = [attacker(network, img, target_tensor) for attacker in attackers]

            feature_extractor.attach(network)
            # 正样本
            feature_extractor.ready()
            prediction = network.inference(img)

            correct = match_nms_prediction_fp_only(prediction, target, img.shape)
            for i in range(len(prediction)):
                prediction[i] = prediction[i][correct[i]]  # 只将正确识别的对象作为正样本

            feature_dict = feature_roi_flatten(feature_extractor.get_features(), prediction)

            saver.save_positive_feature(feature_dict)

            # 对抗攻击目标特征
            for sample, attacker in zip(attack_samples, attackers):
                feature_extractor.ready()
                _ = network(sample)  # 不需要结果
                feature_dict = feature_roi_flatten(feature_extractor.get_features(), prediction)
                saver.save_features_h5(attacker.name, feature_dict)

            feature_extractor.detach()  # 在进行对抗攻击时不记录特征
        pass

    saver.complete()
    pass




