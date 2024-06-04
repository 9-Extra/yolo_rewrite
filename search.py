import os
import re

import numpy
import torch
import torchvision
from matplotlib import pyplot
from rich.progress import track
from sklearn import metrics

from torch.utils.data import DataLoader, Dataset

import utils
import json
from dataset.h5Dataset import H5DatasetYolo
from safe.FeatureExtract import FeatureExtract, Strategy
from safe.safe_method import mlp_build_dataset_separate, MLP, train_mlp
from val import process_batch
from yolo.Network import FeatureExporter, BackBone, Conv, FeatureConcat
from yolo.non_max_suppression import non_max_suppression


class ExtractAll(Strategy):

    def __init__(self):
        self.pattern = re.compile("detect|bottlenecks")

    def filter(self, name: str, layer: torch.nn.Module) -> bool:
        if self.pattern.search(name) is None:
            if not isinstance(layer, (
                    FeatureExporter, FeatureConcat, BackBone, Conv, torch.nn.Identity, torch.nn.MaxPool2d)):
                if name != "backbone.inner" and name != "":
                    return True


def extract_features(network, dataset: Dataset, epsilon: float, dir_name: str):
    os.makedirs(dir_name, exist_ok=True)

    feature_name_set = ExtractAll().get_name_set(network)
    print(f"共提取{len(feature_name_set)}层的特征")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True,
                            collate_fn=H5DatasetYolo.collate_fn)

    positive_features, negative_features = mlp_build_dataset_separate(network, feature_name_set, dataloader, epsilon)
    for name in list(positive_features.keys()):
        positive = positive_features.pop(name)  # save memory
        negative = negative_features.pop(name)
        assert sum(p.shape[0] for p in positive) == sum(n.shape[0] for n in negative)  # 正负样本数相同

        x = torch.cat((*positive, *negative))

        feature_dim = x.shape[1]
        print(f"导出：{name}, 样本数：{x.shape[0]}, 特征长度：{feature_dim}")

        torch.save(x, os.path.join(dir_name, name))


class FeatureDataset:
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
        data = torch.load(path)
        return FeatureDataset(data["tp"], data["conf"], data["ood_features"])


@torch.no_grad()
def collect_stats(network: torch.nn.Module, val_dataset: H5DatasetYolo):
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

    for i, (img, target) in enumerate(track(dataloader)):
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
        output = network(img)
        output = network.detect.inference_post_process(output)
        output = non_max_suppression(output)

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

                    relative_feature = torchvision.ops.roi_align(feature, batch_bbox, (2, 2), w)
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

    return FeatureDataset(tp[:, 0], conf, ood_feature_collect)


def compute_auroc_fpr95(mlp: MLP, feature_data: FeatureDataset):
    # 需要保证feature_dict中顺序与network.named_modules的顺序一致
    # feature_dict为每一层已经经过roi_align的特征，包含 样本数 个batch
    collected = []
    name_set = mlp.feature_name_set
    for name in feature_data.ood_features.keys():
        if name in name_set:
            collected.append(feature_data.ood_features[name])

    assert len(collected) == len(name_set)

    collected = torch.cat(collected, dim=1)  # 拼接来自不同层的特征
    assert collected.shape[0] == feature_data.tp.shape[0]  # 样本数一致

    mlp.eval()
    ood_score = mlp(collected).numpy(force=True)

    fpr, tpr, _ = metrics.roc_curve(feature_data.tp, ood_score)
    # pyplot.figure("ROC")
    # pyplot.plot(fpr, tpr)
    # pyplot.show()
    auroc = metrics.auc(fpr, tpr)
    fpr95 = fpr[numpy.where(tpr > 0.95)[0][0]].item()

    return auroc, fpr95


def train_mlp_from_features_dir(feature_name_set: set, feature_data: FeatureDataset, train_features_dir: str,
                                epoch: int,
                                device: torch.device):
    assert len(feature_name_set) != 0, "搞啥呢"
    if len(feature_name_set) > 1:
        x = []  # 需要保证顺序network.named_modules的顺序一致
        for name in feature_data.ood_features.keys():  # 这里只是利用feature_data中特征顺序与网络相同的特性
            if name in feature_name_set:
                fx = torch.load(os.path.join(train_features_dir, name))
                x.append(fx)

        assert len(x) == len(feature_name_set)  # 每个特征都找到了
        x = torch.cat(x, dim=1).to(device, non_blocking=True)
        y = torch.zeros(x.shape[0], device=device, dtype=torch.float32)
        y[0:x.shape[0] // 2] = 1  # 前一半为正样本
    else:
        # shortcut 只有一个特征就不需要考虑顺序
        for name in feature_name_set:
            break  # 就是取第一个
        x = torch.load(os.path.join(train_features_dir, name)).to(device, non_blocking=True)  # noqa
        y = torch.zeros(x.shape[0], device=device, dtype=torch.float32)
        y[0:x.shape[0] // 2] = 1  # 前一半为正样本

    feature_dim = x.shape[1]
    print(f"使用特征层：{feature_name_set}\n特征长度：{feature_dim}")

    dataset = torch.utils.data.TensorDataset(x, y)

    mlp = MLP(feature_dim, feature_name_set)
    mlp_acc = train_mlp(mlp, dataset, 64, epoch, device)

    return mlp, mlp_acc


def search_mlp_classifier_single(feature_data: FeatureDataset, train_features_dir: str, epoch: int,
                                 device: torch.device):
    """
    搜索所有的单层
    """
    results = []
    for name in os.listdir(train_features_dir):
        # train
        mlp_network, mlp_acc = train_mlp_from_features_dir({name}, feature_data, train_features_dir, epoch, device)
        # val
        auroc, fpr95 = compute_auroc_fpr95(mlp_network, feature_data)

        print(f"{mlp_acc=:%} {auroc=} {fpr95=}")
        results.append((name, auroc, fpr95))

    os.makedirs("summary", exist_ok=True)
    with open("summary/feature_layer_result_odd.csv", "w") as f:
        f.write(f"{'name'},{'auroc'},{'fpr95'}\n")
        for name, auroc, fpr95 in results:
            f.write(f"{name},{auroc},{fpr95}\n")

    # json.dump(results, open("summary/feature_layer_result.json", "w"))


def search_mlp_classifier_multi(name_set_list: list[set],
                                feature_data: FeatureDataset,
                                train_features_dir: str,
                                epoch: int,
                                device: torch.device):
    """
    搜索指定的特征策略
    """
    results = []
    for name_set in name_set_list:
        # train
        mlp_network, mlp_acc = train_mlp_from_features_dir(name_set, feature_data, train_features_dir, epoch, device)
        # val
        auroc, fpr95 = compute_auroc_fpr95(mlp_network, feature_data)

        print(f"{mlp_acc=:%} {auroc=} {fpr95=}")
        results.append((list(name_set), auroc, fpr95))

    json.dump(results, open("summary/feature_layer_result_multi_deep.json", "w"))


def main(weight_path: str, data_path: str):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"正在使用权重{weight_path}， 使用数据集{data_path}")

    network, _, _ = utils.load_network(weight_path, load_ood_evaluator=False)
    network.to(device, non_blocking=True)

    # train_dataset = H5DatasetYolo(data_path)  # 用于训练OOD检测用
    # extract_features(network, train_dataset, 0.02, "extract_features")

    feature_dataset_path = "preprocess/feature_pure_val n.h5"
    if os.path.isfile(feature_dataset_path):
        feature_dataset = FeatureDataset.load(feature_dataset_path)
    else:
        val_dataset = H5DatasetYolo("preprocess/pure_drone_full_val.h5")
        print("样本数：", len(val_dataset))
        feature_dataset = collect_stats(network, val_dataset)
        feature_dataset.save(feature_dataset_path)

    del network

    # print("开始尝试每一层")
    # search_mlp_classifier_single(feature_dataset, "extract_features", 10, device)

    print("尝试策略")
    name_set_list = [*[{"backbone.inner.25.cv3.conv"} for _ in range(3)],
                     *[{"backbone.inner.25.cv3.conv", "backbone.inner.29.cv3.conv"} for _ in range(3)],
                     *[{"backbone.inner.25.cv3.conv", "backbone.inner.29.cv3.conv", "backbone.inner.27.conv"} for _ in range(3)],
                     ]
    # search_mlp_classifier_multi(name_set_list, feature_dataset, "extract_features", 20, device)
    mlp_network, mlp_acc = train_mlp_from_features_dir({"backbone.inner.25.cv3.conv"}, feature_dataset, "extract_features", 30, device)
    auroc, fpr95 = compute_auroc_fpr95(mlp_network, feature_dataset)

    print(f"{mlp_acc=:%} {auroc=} {fpr95=}")

    torch.save(mlp_network.to_static_dict(), "mlp.pth")


if __name__ == '__main__':
    main("weight/yolo_checkpoint_72.pth", "preprocess/pure_drone_train_full.h5")
