import functools
import time
from collections import defaultdict
from typing import Sequence

import h5py
import numpy
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn

import yolo
from dataset.h5Dataset import H5DatasetYolo
from .FeatureExtract import FeatureExtract
from .attack import Attacker


class MLP(torch.nn.Module):
    in_dim: int
    feature_name_set: set[str]

    def __init__(self, in_dim, feature_name_set: set[str]):
        super().__init__()
        self.in_dim = in_dim
        self.feature_name_set = feature_name_set
        self.inner = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim // 2),
            # torch.nn.BatchNorm1d(in_dim // 2),
            torch.nn.ReLU(True),

            torch.nn.Linear(in_dim // 2, in_dim // 4),
            # torch.nn.BatchNorm1d(in_dim // 4),
            torch.nn.ReLU(True),

            torch.nn.Linear(in_dim // 4, 1),
            torch.nn.Flatten(0),
        )

    def forward(self, x):
        if self.training:
            return self.inner(x)
        else:
            return torch.sigmoid_(self.inner(x))

    def to_static_dict(self):
        return {
            "in_dim": self.in_dim,
            "feature_name_set": self.feature_name_set,
            "weight": self.state_dict()
        }

    @staticmethod
    def from_static_dict(mlp_dict: dict):
        mlp = MLP(mlp_dict["in_dim"], mlp_dict["feature_name_set"])
        mlp.load_state_dict(mlp_dict["weight"])

        return mlp


class ExtractFeatureDatabase:
    h5: h5py.File

    def __init__(self, path: str):
        self.h5 = h5py.File(path)

    @functools.cache
    def count(self):
        for attacker_name in self.h5.keys():
            for feature in self.h5[attacker_name].keys():
                return self.h5[attacker_name][feature].shape[0]
        return 0

    def attacker_names(self):
        return self.h5.keys()

    def query_features(self, attacker_name: str, layer_name_list: list):
        assert len(layer_name_list) != 0
        neg_ds = [self.h5[f"{attacker_name}/{layer_name}"] for layer_name in layer_name_list]
        pos_ds = [self.h5[f"{attacker_name}/{layer_name}_pos"] for layer_name in layer_name_list]
        count = neg_ds[0].shape[0]
        assert all(d.shape[0] == count for d in pos_ds) and all(d.shape[0] == count for d in neg_ds)

        neg = numpy.concatenate([d[()] for d in neg_ds], axis=1)
        pos = numpy.concatenate([d[()] for d in pos_ds], axis=1)
        return neg, pos


def peek_relative_feature_batched(feature_dict: dict[str, torch.Tensor], prediction: list[torch.Tensor]):
    device = prediction[0].device
    # 遍历每一个batch的输出
    result = []
    for batch_id, batch_p in enumerate(prediction):
        collected = peek_roi_single_batch(feature_dict, batch_p, batch_id)
        result.append(collected)

    return torch.cat(result) if len(result) != 0 else torch.empty(0, device=device, dtype=torch.float32)


def peek_roi_single_batch(feature_dict: dict[str, torch.Tensor], prediction: torch.Tensor, batch_id: int):
    device = prediction.device

    if prediction.shape[0] == 0:
        return torch.empty(0, device=device)

    origin_bbox = prediction[..., 5: 9]

    batch_bbox = torch.empty([prediction.shape[0], 5], dtype=torch.float32, device=device)
    batch_bbox[..., 0] = batch_id
    batch_bbox[..., 1:] = origin_bbox

    collected = []

    for name, feature in feature_dict.items():
        b, c, h, w = feature.shape

        relative_feature = torchvision.ops.roi_align(feature, batch_bbox, (2, 2), w)
        collected.append(relative_feature.flatten(start_dim=1))  # 保留第0维

    collected = torch.cat(collected, dim=1)  # 将来自不同层的特征拼接起来
    assert collected.shape[0] == prediction.shape[0]

    return collected


def peek_roi_to_dict(
        feature_dict: dict[str, torch.Tensor],
        prediction: list[torch.Tensor],
):
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

        return out_dict


class _FeatureSaver:
    h5_dataset_dict: dict[str, h5py.Dataset] # cache
    h5_database: h5py.File
    positive_group_name: str

    def __init__(self, h5_database: h5py.File, positive_group_name: str):
        self.h5_database = h5_database
        self.positive_group_name = positive_group_name
        self.h5_dataset_dict = {}

    def is_completed(self, group_name: str, layer_name_set: set[str]):
        if group_name not in self.h5_database:
            return False

        group: h5py.Group = self.h5_database[group_name]
        if "incomplete" in group.attrs:
            return False

        return set(group.keys()).issuperset(layer_name_set)

    def clear_incomplete(self, group_names: Sequence[str], layer_name_set: set[str]):
        for group_name in group_names:
            if not self.is_completed(group_name, layer_name_set):
                self.delete_group(group_name)

    def delete_group(self, group_name: str):
        if group_name not in self.h5_database:
            return
        del self.h5_database[group_name]


    def save_features_h5(self, group_name: str, feature_dict: dict):
        for layer_name, feature in feature_dict.items():
            if len(feature) == 0:
                continue
            assert len(feature[0].shape) == 2

            name = f"{group_name}/{layer_name}"
            # 不存在或者不完整就重新创建数据集，否则从缓存self.h5_dataset_dict中取
            if name not in self.h5_dataset_dict:
                if name in self.h5_database:
                    del self.h5_database[name]

                dim = feature[0].shape[-1]
                h5 = self.h5_database.create_dataset(name, (0, dim), chunks=(32, dim), dtype="f4",
                                                maxshape=(None, dim))
                self.h5_database[group_name].attrs["incomplete"] = "True"
                self.h5_dataset_dict[name] = h5
                # 创建到对应正样本的硬连接
                link_name = f"{name}_pos"
                if link_name in self.h5_database:
                    del self.h5_database[link_name]
                self.h5_database[link_name] = self.h5_dataset_dict[f"{self.positive_group_name}/{layer_name}"]

            else:
                h5 = self.h5_dataset_dict[name]

            current_idx = h5.shape[0]
            f = torch.cat(feature).numpy(force=True)
            feature.clear()
            h5.resize(current_idx + f.shape[0], 0)
            h5.write_direct(f, dest_sel=slice(current_idx, current_idx + f.shape[0]))

    def complete(self):
        for name in self.h5_dataset_dict.keys():
            group_name = name.split('/')[0]
            g: h5py.Group = self.h5_database[group_name]
            if "incomplete" in g.attrs:
                del g.attrs["incomplete"]

        self.h5_dataset_dict.clear()


def extract_features_h5(network: yolo.Network.Yolo,
                        feature_name_set: set,
                        dataset: Dataset,
                        attackers: Sequence[Attacker],
                        h5_file_name: str):
    with h5py.File(h5_file_name, "a") as h5_database:

        positive_group_name = f"positive_{time.strftime('%Y.%m.%d-%H:%M:%S')}"
        saver = _FeatureSaver(h5_database, positive_group_name)
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


        train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True,
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
                target = torch.from_numpy(target).to(device, non_blocking=True)

                attack_samples = [attacker(network, img, target) for attacker in attackers]

                feature_extractor.attach(network)
                # 正样本
                feature_extractor.ready()
                prediction = network.inference(img)
                feature_dict = peek_roi_to_dict(feature_extractor.get_features(), prediction)
                saver.save_features_h5(positive_group_name, feature_dict)

                # 对抗攻击目标特征
                for sample, attacker in zip(attack_samples, attackers):
                    feature_extractor.ready()
                    _ = network(sample)  # 不需要结果
                    feature_dict = peek_roi_to_dict(feature_extractor.get_features(), prediction)
                    saver.save_features_h5(attacker.name, feature_dict)

                feature_extractor.detach()  # 在进行对抗攻击时不记录特征
            pass

        saver.complete()

    pass


def train_mlp(mlp: MLP, train_dataset, batch_size: int, epoch: int, device: torch.device):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)

    mlp = mlp.to(device).train()
    opt = torch.optim.Adam(mlp.parameters(), lr=0.0001)
    loss_func = torch.nn.BCEWithLogitsLoss()

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[acc]:%}"),
    ) as progress:
        train_task = progress.add_task("train", acc=0)
        epoch_task = progress.add_task("epoch", acc=0)
        for _ in progress.track(range(epoch), task_id=train_task, description="训练MLP"):
            progress.reset(epoch_task)

            for x, y in progress.track(train_dataloader, len(train_dataloader), task_id=epoch_task,
                                       description="epoch"):
                x = x.to(device)
                y = y.to(device)

                opt.zero_grad()
                output = mlp(x)
                loss = loss_func(output, y)
                loss.backward()
                opt.step()

                # acc
                p = torch.zeros_like(output)
                p[output > 0.5] = 1
                val_acc = torch.count_nonzero(p == y).item() / y.shape[0]
                progress.update(epoch_task, acc=val_acc)

        progress.reset(epoch_task)

        mlp.eval()
        val_acc = torch.tensor(0, device=device)
        count = 0
        with torch.no_grad():
            for x, y in progress.track(train_dataloader, len(train_dataloader), task_id=epoch_task, description="val"):
                output = mlp.forward(x)
                p = torch.zeros_like(output)
                p[output > 0.5] = 1
                val_acc += torch.count_nonzero(p == y)
                count += y.shape[0]

            val_acc = val_acc.item() / count
        return val_acc
