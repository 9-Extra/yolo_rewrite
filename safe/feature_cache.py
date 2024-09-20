from typing import Sequence
import h5py
import numpy


import functools

import torch



class ExtractFeatureDatabase:
    """
    此数据集中只包含用于训练MLP的正样本，负样本由DetectedDataset提供
    """
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

        neg = numpy.concatenate([d[()] for d in neg_ds], axis=1)
        pos = numpy.concatenate([d[()] for d in pos_ds], axis=1)

        assert pos.shape == neg.shape

        return neg, pos


class FeatureSaver:
    h5_dataset_dict: dict[str, h5py.Dataset] # cache
    h5_database: h5py.File
    positive_group_name: str = "positive_group"  # 记录正样本，使用硬连接访问，最后会删掉

    def __init__(self, h5_file_name: str):
        self.h5_database = h5py.File(h5_file_name, "a")
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

    def save_positive_feature(self, feature_dict: dict):
        self.save_features_h5(self.positive_group_name, feature_dict)

    def save_features_h5(self, group_name: str, feature_dict: dict[str, torch.Tensor]):
        for layer_name, feature in feature_dict.items():
            assert len(feature.shape) == 2, "压在一起的特征向量， 是二维的"

            name = f"{group_name}/{layer_name}"
            # 不存在或者不完整就重新创建数据集，否则从缓存self.h5_dataset_dict中取
            if name not in self.h5_dataset_dict:
                if name in self.h5_database:
                    del self.h5_database[name]

                dim = feature.shape[-1]
                h5 = self.h5_database.create_dataset(name, (0, dim), chunks=(32, dim), dtype="f4",
                                                maxshape=(None, dim))
                self.h5_database[group_name].attrs["incomplete"] = "True"
                self.h5_dataset_dict[name] = h5
                # 创建到对应正样本的硬连接
                link_name = f"{name}_pos"
                if link_name in self.h5_database:
                    del self.h5_database[link_name]

                assert f"{self.positive_group_name}/{layer_name}" in self.h5_database, "需要先保存正样本"
                self.h5_database[link_name] = self.h5_dataset_dict[f"{self.positive_group_name}/{layer_name}"]

            else:
                h5 = self.h5_dataset_dict[name]

            current_idx = h5.shape[0]
            f = feature.numpy(force=True)
            h5.resize(current_idx + f.shape[0], 0)
            h5.write_direct(f, dest_sel=slice(current_idx, current_idx + f.shape[0]))

    def complete(self):
        for name in self.h5_dataset_dict.keys():
            group_name = name.split('/')[0]
            g: h5py.Group = self.h5_database[group_name]
            if "incomplete" in g.attrs:
                del g.attrs["incomplete"]

        del self.h5_database[self.positive_group_name]

        self.h5_dataset_dict.clear()
        self.h5_database.flush()


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