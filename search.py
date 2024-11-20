import os
from typing import Iterable
import torch
from config import Config
from safe import safe_method
from safe.feature_cache import DetectedDataset, ExtractFeatureDatabase

from dataset.h5Dataset import H5DatasetYolo
from safe.attack import Attacker, FSGMAttack, PDGAttack
from yolo import Yolo


def trained_yolo_network(config: Config):
    network = Yolo(config.num_class)
    network.load_state_dict(torch.load(config.file_yolo_weight, weights_only=True))
    # network = torch.compile(network)
    return network

def cache_detect_result(config: Config):
    """
    在验证集上进行一遍检测并缓存检测的结果，对应的隐藏层特征和结果的正确性
    """
    if os.path.exists(config.safe_cache_detect_result):
        # 存在就跳过
        return

    network = trained_yolo_network(config)
    network.to(config.device, non_blocking=True)

    val_dataset = H5DatasetYolo(config.safe_val_dataset)
    print("样本数：", len(val_dataset))
    result_dataset = safe_method.collect_stats_and_feature(network, val_dataset)
    result_dataset.save(config.safe_cache_detect_result)


def _safe_val(
    config: Config,
    network: torch.nn.Module,
    name_set_list: list[set],
    attackers: Iterable[Attacker],
    summary_name: str,
    epoch: int = 15,
):
    # 缓存验证集上的检测结果，在进行评估的时候就不需要再检测一遍
    cache_detect_result(config)

    layer_name_set = set().union(*name_set_list)  # 所有需要收集的特征层集合

    # 进行对抗攻击收集用于训练mlp的特征数据，检测的样本就是训练集，样本缓存到config.h5_extract_features
    safe_method.extract_features_h5(
        network,
        layer_name_set,
        config.yolo_train_dataset,
        attackers,
        config.h5_extract_features,
    )

    for attacker in attackers:
        # 使用收集的数据训练mlp，并在验证集上检验其有效性，生成统计结果
        safe_method.search_layers(
            name_set_list,
            DetectedDataset.load(config.safe_cache_detect_result),
            ExtractFeatureDatabase(config.h5_extract_features),
            attacker.name,
            config.summary_path / f"{summary_name}_{attacker.name}.csv",
            epoch,
            config.device,
        )



def search_all_single_layer(config: Config):
    """
    搜索每一个单层
    """
    network = trained_yolo_network(config)
    network.to(config.device, non_blocking=True)

    attackers: list[Attacker] = [
        FSGMAttack(0.08),
        # PDGAttack(0.006, 20),
        # *(PDGAttack(e, 10) for e in (0.005, 0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1))
    ]

    feature_name_set = safe_method.ExtractAll().get_name_set(network)
    print(f"共提取{len(feature_name_set)}层的特征")

    name_set_list = [{layer_name} for layer_name in feature_name_set]  # 所有的单层
    
    _safe_val(config, network, name_set_list, attackers, "single_layer")


def search_combine_layer(config: Config):
    
    network = trained_yolo_network(config)
    network.to(config.device, non_blocking=True)
    
    attackers = [FSGMAttack(0.08), PDGAttack(0.003, 5)]

    name_set_list = [
        {"backbone.inner.25"},
        {"backbone.inner.25", "backbone.inner.21"},
        {"backbone.inner.25", "backbone.inner.21", "backbone.inner.23.norm"},
    ] * 3
    
    _safe_val(config, network, name_set_list, attackers, "multi_layer", 20)


def mult_epsilon_compare(config: Config):
    network = trained_yolo_network()
    network.to(config.device, non_blocking=True)

    feature_name_set = {"backbone.inner.25"}
    name_set_list = [feature_name_set] * 3

    attackers = [
        # (FSGMAttack(e) for e in (0.05, 0.06, 0.07, 0.08, 0.09, 0.1)),
        # *(PDGAttack(e, 20) for e in (0.005, 0.006, 0.007)),
        *(PDGAttack(e, 10) for e in (0.005, 0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1))
    ]
    
    _safe_val(config, network, name_set_list, attackers, "episilon_compare", 30)
    
if __name__ == '__main__':
    config = Config.from_profile("./profiles/coco_mixed.toml")
    # search_all_single_layer(config)
    search_combine_layer(config)