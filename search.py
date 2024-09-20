import os
from config import Config
from safe import safe_method
from safe.feature_cache import ExtractFeatureDatabase
from scheduler import Target
import scheduler
import functools

config = Config()

@functools.cache
def trained_yolo_network():
    from yolo import Yolo
    import torch
    network = Yolo(config.num_class)
    network.load_state_dict(torch.load(config.file_yolo_weight, weights_only=True))
    # network = torch.compile(network)
    return network


@Target("_yolo_train")
def target_collect_result():
    if os.path.exists(config.file_detected_dataset):
        return

    from dataset.h5Dataset import H5DatasetYolo
    import search

    network = trained_yolo_network()
    network.to(config.device, non_blocking=True)

    val_dataset = H5DatasetYolo(config.file_detected_base_dataset)
    print("样本数：", len(val_dataset))
    result_dataset = safe_method.collect_stats_and_feature(network, val_dataset)
    result_dataset.save(config.file_detected_dataset)


@Target(target_collect_result)
def target_search_layer():
    from safe.attack import FSGMAttack, PDGAttack
    from dataset.h5Dataset import H5DatasetYolo
    import search

    print("开始尝试每一层")

    network = trained_yolo_network()
    network.to(config.device, non_blocking=True)

    attackers = [
        FSGMAttack(0.08),
        PDGAttack(0.006, 20),
        # *(PDGAttack(e, 10) for e in (0.005, 0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1))
    ]

    train_dataset = H5DatasetYolo(config.file_train_dataset)  # 从训练集前向传播过程中抽取特征，用于训练OOD检测用

    feature_name_set = safe_method.ExtractAll().get_name_set(network)
    print(f"共提取{len(feature_name_set)}层的特征")

    safe_method.extract_features_h5(network, feature_name_set, train_dataset, attackers,
                                         config.h5_extract_features)

    name_set_list = [{layer_name} for layer_name in feature_name_set]  # 所有的单层
    for attacker in attackers:
        safe_method.search_layers(
            name_set_list,
            config.detected_result_dataset,
            ExtractFeatureDatabase(config.h5_extract_features),
            attacker.name,
            f"run/summary/single_layer_search_{attacker.name}.csv",
            15,
            config.device
        )


@Target(target_collect_result)
def search_combine_layer():
    from safe.attack import FSGMAttack, PDGAttack

    print("尝试策略")

    attackers = [FSGMAttack(e) for e in (0.05, 0.06, 0.07, 0.08, 0.09, 0.1)]

    name_set_list = [*[{"backbone.inner.25"} for _ in range(3)],
                     *[{"backbone.inner.25", "backbone.inner.21"} for _ in range(3)],
                     *[{"backbone.inner.25", "backbone.inner.21", "backbone.inner.23.norm"} for _
                       in range(3)],
                     ]

    for attacker in attackers:
        safe_method.search_layers(
            name_set_list,
            config.detected_result_dataset,
            ExtractFeatureDatabase(config.h5_extract_features),
            attacker.name,
            f"run/summary/multi_layer_search_{attacker.name}.csv",
            15,
            config.device
        )


@Target("target_collect_result")
def mult_epsilon_compare():
    from safe.attack import FSGMAttack, PDGAttack
    from dataset.h5Dataset import H5DatasetYolo

    network = trained_yolo_network()
    network.to(config.device, non_blocking=True)

    train_dataset = H5DatasetYolo(config.file_train_dataset)

    feature_name_set = {"backbone.inner.25"}
    name_set_list = [feature_name_set] * 3

    attackers = [
        # (FSGMAttack(e) for e in (0.05, 0.06, 0.07, 0.08, 0.09, 0.1)),
        # *(PDGAttack(e, 20) for e in (0.005, 0.006, 0.007)),
        *(PDGAttack(e, 10) for e in (0.005, 0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1))
    ]

    safe_method.extract_features_h5(network, feature_name_set, train_dataset, attackers,
                                         config.h5_extract_features)

    for attacker in attackers:
        @Target(name=f"mult_epsilon_compare.val_{attacker.name}")
        def val():
            safe_method.search_layers(name_set_list,
                                 config.detected_result_dataset,
                                 ExtractFeatureDatabase(config.h5_extract_features),
                                 attacker.name,
                                 f"run/summary/multi_layer_search_{attacker.name}.csv",
                                 30,
                                 config.device)

        scheduler.run_target(val)

