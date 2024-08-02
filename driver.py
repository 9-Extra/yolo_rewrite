import torch

import safe
from dataset.h5Dataset import H5DatasetYolo
from safe.attack import FSGMAttack, PDGAttack
from scheduler import Target
from schedules.schedule import Config
import preprocess
import train
import search
import scheduler

config = Config()


@Target()
def target_preprocess_train_dataset():
    preprocess.main(config.file_train_dataset, config.raw_train_dataset)


@Target(target_preprocess_train_dataset)
def target_train():
    train.main(config)


@Target(target_train, target_preprocess_train_dataset)
def target_extract_features():
    network = config.trained_yolo_network
    network.to(config.device, non_blocking=True)

    attackers = (
        FSGMAttack(0.05),
    )

    train_dataset = H5DatasetYolo(config.file_train_dataset)  # 从训练集前向传播过程中抽取特征，用于训练OOD检测用

    feature_name_set = search.ExtractAll().get_name_set(network)
    print(f"共提取{len(feature_name_set)}层的特征")

    safe.safe_method.extract_features_h5(network, feature_name_set, train_dataset, attackers,
                                         config.h5_extract_features)


@Target()
def target_preprocess_result_val_dataset():
    preprocess.main(config.file_detected_base_dataset, config.raw_detected_base_dataset)


@Target(target_train, target_preprocess_result_val_dataset)
def target_collect_result():
    network = config.trained_yolo_network
    network.to(config.device, non_blocking=True)

    val_dataset = H5DatasetYolo(config.file_detected_base_dataset)
    print("样本数：", len(val_dataset))
    result_dataset = search.collect_stats(network, val_dataset)
    result_dataset.save(config.file_detected_dataset)


@Target(target_collect_result, target_extract_features)
def target_search_layer():
    print("开始尝试每一层")
    search.search_single_layers(config.detected_result_dataset,
                                config.h5_extract_features,
                                "attacker",
                                config.file_single_layer_search_summary,
                                10,
                                config.device)


@Target(target_collect_result, target_extract_features)
def search_combine_layer():
    print("尝试策略")
    name_set_list = [*[{"backbone.inner.25"} for _ in range(3)],
                     *[{"backbone.inner.25", "backbone.inner.21"} for _ in range(3)],
                     *[{"backbone.inner.25", "backbone.inner.21", "backbone.inner.23.norm"} for _
                       in range(3)],
                     ]
    search.search_multi_layers(name_set_list,
                               config.detected_result_dataset,
                               config.h5_extract_features,
                               "attacker",
                               config.file_multi_layer_search_summary,
                               15,
                               config.device)


@Target(target_collect_result, target_extract_features)
def train_mlp():
    name_set = {"backbone.inner.25"}
    mlp_network, mlp_acc = (
        search.train_mlp_from_features(name_set,
                                       config.detected_result_dataset,
                                       config.extract_features_database, "attacker", config.mlp_epoch, config.device))

    auroc, fpr95 = search.compute_auroc_fpr95(mlp_network, config.detected_result_dataset)
    print(f"{mlp_acc=:%} {auroc=} {fpr95=}")

    torch.save(mlp_network.to_static_dict(), config.file_mlp_weight)


@Target(target_collect_result)
def mult_epsilon_compare():
    network = config.trained_yolo_network
    network.to(config.device, non_blocking=True)

    train_dataset = H5DatasetYolo(config.file_train_dataset)

    feature_name_set = {"backbone.inner.25"}
    name_set_list = [feature_name_set] * 3

    attackers = [
        *(FSGMAttack(e) for e in (0.05, 0.06, 0.07, 0.08, 0.09, 0.1)),
        # *(PDGAttack(0.06, ep) for ep in (5, 10, 15))
    ]

    @Target(name=f"mult_epsilon_compare.extract_features")
    def extract_features():
        safe.safe_method.extract_features_h5(network, feature_name_set, train_dataset, attackers,
                                             config.h5_extract_features)

    scheduler.run_target(extract_features)

    for attacker in attackers:
        @Target(extract_features, name=f"mult_epsilon_compare.val_{attacker.name}")
        def val():
            search.search_multi_layers(name_set_list,
                                       config.detected_result_dataset,
                                       config.extract_features_database,
                                       attacker.name,
                                       f"run/summary/multi_layer_search_{attacker.name}.csv",
                                       30,
                                       config.device)

        scheduler.run_target(val)


# def configure_inject(binder: inject.Binder):
#     binder.bind_to_constructor(yolo.Network.Yolo, config.trained_yolo_network)
#     binder.bind(AttackMethod, config.attack_method)


if __name__ == '__main__':
    print("Start")
    scheduler.init_context(config.file_state_record)
    scheduler.re_run_target( mult_epsilon_compare)
    print("Done!")
