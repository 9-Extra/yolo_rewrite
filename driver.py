import torch

import pytorch_lightning

from dataset.h5Dataset import H5DatasetYolo
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

    train_dataset = H5DatasetYolo(config.file_train_dataset)  # 从训练集前向传播过程中抽取特征，用于训练OOD检测用
    search.extract_features(network, train_dataset, config.attack_method, config.mlp_epsilon,
                            config.dir_extract_features)


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
                                config.dir_extract_features,
                                config.file_single_layer_search_summary,
                                10,
                                config.device)


@Target(target_collect_result, target_extract_features)
def search_combine_layer():
    print("尝试策略")
    name_set_list = [*[{"backbone.inner.25.cv3.norm"} for _ in range(3)],
                     *[{"backbone.inner.25.cv3.norm", "backbone.inner.21.cv3.norm"} for _ in range(3)],
                     *[{"backbone.inner.25.cv3.norm", "backbone.inner.21.cv3.norm", "backbone.inner.29.cv3.norm"} for _
                       in range(3)],
                     ]
    search.search_multi_layers(name_set_list,
                               config.detected_result_dataset,
                               config.dir_extract_features,
                               config.file_multi_layer_search_summary,
                               15,
                               config.device)


@Target(target_collect_result, target_extract_features)
def train_mlp():
    name_set = {"backbone.inner.25.cv3.norm"}
    mlp_network, mlp_acc = (
        search.train_mlp_from_features_dir(name_set,
                                           config.detected_result_dataset,
                                           config.dir_extract_features, config.mlp_epoch, config.device))

    auroc, fpr95 = search.compute_auroc_fpr95(mlp_network, config.detected_result_dataset)
    print(f"{mlp_acc=:%} {auroc=} {fpr95=}")

    torch.save(mlp_network.to_static_dict(), config.file_mlp_weight)


if __name__ == '__main__':
    print("Start")
    scheduler.init_context(config.file_state_record)
    scheduler.re_run_target(target_extract_features)
    print("Done!")
