from typing import Literal
import pathlib
import dataclasses

import torch

AttackMethod = Literal["pgd", "fgsm"]

@dataclasses.dataclass(slots=True)
class Config:
    device = torch.device("cuda")
    
    # main directory
    run_path: pathlib.Path = pathlib.Path("run")
    checkpoint_path = run_path / "weight"
    cache_path = run_path / "preprocess"
    log_path = run_path / "log"
    summary_path = run_path / "summary"
    
    # datasets path
    dataset_path_drone_train = r"/run/media/yty/盘盘/datasets/DroneTrainDataset/"
    dataset_path_drone_test = r"/run/media/yty/盘盘/datasets/DroneTestDataset/"
    dataset_path_coco = r"/run/media/yty/DATA/迅雷下载"
    dataset_path_drone_vs_bird = r"/run/media/yty/盘盘/datasets/BirdVsDrone"
    dataset_path_pascal_vos = r"/home/yty/桌面/workspace/VOC2012/"
    
    # datasets preprocss cache path
    h5_drone_train = cache_path / "drone_train.h5"
    h5_drone_val = cache_path / "drone_val.h5"
    h5_drone_test = cache_path / "drone_test.h5"
    h5_drone_train = cache_path / "drone_train.h5"
    h5_drone_test_with_coco = cache_path / "drone_test_with_coco.h5"
    
    # yolo
    num_class = 1
    batch_size = 8
    yolo_train_epoch = 100
    
    yolo_train_dataset = h5_drone_train
    yolo_val_dataset = h5_drone_val
    file_yolo_weight = checkpoint_path / "yolo.pth"

    # mlp
    mlp_epoch = 20
    file_detected_base_dataset = h5_drone_val
    cache_detected_dataset = cache_path / "cache_detected_dataset.pth"
    file_multi_layer_search_summary = summary_path / "multi_layer_search.json"

    # attack
    h5_extract_features = run_path / "extract_features.h5"
    # vos
    file_vos_yolo_weight = checkpoint_path / "vos_yolo.pth"