from typing import Literal
from pathlib import Path
import dataclasses

import torch

AttackMethod = Literal["pgd", "fgsm"]

@dataclasses.dataclass(slots=True)
class Config:
    device = torch.device("cuda")
    
    # main directory
    run_path: Path = Path("run")
    cache_path = run_path / "preprocess"
    
    model_name = "yolo"
    model_specific_path = run_path / model_name
    checkpoint_path = model_specific_path / "weight"
    log_path = model_specific_path / "log"
    summary_path = model_specific_path / "summary"
    
    # datasets path
    dataset_path_drone_train = Path(r"/run/media/yty/盘盘/datasets/DroneTrainDataset/")
    dataset_path_drone_test = Path(r"/run/media/yty/盘盘/datasets/DroneTestDataset/")
    dataset_path_coco = Path(r"/run/media/yty/DATA/迅雷下载")
    dataset_path_drone_vs_bird = Path(r"/run/media/yty/盘盘/datasets/BirdVsDrone")
    dataset_path_pascal_vos = Path(r"/home/yty/桌面/workspace/VOC2012/")
    
    # datasets preprocss cache path
    h5_drone_train = cache_path / "drone_train.h5"
    h5_drone_val = cache_path / "drone_val.h5"
    h5_drone_test = cache_path / "drone_test.h5"
    h5_drone_train = cache_path / "drone_train.h5"
    h5_drone_test_with_coco = cache_path / "drone_test_with_coco.h5"
    h5_drone_test_with_bird = cache_path / "drone_test_with_bird.h5"
    
    # yolo
    num_class = 1
    batch_size = 8
    yolo_train_epoch = 100
    yolo_val_interval = 1 # 训练几轮验证一次
    
    yolo_train_dataset = h5_drone_train
    yolo_val_dataset = h5_drone_val
    file_yolo_weight = checkpoint_path / f"{model_name}.pth"

    # safe
    safe_mlp_epoch = 20
    safe_val_dataset = h5_drone_val
    safe_cache_detect_result = cache_path / "cache_detected_dataset.pth"

    # attack
    h5_extract_features = model_specific_path / "extract_features.h5"
    # vos
    file_vos_yolo_weight = model_specific_path / f"vos_{model_name}.pth"
