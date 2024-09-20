from functools import cached_property
from typing import Literal

import torch

AttackMethod = Literal["pgd", "fgsm"]


class Config:
    device = torch.device("cuda")
    # yolo
    num_class = 1
    batch_size = 8
    dir_checkpoint = "run/weight"
    file_train_dataset = "run/preprocess/drone_train.h5"
    file_val_dataset = "run/preprocess/drone_val.h5"
    file_yolo_weight = "run/weight/yolo.pth"
    yolo_epoch = 100

    # mlp
    mlp_epoch = 20
    file_detected_base_dataset = "run/preprocess/drone_val.h5"
    file_detected_dataset = "run/preprocess/detected_dataset.pth"
    file_multi_layer_search_summary = "run/summary/multi_layer_search.json"

    # attack
    h5_extract_features = "run/extract_features.h5"
    # vos
    file_vos_yolo_weight = "run/weight/vos_yolo.pth"