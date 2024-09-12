from functools import cached_property
from typing import Literal

import torch

AttackMethod = Literal["pgd", "fgsm"]


class Config:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # yolo
    num_class = 1
    dir_checkpoint = "run/weight"
    file_train_dataset = "run/preprocess/drone_train.h5"
    file_val_dataset = "run/preprocess/drone_val.h5"
    file_yolo_weight = "run/weight/yolo.pth"
    yolo_epoch = 100

    @cached_property
    def trained_yolo_network(self):
        from yolo.Network import Yolo
        network = Yolo(self.num_class)
        network.load_state_dict(torch.load(self.file_yolo_weight, weights_only=True))
        # network = torch.compile(network)
        return network


    # mlp
    mlp_epoch = 20
    file_detected_base_dataset = "run/preprocess/drone_val.h5"
    file_detected_dataset = "run/preprocess/detected_dataset.pth"
    file_multi_layer_search_summary = "run/summary/multi_layer_search.json"

    @cached_property
    def detected_result_dataset(self):
        import search
        return search.DetectedDataset.load(self.file_detected_dataset)

    # extract_feature
    # dir_extract_features = "run/extract_features"
    # attack_method: AttackMethod = "fgsm"

    # attack
    h5_extract_features = "run/extract_features.h5"

    @property
    def extract_features_database(self):
        from search import ExtractFeatureDatabase
        return ExtractFeatureDatabase(self.h5_extract_features)

    # vos
    file_vos_yolo_weight = "run/weight/vos_yolo.pth"