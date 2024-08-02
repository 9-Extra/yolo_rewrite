from functools import cached_property
from typing import Literal

import torch

AttackMethod = Literal["pgd", "fgsm"]


class Config:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    file_state_record = "run/state.txt"

    # yolo
    num_class = 1
    dir_checkpoint = "run/weight"
    file_train_dataset = "run/preprocess/pure_drone_train_full.h5"
    file_yolo_weight = "run/weight/yolo.pth"
    yolo_epoch = 100

    @cached_property
    def raw_train_dataset(self):
        from dataset.DroneDataset import DroneDataset
        return DroneDataset(r"G:\datasets\DroneTrainDataset", split="train")

    @cached_property
    def train_dataset(self):
        from dataset.h5Dataset import H5DatasetYolo
        return H5DatasetYolo(self.file_train_dataset)

    @cached_property
    def trained_yolo_network(self):
        import yolo
        network = yolo.Network.Yolo.load_from_checkpoint("run/weight/yolo_100.pth", num_class=self.num_class)
        return network

    @cached_property
    def yolo_layer_order(self):
        import yolo
        num_class = self.num_class
        network = yolo.Network.Yolo(num_class)
        name_order = []
        for name, _ in network.named_modules():
            name_order.append(name)
        return name_order

    # mlp
    file_mlp_weight = "run/weight/mlp.pth"
    mlp_epoch = 20
    file_detected_base_dataset = "run/preprocess/detected_base.h5"
    file_detected_dataset = "run/preprocess/detected_dataset.pth"
    file_single_layer_search_summary = "run/summary/single_layer_search.csv"
    file_multi_layer_search_summary = "run/summary/multi_layer_search.json"

    @cached_property
    def raw_detected_base_dataset(self):
        from dataset.DroneDataset import DroneDataset
        from dataset.CocoBird import CocoBird
        from dataset.RawDataset import mix_raw_dataset, delete_all_object
        drone_val = DroneDataset(r"G:\datasets\DroneTrainDataset", split="val")
        coco_bird = CocoBird(r"D:\迅雷下载\train2017", r"D:\迅雷下载\annotations\instances_train2017.json")
        delete_all_object(coco_bird)
        return mix_raw_dataset([drone_val, coco_bird])

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