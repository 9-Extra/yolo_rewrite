import tomllib
from typing import Any, Literal
from pathlib import Path
import dataclasses

import torch

AttackMethod = Literal["pgd", "fgsm"]


@dataclasses.dataclass(slots=True)
class Config:
    device = torch.device("cuda")

    # main directory
    run_path: Path = dataclasses.field(init=False)
    cache_path: Path = dataclasses.field(init=False)

    model_name: str
    model_specific_path: Path = dataclasses.field(init=False)
    checkpoint_path: Path = dataclasses.field(init=False)
    log_path: Path = dataclasses.field(init=False)
    summary_path: Path = dataclasses.field(init=False)

    # yolo
    num_class: int
    # num_class = 79
    batch_size: int
    img_size: tuple[int, int] = dataclasses.field(init=False)
    yolo_train_epoch: int
    yolo_val_interval: int  # 训练几轮验证一次

    yolo_train_dataset: str  # h5
    yolo_val_dataset: str
    file_yolo_weight: Path = dataclasses.field(init=False)

    # safe
    safe_mlp_epoch: int = dataclasses.field(init=False)
    safe_val_dataset: Path
    safe_cache_detect_result: Path = dataclasses.field(init=False)

    # attack
    h5_extract_features: Path = dataclasses.field(init=False)
    # vos
    file_vos_yolo_weight: Path = dataclasses.field(init=False)

    def __post_init__(self):
        self.run_path = Path("run")
        self.cache_path = Path("/mnt/panpan/tmp") / "preprocess"

        self.model_specific_path = self.run_path / self.model_name
        self.checkpoint_path = self.model_specific_path / "weight"
        self.log_path = self.model_specific_path / "log"
        self.summary_path = self.model_specific_path / "summary"
        
        # datasets
        self.yolo_train_dataset = self.cache_path / self.yolo_train_dataset
        self.yolo_val_dataset = self.cache_path / self.yolo_val_dataset

        self.img_size = (640, 640)
        self.file_yolo_weight = self.checkpoint_path / f"{self.model_name}.pth"

        # safe
        self.safe_mlp_epoch = 20

        self.safe_cache_detect_result = (
            self.cache_path / f"{self.model_name}_cache_detected_dataset.pth"
        )

        self.h5_extract_features = (
            self.cache_path / f"{self.model_name}_extract_features.h5"
        )
        # vos
        self.file_vos_yolo_weight: Path = (
            self.model_specific_path / f"vos_{self.model_name}.pth"
        )

    @staticmethod
    def from_profile(toml_path: str = None, **kw_args):
        toml_dict: dict = (
            tomllib.load(open(toml_path, "rb")) if toml_path is not None else {}
        )
        return Config(
            **dict(
                model_name=toml_dict["model_name"],
                num_class=toml_dict["num_class"],
                batch_size=toml_dict.get("batch_size", 64),
                yolo_train_epoch=toml_dict.get("train_epoch", 100),
                yolo_val_interval=toml_dict.get("val_interval", 1),
                yolo_train_dataset=toml_dict["train_dataset"],
                yolo_val_dataset=toml_dict["val_dataset"],
                safe_val_dataset=toml_dict.get(
                    "safe_val_dataset", toml_dict["val_dataset"]
                ),
                **kw_args,
            )
        )
