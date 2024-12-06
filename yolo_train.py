import os.path

import pytorch_lightning
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

import config
from yolo.network import Yolo
from config import Config
import torch
from torch.utils.data import DataLoader

from dataset.h5Dataset import H5DatasetYolo


class RichProgressBarTinkered(RichProgressBar):

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


def _trainer(config: Config, fast_dev_run: bool = False):

    checkpoint = ModelCheckpoint(
        monitor="train_loss",
        save_last=True,
        every_n_epochs=5,
    )

    return pytorch_lightning.Trainer(
        max_epochs=config.yolo_train_epoch,
        precision="32-true",
        check_val_every_n_epoch=config.yolo_val_interval,
        callbacks=[
            RichProgressBarTinkered(leave=True),
            RichModelSummary(max_depth=3),
            checkpoint,
        ],
        default_root_dir=config.run_path,
        logger=[
            TensorBoardLogger(config.log_path, name="yolo_logs_tensorboard"),
            CSVLogger(config.log_path, name="yolo_logs_csv"),
        ],
        fast_dev_run=fast_dev_run,
        benchmark=True,
    )


def train_val(config: Config, skip_train_if_exists: bool = True):
    torch.set_float32_matmul_precision("high")

    num_class = config.num_class
    network = Yolo(num_class)
    # network = torch.compile(network, backend="cudagraphs")

    trainer = _trainer(config)

    val_dataloader = DataLoader(
        H5DatasetYolo(config.yolo_val_dataset),
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=H5DatasetYolo.collate_fn,
    )

    # assert (
    #     len(val_dataloader.dataset.get_label_names()) == num_class
    # ), "数据集类型数需要和网络一致"

    if not (skip_train_if_exists and os.path.isfile(config.file_yolo_weight)):
        # do train
        train_dataloader = DataLoader(
            H5DatasetYolo(config.yolo_train_dataset),
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=H5DatasetYolo.collate_fn,
        )

        # assert (
        #     len(val_dataloader.dataset.get_label_names()) == num_class
        # ), "数据集类型数需要和网络一致"

        trainer.fit(
            model=network,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        if trainer.interrupted:  # 如果中断则不保存
            exit(-1)

        os.makedirs(os.path.dirname(config.file_yolo_weight), exist_ok=True)
        torch.save(network.state_dict(), config.file_yolo_weight)
    else:
        print("权重存在，跳过训练")
    pass

    train_dataloader = DataLoader(
        H5DatasetYolo(config.yolo_train_dataset),
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=H5DatasetYolo.collate_fn,
    ) # 这次不需要shuffle

    # val
    network.load_state_dict(
        torch.load(config.file_yolo_weight, weights_only=True)
    )  # 都加载，不少那几毫秒
    
    # 在验证集上验证
    result = trainer.validate(model=network, dataloaders=val_dataloader)
    
    if trainer.interrupted:
        exit(-1)

    print(result)
    
    # 在训练集上验证
    result = trainer.validate(model=network, dataloaders=train_dataloader)
    
    if trainer.interrupted:
        exit(-1)

    print(result)


pass


if __name__ == "__main__":
    train_val(Config.from_profile("./profiles/pure_voc.toml"))
