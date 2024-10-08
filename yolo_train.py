import os.path

import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from yolo.Network import Yolo
from config import Config
import torch
from torch.utils.data import DataLoader

from dataset.h5Dataset import H5DatasetYolo


class RichProgressBarTinkered(RichProgressBar):

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


def _trainer(max_epochs: int, fast_dev_run: bool = False):

    default_root_dir = "run"

    checkpoint = ModelCheckpoint(
        monitor="train_loss",
        save_last=True,
        every_n_epochs=5,
    )

    return pytorch_lightning.Trainer(
        max_epochs=max_epochs,
        precision="32-true",
        val_check_interval=0.1,
        callbacks=[RichProgressBarTinkered(leave=True), RichModelSummary(max_depth=3), checkpoint],
        default_root_dir=default_root_dir,
        logger=TensorBoardLogger(save_dir=default_root_dir, name="yolo_logs"),
        fast_dev_run=fast_dev_run,
        benchmark=True
    )


def train(config: Config, skip_if_exists: bool = True):
    torch.set_float32_matmul_precision('medium')
    if skip_if_exists and os.path.isfile(config.file_yolo_weight):
        print("Weight exists. Skip yolo training.")
        return

    num_class = config.num_class
    network = Yolo(num_class)
    # network = torch.compile(network, backend="cudagraphs")

    train_dataloader = DataLoader(H5DatasetYolo(config.file_train_dataset),
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  collate_fn=H5DatasetYolo.collate_fn
                                  )

    val_dataloader = DataLoader(H5DatasetYolo(config.file_val_dataset),
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                collate_fn=H5DatasetYolo.collate_fn
                                )

    trainer = _trainer(config.yolo_epoch)
    trainer.fit(model=network, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    if not trainer.interrupted: # 如果中断则不保存
        os.makedirs(os.path.dirname(config.file_yolo_weight), exist_ok=True)
        torch.save(network.state_dict(), config.file_yolo_weight)


pass

def val(config: Config):
    torch.set_float32_matmul_precision('medium')

    num_class = config.num_class
    network = Yolo(num_class)
    network.load_state_dict(torch.load(config.file_yolo_weight, weights_only=True))
    # network = torch.compile(network, backend="cudagraphs")
    val_dataloader = DataLoader(H5DatasetYolo(config.file_val_dataset),
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                collate_fn=H5DatasetYolo.collate_fn
                                )

    trainer = _trainer(config.yolo_epoch)
    result = trainer.validate(model=network, dataloaders=val_dataloader)

    if not trainer.interrupted:
        print(result)

pass

if __name__ == '__main__':
    train(Config(), skip_if_exists=False)
