import os

import pytorch_lightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from torch.utils.data import DataLoader

from dataset.h5Dataset import H5DatasetYolo
from schedules.schedule import Config
from vos.vos_yolo import VosYolo


class RichProgressBarTinkered(RichProgressBar):

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


def _trainer(max_epochs: int, fast_dev_run: bool = False):
    checkpoint = ModelCheckpoint(
        save_weights_only=False,
        dirpath="run/checkpoints",
        save_last=True
    )

    return pytorch_lightning.Trainer(
        max_epochs=max_epochs,
        precision="32-true",
        check_val_every_n_epoch=2,
        callbacks=[RichProgressBarTinkered(leave=True), RichModelSummary(max_depth=3), checkpoint],
        default_root_dir="run",
        fast_dev_run=fast_dev_run,
        benchmark=True
    )


def vos_finetune(config: Config):
    torch.set_float32_matmul_precision('medium')

    dataset = config.train_dataset
    vos = VosYolo(config.num_class)
    vos.yolo.load_state_dict(torch.load(config.file_yolo_weight, weights_only=True)['state_dict'])
    # network = torch.compile(network, backend="cudagraphs")

    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True,
                            collate_fn=H5DatasetYolo.collate_fn
                            )

    trainer = _trainer(2)
    # trainer.validate(model=vos, dataloaders=dataloader)
    trainer.fit(model=vos, train_dataloaders=dataloader)

    os.makedirs(os.path.dirname(config.file_vos_yolo_weight), exist_ok=True)
    torch.save(vos.yolo.state_dict(), config.file_vos_yolo_weight)


def vos_finetune_val(config: Config):
    torch.set_float32_matmul_precision('medium')

    train_dataset = config.train_dataset
    val_datasets = [
        H5DatasetYolo(d) for d in
        ["run/preprocess/drone_test.h5",
         "run/preprocess/test_drone_with_bird.h5",
         "run/preprocess/test_drone_with_coco.h5"]
    ]
    vos = VosYolo(config.num_class)
    vos.yolo.load_state_dict(torch.load(config.file_yolo_weight, weights_only=True)['state_dict'])
    # network = torch.compile(network, backend="cudagraphs")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=H5DatasetYolo.collate_fn
    )
    dataloader = [DataLoader(dataset,
                             batch_size=8,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=False,
                             collate_fn=H5DatasetYolo.collate_fn
                             ) for dataset in val_datasets]

    trainer = _trainer(10)
    # trainer.validate(model=vos, dataloaders=train_dataloader)
    # trainer.fit(model=vos, train_dataloaders=train_dataloader)
    trainer.fit(model=vos, train_dataloaders=train_dataloader, val_dataloaders=dataloader)

    os.makedirs(os.path.dirname(config.file_vos_yolo_weight), exist_ok=True)
    torch.save(vos.yolo.state_dict(), config.file_vos_yolo_weight)


if __name__ == '__main__':
    # vos_finetune(Config())
    vos_finetune_val(Config())
