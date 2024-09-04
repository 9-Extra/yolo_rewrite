import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar

import yolo
from schedules.schedule import Config
import torch
from torch.utils.data import DataLoader

from dataset.h5Dataset import H5DatasetYolo


class RichProgressBarTinkered(RichProgressBar):

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


def _trainer(max_epochs: int, fast_dev_run: bool = False):
    checkpoint = ModelCheckpoint(
        save_weights_only=True,
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


def main(config: Config):
    torch.set_float32_matmul_precision('medium')

    dataset = config.train_dataset
    num_class = config.num_class
    network = yolo.Network.Yolo(num_class)
    # network = torch.compile(network, backend="cudagraphs")

    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True,
                            collate_fn=H5DatasetYolo.collate_fn
                            )

    trainer = _trainer(config.yolo_epoch)
    trainer.fit(model=network, train_dataloaders=dataloader)

    trainer.save_checkpoint(config.file_yolo_weight, weights_only=True)

pass

if __name__ == '__main__':
    main(Config())
