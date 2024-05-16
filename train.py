import os

import cv2

import yolo
import torch
from torch.utils.data import DataLoader
from dataset.h5Dataset import H5DatasetYolo
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn, TaskProgressColumn
from yolo.ood_score import build_mlp_classifier


def train(network: torch.nn.Module,
          opt: torch.optim.Optimizer,
          train_loader: DataLoader,
          epochs: int,
          save_path: str,
          save_interval: int
          ):
    os.makedirs(save_path, exist_ok=True)
    device = next(network.parameters()).device

    loss_func = yolo.loss.ComputeLoss(network)

    network.train()
    loss = torch.zeros([1])
    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
    ) as process:
        for epoch in process.track(range(epochs), epochs, description=f"epoch"):
            for i, (img, target) in enumerate(process.track(train_loader, len(train_loader), description=f"Train")):
                # ori =cv2.cvtColor(img[0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                # cv2.imshow("show", ori)
                # cv2.waitKey(0)

                img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
                target = torch.from_numpy(target).to(device, non_blocking=True)

                opt.zero_grad()
                output = network(img)
                loss = loss_func(output, target)
                loss.backward()
                opt.step()

            if epoch % save_interval == 0:
                torch.save({
                    "num_class": network.detect.nc,
                    "network": network.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch
                }, os.path.join(save_path, f"yolo_checkpoint_{epoch}.pth"))
                print(f"Loss: {loss.item()}")

    # torch.save(network.state_dict(), os.path.join(save_path, f"yolo_original.pth"))


def build_ood_eval(network: yolo.Network.NetWork, train_loader: DataLoader):
    print("开始构造ood求值")
    return build_mlp_classifier(network, train_loader)


def main(dataset_dir, checkpoint=None):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    epoch = 0

    dataset = H5DatasetYolo(dataset_dir)

    if checkpoint:
        network, opt = yolo.Network.load_checkpoint(checkpoint)
        num_class = network.detect.nc
        print(f"从模型{os.path.abspath(checkpoint)}开始")
    else:
        num_class = len(dataset.get_label_names())
        network = yolo.Network.NetWork(num_class)
        opt = torch.optim.Adam(network.parameters())

    network.train().to(device, non_blocking=True)

    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True,
                            collate_fn=H5DatasetYolo.collate_fn
                            )

    train(network, opt, dataloader, epoch, "weight", 1)
    ood_evaluator = build_ood_eval(network, dataloader)

    torch.save({
        "num_class": num_class,
        "network": network.state_dict(),
        "ood_evaluator": ood_evaluator.to_static_dict(),
        "epoch": epoch,
        "label_names": dataset.get_label_names()
    }, os.path.join("weight", f"yolo_final_full_20.pth"))


pass

if __name__ == '__main__':
    main("preprocess/pure_drone_train_full.h5", "weight/yolo_checkpoint_1.pth")
