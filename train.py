import os

import yolo
from schedules.schedule import Config
import torch
from torch.utils.data import DataLoader

from dataset.h5Dataset import H5DatasetYolo
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn, TaskProgressColumn
from safe.safe_method import mlp_build_dataset, MLP, train_mlp


def build_mlp_classifier(network, train_loader: DataLoader, name_set: set, attack_method, epsilon: float,
                         epoch=50):
    device = next(network.parameters()).device
    x, y = mlp_build_dataset(network, name_set, train_loader, attack_method, epsilon)
    feature_dim = x.shape[1]
    print("训练样本数：", x.shape[0])
    print("特征长度：", feature_dim)
    dataset = torch.utils.data.TensorDataset(x, y)

    mlp = MLP(feature_dim, name_set)
    acc = train_mlp(mlp, dataset, 64, epoch, device)
    print(f"{acc=}")

    return mlp


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
    ) as progress:
        train_task = progress.add_task("train")
        epoch_task = progress.add_task("epoch")

        for epoch in progress.track(range(epochs), epochs, train_task, description=f"epoch"):
            for i, (img, target) in enumerate(
                    progress.track(train_loader, len(train_loader), epoch_task, description=f"train")):
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

            # progress.reset(epoch_task)

    # torch.save(network.state_dict(), os.path.join(save_path, f"yolo_original.pth"))


def build_ood_eval(network: yolo.Network.Yolo, name_set: set, train_loader: DataLoader, attack_method, epsilon):
    print("开始构造ood求值")
    return build_mlp_classifier(network,
                                train_loader,
                                name_set,
                                attack_method,
                                epsilon
                                )


def main(config: Config):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    dataset = config.train_dataset
    num_class = config.num_class
    network = yolo.Network.Yolo(num_class)
    network.to(device, non_blocking=True)
    opt = config.yolo_optimizer(network.parameters())

    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True,
                            collate_fn=H5DatasetYolo.collate_fn
                            )

    train(network, opt, dataloader, config.yolo_epoch, config.dir_checkpoint, 1)
    del opt

    os.makedirs(os.path.dirname(config.file_yolo_weight), exist_ok=True)
    torch.save({
        "network": network.state_dict(),
    }, config.file_yolo_weight)

    os.makedirs(os.path.dirname(config.file_mlp_weight), exist_ok=True)
    name_set = {"backbone.inner.25.cv3.conv", "backbone.inner.29.cv3.conv"}
    ood_evaluator = build_ood_eval(network, name_set, dataloader, config.attack_method, config.mlp_epsilon)
    torch.save(ood_evaluator.to_static_dict(), config.file_yolo_weight)


pass

if __name__ == '__main__':
    main(Config())
