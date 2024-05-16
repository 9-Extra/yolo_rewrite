import typing

import torch
import torchvision
from torch.utils.data import DataLoader
from rich.progress import track, Progress, BarColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn

import yolo


class MLP(torch.nn.Module):
    in_dim: int

    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.inner = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(True),

            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(True),

            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(True),

            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(True),

            torch.nn.Linear(256, 1),
            torch.nn.Flatten(0),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.inner(x)

    def score(self, feature_dict: dict[str, torch.Tensor], prediction: list[torch.Tensor]):
        scores = []
        self.eval()
        for p in prediction:
            if p.shape[0] == 0:
                scores.append(torch.empty(0))
                continue
            feature = self.peek_relative_feature(feature_dict, [p])
            score = self.forward(feature)

            assert score.shape[0] == p.shape[0]

            scores.append(score)

        return scores

    def to_static_dict(self):
        return {
            "in_dim": self.in_dim,
            "weight": self.state_dict()
        }

    @staticmethod
    def from_static_dict(mlp_dict: dict):
        mlp = MLP(mlp_dict["in_dim"])
        mlp.load_state_dict(mlp_dict["weight"])

        return mlp

    @staticmethod
    def peek_relative_feature(feature_dict: dict[str, torch.Tensor], prediction: list[torch.Tensor]):
        device = prediction[0].device
        # 遍历每一个batch的输出
        result = []
        for batch_id, batch_p in enumerate(prediction):
            if batch_p.shape[0] == 0:
                # result.append(torch.empty(0, device=device))
                continue

            origin_bbox = batch_p[..., 5: 9]

            batch_bbox = torch.empty([batch_p.shape[0], 5], dtype=torch.float32, device=device)
            batch_bbox[..., 0] = batch_id
            batch_bbox[..., 1:] = origin_bbox

            collected = []

            for name, feature in feature_dict.items():
                b, c, h, w = feature.shape

                relative_feature = torchvision.ops.roi_pool(feature, batch_bbox, (2, 2), w)
                collected.append(relative_feature.flatten(start_dim=1))  # 保留第0维

            collected = torch.cat(collected, dim=1)  # 此时collected第0维为
            assert collected.shape[0] == batch_p.shape[0]

            result.append(collected)

        return torch.cat(result) if len(result) != 0 else torch.empty(0, device=device, dtype=torch.float32)


@torch.no_grad()
def mlp_build_dataset(network, train_loader: DataLoader, epsilon=0.1):
    device = next(network.parameters()).device
    loss_func = yolo.loss.ComputeLoss(network)

    positive_features, negative_features = [], []
    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
    ) as process:
        for img, target in process.track(train_loader, description="收集特征"):
            img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
            target = torch.from_numpy(target).to(device, non_blocking=True)
            img = img.requires_grad_()

            extract_features = {}
            with torch.enable_grad():
                output = network(img, extract_features)
                loss = loss_func(output, target)

            # FGSM产生对抗样本
            grad = torch.autograd.grad(loss, img)[0]
            attack_sample = (img + torch.sign(grad) * epsilon).clip(0, 1)

            output = network.detect.inference_post_process(output)
            prediction = yolo.non_max_suppression.non_max_suppression(output)

            positive_features.append(MLP.peek_relative_feature(extract_features, prediction))

            extract_features = {}
            _ = network(attack_sample, extract_features)
            negative_features.append(MLP.peek_relative_feature(extract_features, prediction))

    # 构造MLP训练集
    positive_features = torch.cat(positive_features)
    negative_features = torch.cat(negative_features)
    y = torch.zeros(positive_features.shape[0] + negative_features.shape[0], dtype=torch.float32, device=device)
    y[:positive_features.shape[0]] = 1  # 正样本值为1
    x = torch.cat([positive_features, negative_features])

    assert x.shape[0] == y.shape[0]

    return x, y


def build_mlp_classifier(network, train_loader: DataLoader, epsilon=0.05, epoch=30):
    device = next(network.parameters()).device
    x, y = mlp_build_dataset(network, train_loader, epsilon)
    print("训练样本数：", x.shape[0])
    print("特征长度：", x.shape[1])
    dataset = torch.utils.data.TensorDataset(x, y)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64)

    mlp = MLP(x.shape[1])
    mlp = mlp.to(device).train()
    opt = torch.optim.Adam(mlp.parameters())
    loss_func = torch.nn.BCELoss()

    with Progress() as progress:
        train_task = progress.add_task("train")
        epoch_task = progress.add_task("epoch")
        for e in progress.track(range(epoch), task_id=train_task, description="训练MLP"):
            network.train()
            for x, y in progress.track(train_dataloader, task_id=epoch_task, description="epoch"):
                opt.zero_grad()
                output = mlp.forward(x)
                loss = loss_func(output, y)
                loss.backward()
                opt.step()

            network.eval()
            val_acc = torch.tensor(0, device=device)
            with torch.no_grad():
                for x, y in val_dataloader:
                    output = mlp.forward(x)
                    p = torch.zeros_like(output)
                    p[output > 0.5] = 1
                    val_acc += torch.count_nonzero(p == y)

            val_acc = val_acc.item() / len(val_dataset)

            print(f"epoch {e}: acc = {val_acc:%}")

    return mlp
