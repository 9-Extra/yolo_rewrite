import torch
import torchvision
from torch.utils.data import DataLoader
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn

import yolo
from .FeatureExtract import FeatureExtract


class MLP(torch.nn.Module):
    in_dim: int
    feature_name_set: set[str]

    def __init__(self, in_dim, feature_name_set: set[str]):
        super().__init__()
        self.in_dim = in_dim
        self.feature_name_set = feature_name_set
        self.inner = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim // 2),
            # torch.nn.BatchNorm1d(in_dim // 2),
            torch.nn.ReLU(True),

            torch.nn.Linear(in_dim // 2, in_dim // 4),
            # torch.nn.BatchNorm1d(in_dim // 4),
            torch.nn.ReLU(True),

            torch.nn.Linear(in_dim // 4, 1),
            torch.nn.Flatten(0),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.inner(x)

    def score(self, feature_dict: dict[str, torch.Tensor], prediction: list[torch.Tensor]):
        score_list = []
        self.eval()
        feature = peek_relative_feature_batched(feature_dict, prediction)
        score = self.__call__(feature)

        assert score.shape[0] == sum(p.shape[0] for p in prediction)

        return score

    def to_static_dict(self):
        return {
            "in_dim": self.in_dim,
            "feature_name_set": self.feature_name_set,
            "weight": self.state_dict()
        }

    @staticmethod
    def from_static_dict(mlp_dict: dict):
        mlp = MLP(mlp_dict["in_dim"], mlp_dict["feature_name_set"])
        mlp.load_state_dict(mlp_dict["weight"])

        return mlp


def peek_relative_feature_batched(feature_dict: dict[str, torch.Tensor], prediction: list[torch.Tensor]):
    device = prediction[0].device
    # 遍历每一个batch的输出
    result = []
    for batch_id, batch_p in enumerate(prediction):
        collected = peek_relative_feature_single_batch(feature_dict, batch_p, batch_id)
        result.append(collected)

    return torch.cat(result) if len(result) != 0 else torch.empty(0, device=device, dtype=torch.float32)


def peek_relative_feature_single_batch(feature_dict: dict[str, torch.Tensor], prediction: torch.Tensor, batch_id: int):
    device = prediction.device
    # 遍历每一个batch的输出
    result = []

    if prediction.shape[0] == 0:
        return torch.empty(0, device=device)

    # origin_bbox = batch_p[..., 0: 4]
    origin_bbox = prediction[..., 5: 9]

    batch_bbox = torch.empty([prediction.shape[0], 5], dtype=torch.float32, device=device)
    batch_bbox[..., 0] = batch_id
    batch_bbox[..., 1:] = origin_bbox

    collected = []

    for name, feature in feature_dict.items():
        b, c, h, w = feature.shape

        relative_feature = torchvision.ops.roi_align(feature, batch_bbox, (2, 2), w)
        collected.append(relative_feature.flatten(start_dim=1))  # 保留第0维

    collected = torch.cat(collected, dim=1)  # 将来自不同层的特征拼接起来
    assert collected.shape[0] == prediction.shape[0]

    return collected


def peek_relative_feature_to_dict(feature_dict: dict[str, torch.Tensor], prediction: list[torch.Tensor],
                                  out_dict: dict):
    device = prediction[0].device
    # 遍历每一个batch的输出
    for batch_id, batch_p in enumerate(prediction):
        if batch_p.shape[0] == 0:
            # result.append(torch.empty(0, device=device))
            continue

        # origin_bbox = batch_p[..., 0: 4]
        origin_bbox = batch_p[..., 5: 9]

        batch_bbox = torch.empty([batch_p.shape[0], 5], dtype=torch.float32, device=device)
        batch_bbox[..., 0] = batch_id
        batch_bbox[..., 1:] = origin_bbox

        for name, feature in feature_dict.items():
            b, c, h, w = feature.shape

            relative_feature = torchvision.ops.roi_align(feature, batch_bbox, (2, 2), w)
            out_dict.setdefault(name, []).append(relative_feature.flatten(start_dim=1).cpu())  # 保留第0维


@torch.no_grad()
def mlp_build_dataset(network: torch.nn.Module, name_set: set, train_loader: DataLoader,
                      epsilon: float):
    device = next(network.parameters()).device
    loss_func = yolo.loss.ComputeLoss(network)

    network.eval()
    feature_extractor = FeatureExtract(name_set)
    feature_extractor.attach(network)

    positive_features, negative_features = [], []
    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
    ) as process:
        for img, target in process.track(train_loader, len(train_loader), description="收集特征"):
            img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
            target = torch.from_numpy(target).to(device, non_blocking=True)
            img = img.requires_grad_()

            with torch.enable_grad():
                feature_extractor.ready()
                output = network(img)
                loss = loss_func(output, target)

            # FGSM产生对抗样本
            grad = torch.autograd.grad(loss, img)[0]
            attack_sample = (img + torch.sign(grad) * epsilon).clip(0, 1)

            output = network.detect.inference_post_process(output)
            prediction = yolo.non_max_suppression.non_max_suppression(output)

            positive_features.append(peek_relative_feature_batched(feature_extractor.get_features(), prediction))
            feature_extractor.ready()
            _ = network(attack_sample)
            negative_features.append(peek_relative_feature_batched(feature_extractor.get_features(), prediction))

    feature_extractor.detach()

    # 构造MLP训练集
    positive_features = torch.cat(positive_features)
    negative_features = torch.cat(negative_features)
    y = torch.zeros(positive_features.shape[0] + negative_features.shape[0], dtype=torch.float32, device=device)
    y[:positive_features.shape[0]] = 1  # 正样本值为1
    x = torch.cat([positive_features, negative_features])

    assert x.shape[0] == y.shape[0]

    return x, y


@torch.no_grad()
def mlp_build_dataset_separate(network: torch.nn.Module, name_set: set, train_loader: DataLoader,
                               epsilon: float):
    device = next(network.parameters()).device
    loss_func = yolo.loss.ComputeLoss(network)

    network.eval()

    feature_extractor = FeatureExtract(name_set)
    feature_extractor.attach(network)

    positive_features, negative_features = {}, {}
    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
    ) as process:
        for i, (img, target) in enumerate(process.track(train_loader, len(train_loader), description="收集特征")):
            img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
            target = torch.from_numpy(target).to(device, non_blocking=True)
            img = img.requires_grad_()

            with torch.enable_grad():
                feature_extractor.ready()
                output = network(img)
                loss = loss_func(output, target)

            # FGSM产生对抗样本
            grad = torch.autograd.grad(loss, img)[0]
            attack_sample = (img + torch.sign(grad) * epsilon).clip(0, 1)

            output = network.detect.inference_post_process(output)
            prediction = yolo.non_max_suppression.non_max_suppression(output)

            peek_relative_feature_to_dict(feature_extractor.get_features(), prediction, out_dict=positive_features)
            feature_extractor.ready()
            _ = network(attack_sample)
            peek_relative_feature_to_dict(feature_extractor.get_features(), prediction, out_dict=negative_features)

            if i % 128 == 127:  # 拼接成块
                block = i // 128
                positive_features = {k: [*v[:block], torch.cat(v[block:])] for k, v in positive_features.items()}
                negative_features = {k: [*v[:block], torch.cat(v[block:])] for k, v in negative_features.items()}

    feature_extractor.detach()

    return positive_features, negative_features


def train_mlp(mlp: MLP, train_dataset, batch_size: int, epoch: int, device: torch.device):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    mlp = mlp.to(device).train()
    opt = torch.optim.Adam(mlp.parameters())
    loss_func = torch.nn.BCELoss()

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[acc]:%}"),
    ) as progress:
        train_task = progress.add_task("train", acc=0)
        epoch_task = progress.add_task("epoch", acc=0)
        for _ in progress.track(range(epoch), task_id=train_task, description="训练MLP"):
            progress.reset(epoch_task)

            for x, y in progress.track(train_dataloader, len(train_dataloader), task_id=epoch_task,
                                       description="epoch"):
                x = x.to(device)
                y = y.to(device)

                opt.zero_grad()
                output = mlp(x)
                loss = loss_func(output, y)
                loss.backward()
                opt.step()

                # acc
                p = torch.zeros_like(output)
                p[output > 0.5] = 1
                val_acc = torch.count_nonzero(p == y).item() / y.shape[0]
                progress.update(epoch_task, acc=val_acc)

        progress.reset(epoch_task)

        mlp.eval()
        val_acc = torch.tensor(0, device=device)
        count = 0
        with torch.no_grad():
            for x, y in progress.track(train_dataloader, len(train_dataloader), task_id=epoch_task, description="val"):
                output = mlp.forward(x)
                p = torch.zeros_like(output)
                p[output > 0.5] = 1
                val_acc += torch.count_nonzero(p == y)
                count += y.shape[0]

            val_acc = val_acc.item() / count
        return val_acc
