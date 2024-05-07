import typing

import torch
import torchvision
from torch.utils.data import DataLoader
import tqdm

import yolo


class ResidualScore(torch.nn.Module):
    in_feature_dim: int
    ns_dim: int
    principal_space: torch.Tensor
    u: torch.Tensor

    def __init__(self, in_feature_dim: int, ns_dim: int, device=None):
        super().__init__()
        self.in_feature_dim = in_feature_dim
        self.ns_dim = ns_dim
        self.register_buffer("principal_space", torch.empty([in_feature_dim, ns_dim], device=device))
        self.register_buffer("u", torch.zeros([in_feature_dim], device=device))

    def forward(self, feature: torch.Tensor):
        with torch.no_grad():
            score = -torch.norm(torch.matmul(feature - self.u, self.principal_space), dim=-1)

        return score


def peek_relative_feature_prediction(feature_dict: dict[str, torch.Tensor], prediction: list[torch.Tensor], num_layer: int):
    """
    在提出prediction相关特征时保证关系不丢失
    :param feature_dict:
    :param prediction:
    :param num_layer:
    :return:
    """
    for name, feature in feature_dict.items():
        b, c, h, w = feature.shape
        device = feature.device
        gain = torch.tensor([w, h, w, h], device=device)

        anchor_bboxes = [[] for _ in range(num_layer)]
        for batch_id, batch_p in enumerate(prediction):
            if batch_p.shape[0] == 0:
                continue

            for i in range(num_layer):
                p = batch_p[batch_p[..., 9] == i]  # 同layer产生的prediction

                if p.shape[0] == 0:
                    continue

                batch_bbox = torch.empty([p.shape[0], 5], dtype=torch.float32, device=device)
                batch_bbox[..., 0] = batch_id

                origin_bbox = p[..., 5: 9]
                batch_bbox[..., 1:] = origin_bbox * gain
                anchor_bboxes[i].append(batch_bbox)

        result = []
        for batch_bbox in anchor_bboxes:
            if len(batch_bbox) != 0:
                batch_bbox = torch.cat(batch_bbox, dim=0)
                size = int((batch_bbox[0, 3] - batch_bbox[0, 1]).item())
                if size > 0:
                    relative_feature = torchvision.ops.roi_pool(feature, batch_bbox, size)
                    result.append(relative_feature)
                else:
                    result.append(torch.empty(0))
            else:
                result.append(torch.empty(0))

    return result


def _peek_relative_feature(feature: torch.Tensor, prediction: list[torch.Tensor], num_layer: int):
    b, c, h, w = feature.shape
    device = feature.device
    gain = torch.tensor([w, h, w, h], device=device)

    anchor_bboxes = [[] for _ in range(num_layer)]
    for batch_id, batch_p in enumerate(prediction):
        if batch_p.shape[0] == 0:
            continue

        for i in range(num_layer):
            p = batch_p[batch_p[..., 9] == i]  # 同layer产生的prediction

            if p.shape[0] == 0:
                continue

            batch_bbox = torch.empty([p.shape[0], 5], dtype=torch.float32, device=device)
            batch_bbox[..., 0] = batch_id

            origin_bbox = p[..., 5: 9]
            batch_bbox[..., 1:] = origin_bbox * gain
            anchor_bboxes[i].append(batch_bbox)

    result = []
    for batch_bbox in anchor_bboxes:
        if len(batch_bbox) != 0:
            batch_bbox = torch.cat(batch_bbox, dim=0)
            size = int((batch_bbox[0, 3] - batch_bbox[0, 1]).item())
            if size > 0:
                relative_feature = torchvision.ops.roi_pool(feature, batch_bbox, size)
                result.append(relative_feature)
            else:
                result.append(torch.empty(0))
        else:
            result.append(torch.empty(0))

    return result


def collect_features_single_infer(extract_features: dict, prediction: list, num_layer: int):
    record_features = [{} for _ in range(num_layer)]
    for k, v in extract_features.items():
        relative_feature = _peek_relative_feature(v, prediction, num_layer)
        for layer_id, layer_feature in enumerate(relative_feature):
            if layer_feature.shape[0] != 0:
                record_features[layer_id][k] = layer_feature

    return record_features


def collect_features(network, train_loader: DataLoader):
    device = next(network.parameters()).device
    num_layer = network.detect.nl
    with torch.no_grad():
        network.eval()

        record_features = [{} for _ in range(num_layer)]

        for img, target in tqdm.tqdm(train_loader):
            img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
            # target = torch.from_numpy(target).to(device, non_blocking=True)

            extract_features = {}
            output = network(img, extract_features)
            output = network.detect.inference_post_process(output)
            prediction = yolo.non_max_suppression.non_max_suppression(output)

            r = collect_features_single_infer(extract_features, prediction, num_layer)

            for i in range(num_layer):
                for k, v in r[i].items():
                    record_features[i].setdefault(k, []).append(v)

        for f_dict in record_features:
            for k in f_dict.keys():
                f_dict[k] = torch.cat(f_dict[k], dim=0)

    return record_features


def build_residual_score(network, train_loader: DataLoader, assume_center=True) -> list[typing.Optional[ResidualScore]]:
    device = next(network.parameters()).device
    num_layer = network.detect.nl
    with torch.no_grad():
        record_features = collect_features(network, train_loader)

        flatten_features = []
        for f_dict in record_features:
            if len(f_dict) != 0:
                feature = torch.cat([f.flatten(1) for f in f_dict.values()], dim=-1)
                print(f"样本数{feature.shape[0]}")
                flatten_features.append(feature)
            else:
                flatten_features.append(None)
                print("有的layer一个正样本也没有，无法生成判别器")

        residuals = []
        for f in flatten_features:
            if f is None:
                residuals.append(None)
                continue

            input_dim = f.shape[1]
            if input_dim >= 1024:
                ns_dim = 512
            else:
                ns_dim = input_dim // 2
            evaluator = ResidualScore(input_dim, ns_dim, device=device)

            feature_dim = evaluator.ns_dim
            print(f'{feature_dim=}')

            if not assume_center:
                evaluator.u = torch.mean(f, list(range(0, len(f.shape) - 1)))

            # computing principal space
            x = f - evaluator.u
            covariance = x.T @ x / x.shape[0]  # 求协方差矩阵
            eig_vals, eigen_vectors = torch.linalg.eig(covariance)
            _, idx = torch.topk(eig_vals.real, feature_dim)
            principal_space = eigen_vectors.real[idx].T.contiguous()
            evaluator.principal_space = principal_space

            residuals.append(evaluator)

        return residuals
