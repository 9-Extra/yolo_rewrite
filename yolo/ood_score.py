import torch
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


def build_residual_score(network, train_loader: DataLoader, assume_center=True):
    device = next(network.parameters()).device
    with torch.no_grad():
        network.eval()
        network.detect.output_odd_feature = True
        target_builder = yolo.loss.ComputeLoss(network)

        features_id_train = [[] for _ in range(network.detect.nl)]  # list for each layer

        for img, target in tqdm.tqdm(train_loader):
            img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
            target = torch.from_numpy(target).to(device, non_blocking=True)

            output, feature = network(img)
            indexed_target = target_builder.build_targets(output, target)
            for f, t, feature_list in zip(feature, indexed_target, features_id_train):
                b, _, gj, gi, _, _, _ = t
                if b.shape[0] != 0:  # 忽略没有target的情况
                    f = f[b, gj, gi]  # 大部分为背景的特征没有意义，只取检测目标的特征
                    feature_list.append(f)

        residuals = network.detect.ood_evaluator
        for f, evaluator in zip(features_id_train, residuals):
            f = torch.concat(f, dim=0)
            evaluator: ResidualScore
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
