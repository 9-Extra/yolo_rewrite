import torch

from .FeatureExtract import FeatureExtract


class OODEvaluator(torch.nn.Module):
    feature_extractor: FeatureExtract

    def __init__(self, feature_extractor: FeatureExtract):
        super().__init__()
        self.feature_extractor = feature_extractor

    def score(self, feature_dict: dict[str, torch.Tensor], prediction: list[torch.Tensor]):
        raise NotImplementedError

    def to_static_dict(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def from_static_dict(mlp_dict: dict) -> "OODEvaluator":
        raise NotImplementedError

    @staticmethod
    def peek_relative_feature(feature_dict: dict[str, torch.Tensor], prediction: list[torch.Tensor]):
        raise NotImplementedError
