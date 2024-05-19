import torch
from torch.utils.hooks import RemovableHandle


class FeatureExtract:
    feature_cache: dict[str, torch.Tensor]
    hooks: list[RemovableHandle]
    is_ready: bool

    def __init__(self):
        self.feature_cache = {}
        self.hooks = []
        self.is_ready = False

        pass

    def attach(self, network: torch.nn.Module):
        assert len(self.hooks) == 0, "需要先detach"

        def _hook(module, args):
            assert self.is_ready, "需要先准备"
            self.is_ready = False

        self.hooks.append(network.register_forward_pre_hook(_hook))

    def ready(self):
        assert len(self.hooks) != 0, "需要与神经网络关联"
        self.feature_cache.clear()
        self.is_ready = True

    def get_features(self) -> dict[str, torch.Tensor]:
        assert not self.is_ready, "还没有收集到数据"
        return self.feature_cache

    def detach(self):
        for h in self.hooks:
            h.remove()
        pass
        self.hooks.clear()

    def register_hook_output(self, name: str, layer: torch.nn.Module):
        def _hook(module, args, output):
            self.feature_cache[name] = output

        self.hooks.append(layer.register_forward_hook(_hook))


if __name__ == '__main__':
    import yolo.Network

    network = yolo.Network.Yolo(1)
    exporter = FeatureExtract()
    exporter.attach(network)
    exporter.detach()
