import torch
from torch.utils.hooks import RemovableHandle


class Strategy:
    def filter(self, name: str, layer: torch.nn.Module) -> bool:
        raise NotImplementedError

    def get_name_set(self, network: torch.nn.Module):
        names = set()
        for name, layer in network.named_modules():
            if self.filter(name, layer):
                names.add(name)

        return names


class FeatureExtract:
    feature_cache: dict[str, torch.Tensor]
    hooks: dict[str, RemovableHandle]
    is_ready: bool

    name_set: set[str]

    def __init__(self, name_set: set[str]):
        self.feature_cache = {}
        self.hooks = {}
        self.is_ready = False
        self.name_set = name_set

        pass

    def reset(self):
        self.detach()
        self.feature_cache.clear()
        self.is_ready = False

    def attach(self, network: torch.nn.Module):
        assert len(self.hooks) == 0, "需要先detach"

        def _hook(module, args):
            assert self.is_ready, "需要先准备"
            self.is_ready = False

        self.hooks["_main"] = network.register_forward_pre_hook(_hook)

        for name, layer in network.named_modules():
            if name in self.name_set:
                self.register_hook_output(name, layer)
            else:
                RuntimeError(f"未知层名称 {name}")

    def ready(self):
        assert len(self.hooks) != 0, "需要与神经网络关联"
        self.feature_cache.clear()
        self.is_ready = True

    def get_features(self) -> dict[str, torch.Tensor]:
        assert not self.is_ready, "还没有收集到数据"
        return self.feature_cache

    def detach(self):
        for h in self.hooks.values():
            h.remove()
        pass
        self.hooks.clear()

    def register_hook_output(self, name: str, layer: torch.nn.Module):
        def _hook(module, args, output):
            self.feature_cache[name] = output

        # assert name not in self.hooks, "名称重复"
        self.hooks[name] = layer.register_forward_hook(_hook)


if __name__ == '__main__':
    import yolo.Network

    network = yolo.Network.Yolo(1)
    exporter = FeatureExtract(set())
    exporter.attach(network)
    exporter.detach()
