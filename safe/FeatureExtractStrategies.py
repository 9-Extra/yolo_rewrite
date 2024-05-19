import torch

from safe.FeatureExtract import FeatureExtract
from yolo.Network import Yolo, C3, FeatureExporter


class ExtractResidual(FeatureExtract):
    def attach(self, network: torch.nn.Module):
        super().attach(network)
        for name, layer in network.named_modules():
            if isinstance(layer, C3):
                self.register_hook_output(name, layer.bottlenecks[-1])


class ExtractConcat(FeatureExtract):
    def attach(self, network: torch.nn.Module):
        super().attach(network)
        for name, layer in network.named_modules():
            if isinstance(layer, FeatureExporter):
                self.register_hook_output(layer.name, layer)

        self.register_hook_output("x23", network.backbone.inner)


if __name__ == '__main__':
    network = Yolo(1)
    extractor = ExtractConcat()
    extractor.attach(network)
    extractor.ready()
    network(torch.rand((1, 3, 256, 256)))
    for n, _ in extractor.get_features().items():
        print(n)
