import torch
from torch.nn import Module
import einops


class FeatureExporter(Module):
    storage: dict
    name: str
    clone: bool

    def __init__(self, storage: dict, name: str, clone=False):
        super().__init__()
        self.storage = storage
        self.name = name
        self.clone = clone

    def forward(self, x: torch.Tensor):
        self.storage[self.name] = x.clone() if self.clone else x
        return x


class Conv(Module):
    default_act = torch.nn.SiLU(True)  # default activation
    inner: Module

    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, act=True):
        super().__init__()
        p = self._auto_pad(kernel_size, dilation) if padding is None else padding
        self.inner = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride, p, groups=groups,
                            dilation=dilation, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            self.default_act if act else torch.nn.Identity()
        )

    def forward(self, x):
        return self.inner(x)

    @staticmethod
    def _auto_pad(k, d=1):  # kernel, padding, dilation
        # Pad to 'same' shape outputs
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        return p


class Bottleneck(Module):
    # Standard bottleneck
    def __init__(self, in_channel, out_channel, shortcut=True, groups=1,
                 e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        hidden_channel = int(out_channel * e)
        self.cv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.cv2 = Conv(hidden_channel, out_channel, 3, 1, groups=groups)
        self.add = shortcut and in_channel == out_channel

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        return x + x1 if self.add else x1


class C3(Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, in_channel, out_channel, num=1, shortcut=True, g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channel = int(out_channel * e)  # hidden channels
        self.cv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.cv2 = Conv(in_channel, hidden_channel, 1, 1)
        self.bottlenecks = torch.nn.Sequential(
            *(Bottleneck(hidden_channel, hidden_channel, shortcut, g, e=1.0) for _ in range(num)))
        self.cv3 = Conv(2 * hidden_channel, out_channel, 1)  # optional act=FReLU(c2)

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        x1 = self.bottlenecks(self.cv1(x))
        x2 = self.cv2(x)
        return self.cv3(torch.cat((x1, x2), 1))


class SPPF(Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, in_channel, out_channel, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = in_channel // 2  # hidden channels
        self.cv1 = Conv(in_channel, c_, 1, 1)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(c_ * 4, out_channel, 1, 1)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)
        y1 = self.max_pool(x)
        y2 = self.max_pool(y1)
        y3 = self.max_pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class Detect(Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction

    def __init__(self, nc: int, anchors: list, ch: list):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = torch.nn.ModuleList(torch.nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        s = 256  # 2x min stride
        self.stride = torch.tensor([s / x for x in (32, 16, 8)])  # [8, 16, 32]
        self.anchors /= self.stride.view(-1, 1, 1)  # 将anchor大小映射到基于grid的比例

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # 检测头，就是1x1卷积，输出an * (nc + 5)的特征图
            # 然后，将an（每层的anchor数）和nc + 5（一个物体的特征数）单独切分到两个维度里, x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = einops.rearrange(x[i], "bs (na no) ny nx -> bs na ny nx no", na=self.na, no=self.no)
            x[i] = x[i].contiguous()
        return x

    def inference_post_process(self, x):
        # 网络输出的x是基于anchor的偏移量，需要转换成基于整个图像的坐标
        z = []
        for i in range(self.nl):
            bs, _, ny, nx, _ = x[i].shape

            # 对于不同大小的图片，检测头的输出的特征图大小是不一样的，所以需要根据特征图的大小来生成网格
            # 如果一样大就用缓存的
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            # grid是网格的偏移量，anchor_grid是网格的大小
            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
            xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
            wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh

            y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, self.na * nx * ny, self.no))

        return torch.cat(z, 1)

    def _make_grid(self, nx: int, ny: int, i: int):
        """
        :param nx: x方向的anchor数量
        :param ny: y方向的anchor数量
        :param i: 要生成的网络对应的anchor层
        :return: grid是网格的偏移量，anchor_grid是网格的大小
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class BackBone(Module):
    feature_storage: dict

    def __init__(self):
        super().__init__()

        self.feature_storage = {}
        self.inner = torch.nn.Sequential(
            Conv(3, 32, 6, 2, 2),
            Conv(32, 64, 3, 2),
            C3(64, 64, 2),
            Conv(64, 128, 3, 2),
            FeatureExporter(self.feature_storage, "b4"),
            C3(128, 128, 4),
            Conv(128, 256, 3, 2),
            FeatureExporter(self.feature_storage, "b6"),
            C3(256, 256, 6),
            Conv(256, 512, 3, 2),
            C3(512, 512, 2),
            SPPF(512, 512, 5)
        )

    def forward(self, x):
        x = self.inner(x)
        b4 = self.feature_storage.get("b4")
        b6 = self.feature_storage.get("b6")
        self.feature_storage.clear()
        return x, b4, b6


_ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326]
]


class NetWork(Module):
    feature_storage: dict

    def __init__(self, num_class: int):
        super(NetWork, self).__init__()
        self.feature_storage = {}
        self.backbone = BackBone()

        self.head = torch.nn.ModuleList([
            Conv(512, 256, 1, 1),
            torch.nn.Upsample(None, 2, "nearest"),
            # cat backbone P4
            C3(256 + 256, 256, 2, False),
            Conv(256, 128, 1, 1),
            torch.nn.Upsample(None, 2, "nearest"),
            # cat backbone P3
            C3(128 + 128, 128, 2, False),
            Conv(128, 128, 3, 2),
            # cat head P4
            C3(256, 256, 2, False),
            Conv(256, 256, 3, 2),
            # cat head P5
            C3(512, 512, 2, False)
        ])

        self.detect = Detect(nc=num_class, anchors=_ANCHORS, ch=[128, 256, 512])

    def forward(self, x):
        x, b4, b6 = self.backbone(x)
        x10 = self.head[0](x)  # 10
        x = self.head[1](x10)
        x = torch.cat([x, b6], 1)
        x = self.head[2](x)  # 13
        x14 = self.head[3](x)
        x = self.head[4](x14)
        x = torch.cat([x, b4], 1)
        x17 = self.head[5](x)  # 17
        x = self.head[6](x17)
        x = torch.cat([x, x14], 1)
        x20 = self.head[7](x)  # 20
        x = self.head[8](x20)
        x = torch.cat([x, x10], 1)
        x23 = self.head[9](x)  # 23
        x = self.detect([x17, x20, x23])

        return x


if __name__ == '__main__':
    network = NetWork(80)
    network(torch.rand((1, 3, 256, 256)))
    print(network)
