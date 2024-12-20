import functools

import numpy
import torch
from torch.nn import Module
from collections import OrderedDict, defaultdict
import einops
import pytorch_lightning

from yolo import yolo_loss
from yolo.non_max_suppression import non_max_suppression
from yolo.validation import match_nms_prediction, ap_per_class


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


class FeatureConcat(Module):
    storage: dict
    name: str
    clone: bool

    def __init__(self, storage: dict, name: str):
        super().__init__()
        self.storage = storage
        self.name = name

    def forward(self, x: torch.Tensor):
        return torch.cat((x, self.storage[self.name]), dim=1)


class Conv(Module):
    default_act = torch.nn.SiLU(True)  # default activation
    inner: Module

    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, act=True):
        super().__init__()
        p = self._auto_pad(kernel_size, dilation) if padding is None else padding
        self.conv = torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride, p, groups=groups, dilation=dilation,
                                    bias=False)
        self.norm = torch.nn.BatchNorm2d(out_channel)
        self.act = self.default_act if act else torch.nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    @staticmethod
    def _auto_pad(k: int | tuple[int, int], d=1):  # kernel, padding, dilation
        # Pad to 'same' shape outputs
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else (d * (k[0] - 1) + 1, d * (k[1] - 1) + 1)  # actual kernel-size
        p = k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2)  # auto-pad
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
        self.bottlenecks = torch.nn.Sequential(OrderedDict(
            {f"bottleneck{i}": Bottleneck(hidden_channel, hidden_channel, shortcut, g, e=1.0) for i in range(num)}
        )
        )
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
            SPPF(512, 512, 5),
            # head
            Conv(512, 256, 1, 1),  # 10
            FeatureExporter(self.feature_storage, "x10"),
            torch.nn.Upsample(None, 2, "nearest"),  # 11
            FeatureConcat(self.feature_storage, "b6"),  # 12
            C3(256 + 256, 256, 2, False),  # 13
            Conv(256, 128, 1, 1),  # 14
            FeatureExporter(self.feature_storage, "x14"),
            torch.nn.Upsample(None, 2, "nearest"),  # 15
            FeatureConcat(self.feature_storage, "b4"),  # 16
            C3(128 + 128, 128, 2, False),  # 17
            FeatureExporter(self.feature_storage, "x17"),
            Conv(128, 128, 3, 2),  # 18
            FeatureConcat(self.feature_storage, "x14"),  # 19
            C3(256, 256, 2, False),  # 20
            FeatureExporter(self.feature_storage, "x20"),
            Conv(256, 256, 3, 2),  # 21
            FeatureConcat(self.feature_storage, "x10"),  # 22
            C3(512, 512, 2, False)  # 23
        )

    def forward(self, x):
        x23 = self.inner(x)
        x17 = self.feature_storage["x17"]
        x20 = self.feature_storage["x20"]

        self.feature_storage.clear()
        return [x17, x20, x23]


class Detect(Module):
    # YOLOv5 Detect head for detection models
    anchors: torch.Tensor
    ch: list[int]

    def __init__(self, nc: int, anchors: list, ch: list):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.origin_bbox = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.ch = ch
        self.m = torch.nn.ModuleList(
            torch.nn.Conv2d(x, self.no * self.na, 1)
            for x in ch
        )  # output conv

        s = 256  # 2x min stride
        self.stride = torch.tensor([s / x for x in (32, 16, 8)])  # [8, 16, 32]
        self.anchors /= self.stride.view(-1, 1, 1)  # 将anchor大小映射到基于grid的比例

    def forward(self, x: list[torch.Tensor]):
        """
        Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`.
        网络输出x的最后一维的格式为：[x坐标, y坐标, 高度, 宽度, 置信度, 每个类别的logit]
        """
        result = []
        for i in range(self.nl):
            y = self.m[i](x[i])  # 检测头，就是1x1卷积，输出an * (nc + 5)的特征图
            # 然后，将an（每层的anchor数）和nc + 5（一个物体的特征数）单独切分到两个维度里, x(bs,255,20,20) to x(bs,3,20,20,85)
            y = einops.rearrange(y, "bs (na no) ny nx -> bs na ny nx no", na=self.na, no=self.no)
            result.append(y.contiguous())

        return result

    def forward_detector_id(self, x: torch.Tensor, detector_id: int):
        x = self.m[detector_id](x)
        x = einops.rearrange(x, "bs (na no) ny nx -> bs na ny nx no", na=self.na, no=self.no)
        x = x.contiguous()

        return x

    def inference_post_process(self, x):
        # 网络输出的x是基于anchor的偏移量，需要转换成基于整个图像的坐标
        z = []
        for i in range(self.nl):
            # x[i]的结构为x(bs,3,20,20,85)
            bs, _, ny, nx, _ = x[i].shape

            # 对于不同大小的图片，检测头的输出的特征图大小是不一样的，所以需要根据特征图的大小来生成网格
            # 如果一样大就用缓存的
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i], self.origin_bbox[i] = self._make_grid(nx, ny, i)

            # grid是网格中心点，anchor_grid是网格的大小
            # 这个sigmoid会对输入的所有数进行sigmoid，因为恰好结果中的每一项每个数都需要sigmoid
            x[i].sigmoid_()
            xy, hw, conf, logit = x[i].split((2, 2, 1, self.nc), 4)
            # xy * 2将结果映射到(0, 2)，再-1得到相对锚框中心点的偏移量(-1, 1)
            # 这个偏移量加上锚框中心点的位置gird得到bbox中心点在整个图像中的绝对坐标
            # 但这个坐标是基于该层的特征图的，每层的结果都是从特征图中通过1x1卷积提取出来，则每个锚框对应该层的特征图上的一个像素
            # 而特征图的尺寸都是原图像按比例缩小的，所以需要将坐标重新按比例放大
            # 于是最终乘以特征图缩放比例self.stride，得到相对原图像的坐标
            xy = (xy * 2 - 1 + self.grid[i]) * self.stride[i]  # xy
            # (wh * 2) ** 2 是yolo定义的从输出到 锚框长宽缩放比例 的映射，可以看出缩放可达最大4倍
            # anchor_grid是预先将预定义锚框大小self.anchors与缩放比例self.stride乘起来
            # 这样通过一次乘法就可以直接得到bbox相对原图像的大小
            hw = (hw * 2) ** 2 * self.anchor_grid[i]  # hw

            # 为了后续的OOD检测时从特征图中的正确的位置提取相关特征，添加每个输出的bbox的具体位置
            # origin_bbox = self.origin_bbox[i].expand(bs, -1, -1, -1, -1)
            img_size = torch.tensor((ny, nx, ny, nx), device=xy.device) * self.stride[i]
            origin_bbox = torch.cat((xy, hw), dim=4) / img_size
            # 为了后续知道此检测结果来自哪个layer，添加一个layer编号
            layer_id = torch.tensor(i, dtype=torch.float32, device=xy.device).expand(bs, self.na, ny, nx, 1)

            y = torch.cat((xy, hw, conf, origin_bbox, layer_id, logit), 4)

            # 将所有位置的所有锚框的结果都合并到一个维度
            y = y.view(bs, self.na * nx * ny, y.shape[-1])
            z.append(y)

        # 合并所有层的结果，因为是哪一层的结果对后面的处理也不重要
        return torch.cat(z, 1)

    def _make_grid(self, nx: int, ny: int, i: int):
        """
        :param nx: x方向的anchor数量
        :param ny: y方向的anchor数量
        :param i: 要生成的网络对应的anchor层
        :return: grid是网格的中心点位置，anchor_grid是网格的大小，grid_bbox是网格归一化后的包围盒
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        left_top = torch.stack((xv, yv), 2)

        # 生成的是锚框坐上角的坐标，加上0.5成中心点的坐标
        grid = (left_top + 0.5).expand(shape)
        bbox = torch.cat([left_top, left_top + 1], 2) / torch.tensor([nx, ny, nx, ny], device=d)
        bbox = bbox.expand(1, self.na, ny, nx, 4)
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid, bbox


_ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326]
]


class Yolo(pytorch_lightning.LightningModule):
    backbone: BackBone
    detect: Detect
    loss_func: yolo_loss.ComputeLoss

    def __init__(self, num_class: int):
        super(Yolo, self).__init__()
        self._val_stats = defaultdict(list)

        self.backbone = BackBone()
        self.detect = Detect(nc=num_class, anchors=_ANCHORS, ch=[128, 256, 512])
        self.loss_func = yolo_loss.ComputeLoss(num_class, _ANCHORS, self.detect.anchors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x17, x20, x23 = self.backbone(x)

        x = self.detect([x17, x20, x23])

        return x

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        return self.loss_func(self(x), y)

    @functools.cached_property
    def layer_order(self):
        name_order = []
        for name, _ in self.named_modules():
            name_order.append(name)
        return name_order

    @torch.inference_mode()
    def inference(self, x):
        raw_output = self(x)
        raw_result = self.detect.inference_post_process(raw_output)
        return non_max_suppression(raw_result, self.detect.nc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters()) # type: ignore

    def training_step(self, batch, batch_idx):
        img, target = batch
        img = torch.from_numpy(img).to(self.device, non_blocking=True).float() / 255
        target = torch.from_numpy(target).to(self.device, non_blocking=True)

        loss = self.loss(img, target)

        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_stats.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        img, target = batch
        img = torch.from_numpy(img).to(self.device, non_blocking=True).float() / 255

        prediction = self.inference(img)
        stats = match_nms_prediction(prediction, target, img.shape)
        self._val_stats[dataloader_idx].extend(stats)

    def on_validation_epoch_end(self):
        for dataloader_idx, stats in self._val_stats.items():
            stats = [numpy.concatenate(x, 0) for x in zip(*stats)]
            if not (len(stats) != 0 and stats[0].any()):
                print("不存在正样本，忽略")
                continue
            tp, fp, p, r, f1, ap, auroc, fpr95, threshold, conf_auroc, conf_fpr95, conf_thr = ap_per_class(stats)
            ap50, ap95 = ap[:, 0], ap[:, -1]  # AP@0.5, AP@0.5:0.95
            mr, map50, map95 = r.mean(), ap50.mean(), ap95.mean()

            summary = dict(map50=map50, map95=map95, recall=mr, conf_auroc=conf_auroc, conf_fpr95=conf_fpr95)
            summary = {f"{k}.{dataloader_idx}": v for k, v in summary.items()}

            self.log_dict(summary)


if __name__ == '__main__':
    network = Yolo(1)
    print(network)
