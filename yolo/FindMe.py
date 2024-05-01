import einops

from yolo.Network import *
import torch
from torch.nn import Module
import torchvision


class TargetEncoder(Module):

    def __init__(self, inner_size: tuple[int, int] = (32, 32)):
        super().__init__()

        self.inner_size = inner_size

        self.inner = torch.nn.Sequential(
            Conv(3, 32, 5, 2, 2),
            C3(32, 32, 1),
            Conv(32, 32, 3, 2, 1),
            C3(32, 32, 1),
            Conv(32, 32, 3, 2, 1),
            C3(32, 32, 1),
            Conv(32, 32, 1),
        )

    def feature_dim(self):
        return self.inner_size[0] * self.inner_size[1] // 2

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        # target: list of [batch, x, y]，no batch
        batch_size = target.shape[0]
        feature = self.inner(target).view(batch_size, -1)
        return feature


class Detect(Module):
    # YOLOv5 Detect head for detection models

    def __init__(self, anchors: list, ch: list, target_dim: int):  # detection layer
        super().__init__()
        self.no = 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

        self.hidden_dim = 512
        self.output_odd_feature = False

        self.k_mlp = torch.nn.ModuleList(torch.nn.Conv2d(x, self.na * self.hidden_dim, 1) for x in ch)  # output conv
        self.v_mlp = torch.nn.ModuleList(torch.nn.Conv2d(x, self.na * self.hidden_dim, 1) for x in ch)  # output conv
        self.q_mlp = torch.nn.Linear(target_dim, self.hidden_dim)
        self.output_mpl = torch.nn.Linear(self.hidden_dim, self.no)

        s = 256  # 2x min stride
        self.stride = torch.tensor([s / x for x in (32, 16, 8)])  # [8, 16, 32]
        self.anchors /= self.stride.view(-1, 1, 1)  # 将anchor大小映射到基于grid的比例

    def forward(self, x: list[torch.Tensor], target: torch.Tensor):
        """
        判断特征图上每个像素与每个目标的相似度
        :param x: backbone输出，包含一个batch内每张图像的三个特征图
        :param target: 搜索目标特征，所有搜索目标合并为batch
        :return: 三个特征图，包含每个像素上每个目标的锚框位置修正和置信度（相似度）
        """
        ood_score = []

        q = self.q_mlp(target)  # q = [bs, hidden_dim]

        for i in range(self.nl):
            k = self.k_mlp[i](x[i])
            v = self.v_mlp[i](x[i])

            bs, _, h, w = k.shape  # _ = self.na * self.hidden_dim
            k = einops.rearrange(k, "b (a d) h w -> (b a h w) d", a=self.na, d=self.hidden_dim)
            v = einops.rearrange(v, "b (a d) h w -> (b a h w) d", a=self.na, d=self.hidden_dim)
            score = torch.einsum("nd, bd ->nbd", k, q)
            r = torch.einsum("nbd, nd -> nbd", score, v)
            r = self.output_mpl(r)

            r = einops.rearrange(r, "(bs a h w) b no -> bs b a h w no", bs=bs, a=self.na, h=h, w=w)

            x[i] = r.contiguous()

        return x

    def loss(self, x: list[torch.Tensor], y: torch.Tensor):
        """
        计算loss
        :param x: forward的输出
        :param y: 真实结果，格式为
        :return: loss
        """

    def inference_post_process(self, x):
        # 网络输出的x是基于anchor的偏移量，需要转换成基于整个图像的坐标
        z = []
        for i in range(self.nl):
            # x[i]的结构为x(bs,3,20,20, 23, 4)
            bs, _, ny, nx, nt, _ = x[i].shape

            # 对于不同大小的图片，检测头的输出的特征图大小是不一样的，所以需要根据特征图的大小来生成网格
            # 如果一样大就用缓存的
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            # grid是网格中心点，anchor_grid是网格的大小
            # 这个sigmoid会对输入的所有数进行sigmoid，因为恰好结果中的每一项每个数都需要sigmoid
            x[i].sigmoid_()
            xy, wh, conf = x[i].split((2, 2, 1), -1)
            # xy * 2将结果映射到(0, 2)，再-1得到相对锚框中心点的偏移量(-1, 1)
            # 这个偏移量加上锚框中心点的位置gird得到bbox中心点在整个图像中的绝对坐标
            # 但这个坐标是基于该层的特征图的，每层的结果都是从特征图中通过1x1卷积提取出来，则每个锚框对应该层的特征图上的一个像素
            # 而特征图的尺寸都是原图像按比例缩小的，所以需要将坐标重新按比例放大
            # 于是最终乘以特征图缩放比例self.stride，得到相对原图像的坐标
            xy = (xy * 2 - 1 + self.grid[i]) * self.stride[i]  # xy
            # (wh * 2) ** 2 是yolo定义的从输出到 锚框长宽缩放比例 的映射，可以看出缩放可达最大4倍
            # anchor_grid是预先将预定义锚框大小self.anchors与缩放比例self.stride乘起来
            # 这样通过一次乘法就可以直接得到bbox相对原图像的大小
            wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh

            y = torch.cat((xy, wh, conf), -1)
            # 将所有位置的所有锚框的结果都合并到一个维度，因为是哪个锚框得出的结果对于后面的处理并不重要
            y = y.view(bs, self.na * nx * ny, nt, self.no)
            z.append(y)

        # 合并所有层的结果，因为是哪一层的结果对后面的处理也不重要
        return torch.cat(z, 1)

    def _make_grid(self, nx: int, ny: int, i: int):
        """
        :param nx: x方向的anchor数量
        :param ny: y方向的anchor数量
        :param i: 要生成的网络对应的anchor层
        :return: grid是网格的中心点位置，anchor_grid是网格的大小
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        # 生成的是锚框坐上角的坐标，加上0.5成中心点的坐标
        grid = torch.stack((xv, yv), 2).expand(shape) + 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


_ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326]
]


class FindMe(Module):
    feature_storage: dict

    def __init__(self):
        super(FindMe, self).__init__()
        self.feature_storage = {}
        self.backbone = BackBone()

        self.image_encoder = TargetEncoder()
        feature_dim = self.image_encoder.feature_dim()

        self.detect = Detect(anchors=_ANCHORS, ch=[128, 256, 512], target_dim=feature_dim)

    def forward(self, source: torch.Tensor, target: list[torch.Tensor]):
        x17, x20, x23 = self.backbone(source)
        target = self.image_encoder(target)

        x = self.detect.forward([x17, x20, x23], target)

        return x


def main():
    # encoder = TargetEncoder()
    # x = encoder([torch.randn([3, 320, 180]), torch.randn([3, 4, 32])])
    # print(x.shape)
    source = torch.randn(1, 3, 640, 640)
    target = [torch.randn([3, 320, 180]), torch.randn([3, 4, 32])]

    findme = FindMe()
    output = findme.forward(source, target)
    for o in output:
        print(o.shape)


if __name__ == '__main__':
    main()
