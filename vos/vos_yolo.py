from collections import defaultdict
from typing import Any

import einops
import numpy
import rich
from typing_extensions import Self

import pytorch_lightning
import torch
from torch.utils.data import DataLoader

from yolo.network import Yolo
from yolo.non_max_suppression import non_max_suppression
from yolo.validation import ap_per_class, match_nms_prediction


def _weighted_log_sum_exp(value, weight, dim: int) -> torch.Tensor:
    # from https://github.com/deeplearning-wisc/vos
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    m, _ = torch.max(value, dim=dim)
    sum_exp = torch.sum(weight * torch.exp(value - m))
    return m + torch.log(sum_exp)


def _gen_ood_samples(roi_features: list[torch.Tensor], select: int = 1, sample_from: int = 10000):
    # from https://github.com/deeplearning-wisc/vos
    # the covariance finder needs the data to be centered.

    X = []  # 中心归一化后的特征
    mean_embed_id = []  # 均值
    for feature in roi_features:
        mean = feature.mean(0)
        X.append(feature - mean)
        mean_embed_id.append(mean)

    X = torch.cat(X)

    # 协方差矩阵
    eye_matrix = 0.0001 * torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
    temp_precision = torch.mm(X.t(), X) / len(X) + eye_matrix

    ood_samples = []
    for mean in mean_embed_id:
        # 使用均值和协方差矩阵构造多变量正态分布
        new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
            mean, covariance_matrix=temp_precision
        )
        negative_samples = new_dis.rsample(torch.Size((sample_from,)))  # 重参数化技巧采样得到负样本，保证此过程可微，这里其实和正常的采样没有区别
        prob_density = new_dis.log_prob(negative_samples)  # 计算对数概率密度

        # 保留概率密度最低的self.select个负样本作为OOD样本
        _, prob_idx = torch.topk(prob_density, select, largest=False, sorted=False)
        ood_samples.append(negative_samples[prob_idx])

    ood_samples = torch.cat(ood_samples)

    return ood_samples

class VOSDetect(torch.nn.Module):
    """
    vos使用分类logit作为输入，这里使用ROI作为输入，in_dim为检测头输入特征的channel数
    因为不是输入不是分类logit，所以使用能量函数用不了。

    和Yolov5的检测头处于同一位置，直接判断是否是分布外样本。其实就是不想改动Yolov5的网络结构。
    """

    def __init__(self, num_anchors: int, ch: list):
        super().__init__()
        self.num_anchors = num_anchors

        # 和原本的检测头起类似作用，用于判断是否为OOD样本
        self.ood_detector = torch.nn.ModuleList(
            torch.nn.Conv2d(c, 1, kernel_size=1) for c in ch
        )

    def forward(self, x: list[torch.Tensor]):
        """
        :param x: 检测头输入的像素（roi），来自三个层
        :return: OOD score
        """
        # squeeze(-3) 用于移除channel layer
        return [torch.sigmoid_(detect(x_i).squeeze_(-3)) for x_i, detect in zip(x, self.ood_detector)]

    def inference_post_process(self, x: list[torch.Tensor]):
        z = []
        for x_i in x:
            # x[i]的结构为x(bs, 20,20)
            y = einops.repeat(x_i, 'bs ny nx -> bs (c ny nx) l', c=self.num_anchors, l=1)
            z.append(y)
        return torch.cat(z, 1)

    def loss(self, x: list[torch.Tensor], y: list[torch.Tensor]):
        loss = sum(torch.nn.functional.mse_loss(x_i, y_i) for x_i, y_i in zip(self(x), y))
        return loss


class _FeatureCache:
    """
    缓存vos需要的特征，每一个检测头，每一个类型都需要有独立特征表
    """
    sample_number: int
    num_class: int

    def __init__(self, detector_num: int, num_class: int, sample_number: int = 300):
        self.sample_number = sample_number
        self.num_class = num_class
        list_len = num_class * detector_num

        self._features = [torch.empty(0) for _ in range(list_len)]
        self._features_offset = [0 for _ in range(list_len)]
        self._is_enough = [False for _ in range(list_len)]

    def _get_index(self, detector_id: int, type_id: int):
        return detector_id * self.num_class + type_id

    def all_enough(self):
        return all(self._is_enough)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        for f in self._features:
            f.to(*args, **kwargs)

        return self

    def is_enough(self, detector_id: int, type_id: int):
        return self._is_enough[self._get_index(detector_id, type_id)]

    def put_feature(self, detector_id: int, type_id: int, feature: torch.Tensor):
        assert len(feature.shape) <= 2
        if len(feature.shape) == 1:
            feature = feature.unsqueeze(0)

        feature_count = feature.shape[0]
        pos_idx = self._get_index(detector_id, type_id)

        if self._features[pos_idx].numel() == 0:  # 还没有开辟空间，就先开空间
            self._features[pos_idx] = torch.empty((self.sample_number, feature.shape[-1]), device=feature.device)

        assert self._features[pos_idx].shape[-1] == feature.shape[-1]

        cur_offset = self._features_offset[pos_idx]  # 最后一个可以容纳的位置
        remaining = self.sample_number - cur_offset  # 直到末尾余下的空间
        if feature_count > remaining:
            if feature_count >= self.sample_number:
                self._features[pos_idx] = feature[0: self.sample_number].detach()
                self._features_offset[pos_idx] = 0
            else:
                self._features[pos_idx][cur_offset:] = feature[:remaining].detach()
                self._features[pos_idx][:feature_count - remaining] = feature[remaining:].detach()
                self._features_offset[pos_idx] = (self._features_offset[pos_idx] + feature_count) % self.sample_number

            self._is_enough[pos_idx] = True
        else:
            self._features[pos_idx][cur_offset:cur_offset + feature_count] = feature.detach()
            self._features_offset[pos_idx] += feature_count

    def get_feature(self, detector_id: int, type_id: int):
        return self._features[self._get_index(detector_id, type_id)]

    def feature_each_detector(self):
        num_detector = len(self._features) // self.num_class
        return [[self.get_feature(d, i) for i in range(self.num_class)] for d in range(num_detector)]


class VosYolo(pytorch_lightning.LightningModule):
    """
    简单包住yolo，添加vos的几个网络层和训练方法，
    vos在原论文中使用时使用输出类别的logit计算ood-score，但是我们没法这么做，尝试使用置信度，
    vos原论文在训练一定轮数后加入vos_loss，要求使用一个特殊神经网络区分检测头输入的ROI是神经网络产生的还是使用正态分布生成
    此网络层产生的loss加入原神经网络的loss中一同训练，以使检测头输入的ROI的分布缩减

    从一同训练改为微调，开始时收集足够数量的特征，随训练更新
    将检测头输入层中每一个像素当成一个ROI，与GT进行匹配（无论预测结果如何）保留包含正样本的收集起来

    每一轮计算收集的样本的

    """
    yolo: Yolo
    vos_detect: VOSDetect

    def __init__(self, num_class: int):
        super().__init__()
        self.loss_weight = 0.1

        self.yolo = Yolo(num_class)
        self._feature_cache = _FeatureCache(self.yolo.detect.nl, num_class)
        self.vos_detect = VOSDetect(self.yolo.detect.na, self.yolo.detect.ch)

        # for val
        self._val_stats = defaultdict(list)
        self._val_iouv = numpy.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self._val_niou = self._val_iouv.size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.yolo.forward(x)

    def configure_optimizers(self):
        return self.yolo.configure_optimizers()

    def training_step(self, batch, batch_idx):
        # yolo train
        img, target = batch
        img = torch.from_numpy(img).to(self.device, non_blocking=True).float() / 255
        target = torch.from_numpy(target).to(self.device, non_blocking=True)

        backbone_output = self.yolo.backbone(img)  # 只需要yolo backbone的输出
        feature_size = tuple((x.shape[-2], x.shape[-1]) for x in backbone_output)
        indexed_target = self.yolo.loss_func.build_targets(feature_size, target)

        detector_output = self.yolo.detect(backbone_output)

        yolo_loss = self.yolo.loss_func(detector_output, target)

        # vos 
        X = []
        Y = []
        for detector_id, index, new_feature in zip(range(self.yolo.detect.nl), indexed_target, backbone_output):
            b, a, gj, gi, _, _, cls = index
            pos_roi: torch.Tensor = new_feature[b, :, gj, gi]

            # 收集特征
            for type_id in range(self._feature_cache.num_class):
                f = pos_roi[torch.logical_and(a == detector_id, cls == type_id)]
                self._feature_cache.put_feature(detector_id, type_id, f)

            collect_feature = [self._feature_cache.get_feature(detector_id, type_id) for type_id in
                               range(self.yolo.detect.nc)]
            neg_roi = _gen_ood_samples(collect_feature)

            x = torch.cat((pos_roi, neg_roi)).unsqueeze_(-1).unsqueeze_(-1)
            label = torch.zeros((pos_roi.shape[0] + neg_roi.shape[0], 1, 1), device=pos_roi.device, dtype=torch.long)
            label[:pos_roi.shape[0]] = 1

            X.append(x)
            Y.append(label)
        pass

        vos_loss = self.vos_detect.loss(X, Y)

        loss = yolo_loss + vos_loss * self.loss_weight

        self.log("yolo_loss", yolo_loss)
        self.log("vos_loss", vos_loss)

        self.log("train_loss", loss)

        return yolo_loss

    def to(self, *args: Any, **kwargs: Any) -> Self:
        self._feature_cache.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def on_train_start(self) -> None:
        self.collect_features(self.trainer.train_dataloader) # type: ignore

    def on_validation_epoch_start(self) -> None:
        self._val_stats.clear()

    def on_validation_epoch_end(self) -> None:
        for dataloader_idx, stats in self._val_stats.items():
            stats = [numpy.concatenate(x, 0) for x in zip(*stats)]
            tp, fp, p, r, f1, ap, auroc, fpr95, threshold, conf_auroc, conf_fpr95, conf_thr = ap_per_class(stats)
            ap50, ap95 = ap[:, 0], ap[:, -1]  # AP@0.5, AP@0.5:0.95
            mr, map50, map95 = r.mean(), ap50.mean(), ap95.mean()

            summary = dict(map50=map50, map95=map95, recall=mr, auroc=auroc, fpr95=fpr95, conf_auroc=conf_auroc,
                           conf_fpr95=conf_fpr95)
            summary = {f"{k}.{dataloader_idx}": v for k, v in summary.items()}

            self.log_dict(summary)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        img, target = batch

        img = torch.from_numpy(img).to(self.device, non_blocking=True).float() / 255

        backbone_output = self.yolo.backbone(img)
        bbox_conf_logit = self.yolo.detect.inference_post_process(self.yolo.detect(backbone_output))
        ood_scores = self.vos_detect.inference_post_process(self.vos_detect(backbone_output))
        prediction = non_max_suppression(torch.cat((bbox_conf_logit, ood_scores), dim=-1), self.yolo.detect.nc, 1)

        stats = match_nms_prediction(prediction, target, img.shape, ood_score_pos=-1)
        self._val_stats[dataloader_idx].extend(stats)


        pass

    def collect_features(self, train_dataloader: DataLoader):
        """
        在进行vos微调之前需要先收集足够多的特征
        :return:
        """
        assert not self._feature_cache.all_enough(), "不要重复收集"
        self.eval()
        rich.print("开始预收集特征")
        e = 0
        for e, (img, target) in enumerate(train_dataloader):
            img = torch.from_numpy(img).to(self.device, non_blocking=True).float() / 255
            target = torch.from_numpy(target).to(self.device, non_blocking=True)

            backbone_output = self.yolo.backbone(img)  # 只需要yolo backbone的输出
            feature_size = tuple((x.shape[-2], x.shape[-1]) for x in backbone_output)
            indexed_target = self.yolo.loss_func.build_targets(feature_size, target)

            for detector_id, index, feature in zip(range(self.yolo.detect.nl), indexed_target, backbone_output):
                b, a, gj, gi, _, _, cls = index

                roi: torch.Tensor = feature[b, :, gj, gi]
                for type_id in range(self._feature_cache.num_class):
                    if self._feature_cache.is_enough(detector_id, type_id):
                        continue
                    f = roi[torch.logical_and(a == detector_id, cls == type_id)]
                    self._feature_cache.put_feature(detector_id, type_id, f)

            if self._feature_cache.all_enough():
                break

        assert self._feature_cache.all_enough()
        rich.print(f"预收集特征结束，使用{e}/{len(train_dataloader)}的数据")

        self.train()
