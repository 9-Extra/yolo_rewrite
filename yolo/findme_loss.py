# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Loss functions."""
import math

import torch
import torch.nn as nn


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates IoU, GIoU, DIoU, or CIoU between two boxes, supporting xywh/xyxy formats.

    Input shapes are box1(1,4) to box2(n,4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)  # 就是sigmoid加上交叉熵

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # sigmoid计算概率
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)  # 交叉熵
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = 1, 0  # positive, negative BCE targets

        # Focal loss
        g = 0.0  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = model.detect  # Detect() module
        self.balance = [4.0, 1.0, 0.4]  # 为不同大小的锚框赋予不同权重，较小的要求更加精确
        self.ssi = 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr = BCEcls, BCEobj, 1.0
        self.na = m.na  # number of anchors
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, predictions, targets_list):  # predictions, targets
        cls_loss = torch.zeros(1, device=self.device)  # class loss
        box_loss = torch.zeros(1, device=self.device)  # box loss
        conf_loss = torch.zeros(1, device=self.device)  # object loss

        for i, targets in enumerate(targets_list):
            tbox_list, indices_list, anchors_list = self.build_targets(predictions, targets)  # targets
            # 计算每一层的loss
            zipped_iter = zip(predictions, indices_list, tbox_list, anchors_list, self.balance)
            # Losses
            for pi, indices, tbox, anchors, balance_weight in zipped_iter:  # layer index, layer predictions
                # pi的形状为[batch, anchor, gridy, gridx, 5+num_classes]

                b, a, gj, gi = indices  # image, anchor, gridy, gridx
                # b 包含此target的图片索引
                # a 包含此target的anchor索引，包括所有在进行偏移和缩放后可能框住目标的anchor
                # gi 包含此target的锚框x方向的索引
                # gj 包含此target的锚框y方向的索引

                # 初始化目标置信度，如果没有目标，所有位置的锚框置信度都为0
                target_conf = torch.zeros(pi.shape[:5], dtype=pi.dtype, device=self.device)

                if b.shape[0] != 0:  # 如果有目标
                    pxy, pwh, _ = pi[b, i, a, gj, gi].tensor_split((2, 4), dim=1)  # faster, requires torch 1.8.0
                    # 从预测的值中取出对应锚框的预测结果，并且将其分割为xy、wh、置信度、类别

                    # 回归，从模型输出值计算实际坐标
                    # sigmoid会将值映射到0-1区间，然后本来应该减1，但是
                    pxy = pxy.sigmoid() * 2 - 0.5
                    pwh = (pwh.sigmoid() * 2) ** 2 * anchors
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    iou = bbox_iou(pbox, tbox, CIoU=True).squeeze()  # iou(prediction, target)
                    box_loss += (1.0 - iou).mean()  # iou loss

                    iou = iou.detach().clamp(0)
                    # 在此填写置信度的拟合目标
                    # 绝大多数的值都为0，只有索引指定的部分锚框的置信度才不是0
                    target_conf[b, i, a, gj, gi] = iou  # iou ratio

                # 计算置信度估计的loss，置信度需要拟合预测出的bbox与真实bbox的iou
                # 如果没有目标，则不产生iou loss和分类loss，但是需要计算置信度的loss，模型必须正确判断为没有目标
                conf = pi[..., 4]
                conf_loss += self.BCEobj(conf, target_conf) * balance_weight  # 对于不同层的不同大小锚框，加上不同权重来平衡

        box_loss *= 0.05
        conf_loss *= 1.0
        cls_loss *= 0.5
        bs = predictions[0].shape[0]  # batch size

        return (box_loss + conf_loss + cls_loss) * bs

    def build_targets(self, prediction, targets):
        """
        Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain
        # 生成与targets形状一致的anchor_index
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # 将targets复制na次，再与anchor_index合并，这样每个anchor都有一个targets，并且可以通过anchor_index判断是哪一个anchor
        # targets最后的格式为target: (anchor数，目标数，7)，最后一维格式为(image ,class, x, y, w, h, anchor_index)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), -1)  # append anchor indices

        g = 0.5  # bias
        off = (torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device, ).float() * g)  # offsets

        for i in range(self.nl):
            # 遍历每一层，获取该层的anchors和输出大小（也是特征图大小和输出bbox数）
            anchors, shape = self.anchors[i], prediction[i].shape
            grid_w, gird_h = shape[4], shape[3]
            # gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            gain[1], gain[2], gain[3], gain[4] = grid_w, gird_h, grid_w, gird_h

            # Match targets to anchors
            t = targets * gain  # 映射到相对特征图的坐标，格式为(image, class, x, y, w, h, anchor_index)
            if nt:
                # Matches
                wh = t[..., 3:5]
                r = wh / anchors[:, None]  # wh 的目标缩放比例
                filter_mash = torch.max(r, 1 / r).max(2)[0] < 4.0  # 只有缩放比小于4的目标才需要被拟合，其它的过滤掉
                # filter_mash = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[filter_mash]  # filter

                # Offsets
                gxy = t[:, 1:3]  # 目标在特征图中的坐标
                gxi = gain[[1, 2]] - gxy  # 映射到相对于锚框的坐标（偏移量）
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]  # 过滤
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.split((1, 2, 2, 1), 1)  # (image, class), grid xy, grid wh, anchors
            a, b = a.long().view(-1), bc.long().view(-1)  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, gird_h - 1), gi.clamp_(0, grid_w - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors

        return tbox, indices, anch
