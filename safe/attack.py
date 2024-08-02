import torch

import yolo
import functools


class Attacker:

    @functools.cached_property
    def name(self):
        raise NotImplementedError

    def __call__(self, model: yolo.Network.Yolo, x: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError


class FSGMAttack(Attacker):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    @functools.cached_property
    def name(self):
        return f"fsgm_epsilon{self.epsilon}"

    @torch.enable_grad()
    def __call__(self, model: yolo.Network.Yolo, x: torch.Tensor, y: torch.Tensor):
        x.detach_().requires_grad_()  # 与之前的计算过程分离，为计算梯度做准备

        loss = model.loss(x, y)  # 计算损失
        grad = torch.autograd.grad(loss, x)[0]

        with torch.no_grad():
            attack = x + self.epsilon * torch.sign(grad)
            attack.clip_(0, 1)

        return attack


class PDGAttack(Attacker):
    def __init__(self, epsilon: float, epoch: int):
        self.epsilon = epsilon
        self.epoch = epoch

    @functools.cached_property
    def name(self):
        return f"pdg_epsilon{self.epsilon}_limit{self.epoch}"

    @torch.enable_grad()
    def __call__(self, model: yolo.Network.Yolo, x: torch.Tensor, y: torch.Tensor):
        attack = x.clone()  # + torch.zeros_like(x).uniform_(-epsilon, epsilon)  # 初始化时添加随机噪声，效果更好

        alpha = self.epsilon / self.epoch * 3
        for i in range(self.epoch):
            attack.detach_().requires_grad_()  # 与之前的计算过程分离，为计算梯度做准备

            loss = model.loss(attack, y)  # 计算损失
            grad = torch.autograd.grad(loss, attack)[0]

            with torch.no_grad():
                attack += alpha * torch.sign(grad)  # 和FSGM的过程相同
                attack = (attack - x).clip(-self.epsilon, self.epsilon) + x  # 限制变化范围(-epsilon, epsilon)
                attack.clip_(0, 1)

        return attack.detach_()
