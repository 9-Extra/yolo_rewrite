import torch
import numpy
from torch.nn.functional import relu

"""
见过的背景
见过的无人机
没见过的背景
没见过的无人机（伪装的无人机）
类似无人机但不是（鸟）
"""


def energy_func(output: torch.Tensor, temperature: float):
    return -temperature * torch.logsumexp(output / temperature, dim=1)


class EnergyLoss:
    def __init__(self, temperature: float, m_in: float = -25, m_out: float = -7):
        self.temperature = temperature
        self.m_in = m_in
        self.m_out = m_out

    def __call__(self, in_set_logit: torch.Tensor, out_set_logit: torch.Tensor):
        ec_out = -torch.logsumexp(out_set_logit, dim=1)
        ec_in = -torch.logsumexp(in_set_logit, dim=1)
        loss = (torch.square_(torch.relu_(ec_in - self.m_in)).mean() + torch.square_(
            torch.relu_(self.m_out - ec_out)).mean())

        return loss
