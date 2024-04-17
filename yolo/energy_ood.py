import torch
import numpy
from torch.nn.functional import relu


def energy_func(output: torch.Tensor, temperature: float):
    return -temperature * torch.logsumexp(output / temperature, dim=1)


def get_ood_scores(model: torch.nn.Module, loader, temperature: float):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for data, target in loader:
            data = data.cuda()

            output = model.forward(data)
            smax = (torch.softmax(output, dim=1)).numpy(force=True)

            score = energy_func(output, temperature)

            _score.append(score.numpy(force=True))

            preds = numpy.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets  # 预测正确的
            wrong_indices = numpy.invert(right_indices)  # 预测错误的

            _right_score.append(-numpy.max(smax[right_indices], axis=1))
            _wrong_score.append(-numpy.max(smax[wrong_indices], axis=1))

    return numpy.concatenate(_score), numpy.concatenate(_right_score), numpy.concatenate(_wrong_score)


pass


class EnergyLoss:
    def __init__(self, m_in: float = -25, m_out: float = -7):
        self.m_in = m_in
        self.m_out = m_out

    def __call__(self, in_set_logit: torch.Tensor, out_set_logit: torch.Tensor):
        ec_out = -torch.logsumexp(out_set_logit, dim=1)
        ec_in = -torch.logsumexp(in_set_logit, dim=1)
        loss = (torch.pow(relu(ec_in - self.m_in), 2).mean() + torch.pow(relu(self.m_out - ec_out), 2).mean())

        return loss
