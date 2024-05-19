import torch
from torch.nn import Module


def noise(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    return (torch.sign(torch.rand_like(x) - 0.5) * epsilon + x).clip(0, 1)


def fsgm_attack(model: Module, x: torch.Tensor, num_classes: int, label: torch.Tensor,
                epsilon: float = 0.01) -> torch.Tensor:
    y = torch.zeros(num_classes, device=x.device, dtype=torch.float32, requires_grad=False)
    y[label] = 1
    attack = x.clone().detach().requires_grad_(True)

    output = model(attack)[0]  # 输入模型进行推理
    loss = torch.nn.functional.cross_entropy(output, y)  # 计算损失函数
    loss.backward()  # 反向传播，得到输入的attack的梯度，记录在attack.grad中

    gradient_sign = torch.sign(attack.grad) * epsilon  # 计算梯度的符号，然后乘以epsilon得到干扰
    return (gradient_sign + x).clip(0, 1)


def pdg_attack(model: Module, x: torch.Tensor, num_classes: int, label: torch.Tensor, epsilon: float, epoch: int = 20):
    y = torch.zeros(num_classes, device=x.device, dtype=torch.float32, requires_grad=False)
    y[label] = 1
    attack = x.clone()  # + torch.zeros_like(x).uniform_(-epsilon, epsilon)  # 初始化时添加随机噪声，效果更好

    alpha = epsilon / epoch * 3
    for i in range(epoch):
        attack.detach_().requires_grad_()  # 与之前的计算过程分离，为计算梯度做准备

        output = model(attack)[0]  # 输入模型进行推理
        if torch.argmax(output) != label:
            break
        loss = torch.nn.functional.cross_entropy(output, y)  # 计算损失函数
        loss.backward()  # 反向传播，得到输入的attack的梯度，记录在attack.grad中

        with torch.no_grad():
            attack += alpha * torch.sign(attack.grad)  # 和FSGM的过程相同
            attack = (attack - x).clip(-epsilon, epsilon) + x  # 限制变化范围(-epsilon, epsilon)
            attack.clip_(0, 1)

    return attack


def deepfool_attack_l2(model: Module, x: torch.Tensor, chosen_class: int = 10, max_epoch: int = 5):
    chosen_class += 1
    with torch.no_grad():
        output = model(x)[0]
        _, chosen_index = output.topk(chosen_class, largest=True)
        correct_y = chosen_index[0]  # 最大的这个是正确分类

    attack = x.clone()

    loop_time = 0
    while loop_time < max_epoch:
        attack.detach_().requires_grad_()  # 与之前的计算过程分离，为计算梯度做准备

        output = model(attack)[0]  # 输入模型进行推理
        if torch.argmax(output) != correct_y:
            break  # 攻击已达成

        output = output[chosen_index]

        # 计算chosen_class种分类的概率输出相对于图像的梯度（巨慢）
        grads = torch.empty([chosen_class, *attack.shape], device=attack.device, dtype=attack.dtype)
        for i in range(0, chosen_class):
            grads[i] = torch.autograd.grad(output[i], attack, retain_graph=True)[0]  # 反向传播

        # 计算对抗样本
        with torch.no_grad():
            grads -= grads[0].clone()
            output = torch.abs_(output - output[0])

            # 计算所有梯度之差的二范数
            grad_norms = torch.norm(grads.view(chosen_class, -1), dim=1)

            l = torch.argmin(output[1:] / grad_norms[1:]) + 1  # 所有类中找出其中除了correct_y以外距离最近的一个

            attack += (output[l] / grad_norms[l] ** 2 * 1.002) * grads[l]  # 使结果接近l
            attack.clip_(0, 1)

        loop_time += 1
        # print("loop: ", loop_time)

    return attack


def deepfool_attack_linf(model: Module, x: torch.Tensor, chosen_class: int = 10, max_epoch: int = 5):
    chosen_class += 1
    with torch.no_grad():
        output = model(x)[0]
        _, chosen_index = output.topk(chosen_class, largest=True)
        correct_y = chosen_index[0]  # 最大的这个是正确分类

    attack = x.clone()

    loop_time = 0
    while loop_time < max_epoch:
        attack.detach_().requires_grad_()  # 与之前的计算过程分离，为计算梯度做准备

        output = model(attack)[0]  # 输入模型进行推理
        if torch.argmax(output) != correct_y:
            break  # 攻击已达成

        output = output[chosen_index]

        # 计算chosen_class种分类的概率输出相对于图像的梯度（巨慢）
        grads = torch.empty([chosen_class, *attack.shape], device=attack.device, dtype=attack.dtype)
        for i in range(0, chosen_class):
            grads[i] = torch.autograd.grad(output[i], attack, retain_graph=True)[0]  # 反向传播

        # 计算对抗样本
        with torch.no_grad():
            grads -= grads[0].clone()
            output = torch.abs_(output - output[0])

            # 计算所有梯度之差的无穷范数
            grad_norms = torch.norm(grads.view(chosen_class, -1), 1, dim=1)

            l = torch.argmin(output[1:] / grad_norms[1:]) + 1  # 所有类中找出其中除了correct_y以外距离最近的一个

            attack += (output[l] / grad_norms[l] * 1.002) * torch.sign(grads[l])  # 使结果接近l
            attack.clip_(0, 1)

        loop_time += 1

    return attack


def bfgs_attack(model: Module, x: torch.Tensor, c: torch.Tensor, epoch: int):
    with torch.no_grad():
        output = model(x)[0]
        _, index = output.topk(2, largest=True)

        correct_y = index[0]  # 最大的这个是正确分类
        chosen_index = index[1]  # 取次大的分类作为攻击的预选类
        y_onehot = torch.zeros_like(output, requires_grad=False)
        y_onehot[chosen_index] = 1

    attack = x.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([attack], lr=1)  # 拟牛顿法优化器

    end = False  # 用于提前终止

    def eval_model():
        opt.zero_grad()
        output = model(attack)[0]  # 输入模型进行推理
        if torch.argmax(output) != correct_y:
            nonlocal end
            end = True
        loss_f = torch.nn.functional.cross_entropy(output, y_onehot)  # 计算损失函数
        loss = torch.norm(attack - x) * c + loss_f
        loss.backward()
        return loss

    loop_time = 0
    for i in range(epoch):
        opt.step(eval_model)

        loop_time += 1
        # print("loop time = ", loop_time)
        if end:
            break

        with torch.no_grad():
            attack.clip_(0, 1)  # 盒约束

    return attack


def cw_attack(model: Module, adv_examples: torch.Tensor, adv_target_onehot: torch.Tensor, iteration: int = 5000,
              lr: float = 0.01, c: float = 1):
    box_max = 1
    box_min = 0
    box_mul = (box_max - box_min) / 2
    box_plus = (box_min + box_max) / 2

    modifier = torch.zeros_like(adv_examples)

    for i in range(iteration):
        modifier.requires_grad_()
        new_example = torch.tanh(adv_examples + modifier) * box_mul + box_plus
        l2dist = torch.sum(torch.square(new_example - adv_examples))  # L2 作为D
        output = model(new_example)
        # 设定攻击目标
        others = torch.max((1 - adv_target_onehot) * output, dim=1).values
        real = torch.sum(output * adv_target_onehot, dim=1)
        loss2 = torch.sum(torch.maximum(torch.zeros_like(others) - 0.01, others - real))
        loss = l2dist + c * loss2

        if modifier.grad is not None:
            modifier.grad.zero_()
        loss.backward()

        modifier = (modifier - modifier.grad * lr).detach()

    new_img = torch.tanh(adv_examples + modifier) * box_mul + box_plus
    return new_img
