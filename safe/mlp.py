import torch
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from torch.utils.data import DataLoader


class MLP(torch.nn.Module):
    in_dim: int
    layer_name_list: list[str]

    def __init__(self, in_dim, feature_name_list: list[str]):
        super().__init__()
        self.in_dim = in_dim
        self.layer_name_list = feature_name_list
        assert len(set(feature_name_list)) == len(feature_name_list), "feature_name_list不应该有重复"

        self.inner = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim // 2),
            # torch.nn.BatchNorm1d(in_dim // 2),
            torch.nn.ReLU(True),

            torch.nn.Linear(in_dim // 2, in_dim // 4),
            # torch.nn.BatchNorm1d(in_dim // 4),
            torch.nn.ReLU(True),

            torch.nn.Linear(in_dim // 4, 1),
            torch.nn.Flatten(0),
        )

    def forward(self, x):
        return torch.sigmoid_(self.inner(x))

    def to_static_dict(self):
        return {
            "in_dim": self.in_dim,
            "feature_name_list": self.layer_name_list,
            "weight": self.state_dict()
        }

    @staticmethod
    def from_static_dict(mlp_dict: dict):
        mlp = MLP(mlp_dict["in_dim"], mlp_dict["feature_name_list"])
        mlp.load_state_dict(mlp_dict["weight"])

        return mlp


def train_mlp(mlp: MLP, train_dataset, batch_size: int, epoch: int, device: torch.device):
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)

    mlp = mlp.to(device).train()
    opt = torch.optim.Adam(mlp.parameters(), lr=0.0001)
    loss_func = torch.nn.BCEWithLogitsLoss()

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[acc]:%}"),
    ) as progress:
        train_task = progress.add_task("train", acc=0)
        for _ in progress.track(range(epoch), task_id=train_task, description="训练MLP"):
            for x, y in train_dataloader:
                x = x.to(device)
                y = y.to(device)

                opt.zero_grad()
                output = mlp(x)
                loss = loss_func(output, y)
                loss.backward()
                opt.step()

                # acc
                p = torch.zeros_like(output)
                p[output > 0.5] = 1
                val_acc = torch.count_nonzero(p == y).item() / y.shape[0]

        mlp.eval()
        val_acc = torch.tensor(0, device=device)
        count = 0
        with torch.no_grad():
            for x, y in train_dataloader:
                output = mlp.forward(x)
                p = torch.zeros_like(output)
                p[output > 0.5] = 1
                val_acc += torch.count_nonzero(p == y)
                count += y.shape[0]

            val_acc = val_acc.item() / count
        return val_acc
