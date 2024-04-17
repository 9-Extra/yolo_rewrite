import os
import yolo
import torch
from torch.utils.data import DataLoader
from dataset.h5Dataset import H5Dataset
import tqdm


def train(network: torch.nn.Module, train_loader: DataLoader, epochs: int, save_path: str, save_interval: int):
    os.makedirs(save_path, exist_ok=True)
    device = next(network.parameters()).device
    loss_func = yolo.loss.ComputeLoss(network)
    opt = torch.optim.Adam(network.parameters())

    network.train()
    for epoch in range(epochs):
        for i, (img, target) in enumerate(tqdm.tqdm(train_loader, desc=f"epoch: {epoch} / {epochs}")):
            img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
            target = torch.from_numpy(target).to(device, non_blocking=True)

            opt.zero_grad()
            output = network(img)
            loss = loss_func(output, target)
            loss.backward()
            opt.step()

        if epoch % save_interval == 0:
            torch.save(network.state_dict(), os.path.join(save_path, f"yolo_{epoch}.pth"))
            print(f"Loss: {loss.item()}")

    torch.save(network.state_dict(), os.path.join(save_path, f"yolo.pth"))


def main(dataset_dir, start_weight_dir=None):
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    network = yolo.Network.NetWork(2)
    if start_weight_dir:
        network.load_state_dict(torch.load(start_weight_dir))
    network.train().to(device, non_blocking=True)

    dataset = H5Dataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True,
                            collate_fn=H5Dataset.collate_fn)

    train(network, dataloader, 10, "weight", 1)


pass

if __name__ == '__main__':
    main("preprocess/pure_drone_train")
