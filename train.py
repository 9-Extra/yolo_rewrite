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
            # ori_img = cv2.cvtColor(img[0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            # objs = target[target[:, 0] == 0][:, 1:]
            # h, w, _ = ori_img.shape
            # for obj in objs:
            #     cls, x, y, width, height = obj
            #     cls = int(cls)
            #     x1 = int((x - width / 2) * w)
            #     y1 = int((y - height / 2) * h)
            #     x2 = int((x + width / 2) * w)
            #     y2 = int((y + height / 2) * h)
            #     ori_img = cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     ori_img = cv2.putText(ori_img, str(train_loader.dataset.obj_record.label_names[cls]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.imshow('image', ori_img)
            # cv2.waitKey()

            img = torch.from_numpy(img).to(device, non_blocking=True).float() / 255
            target = torch.from_numpy(target).to(device)

            opt.zero_grad()
            output = network(img)
            loss = loss_func(output, target)
            loss.backward()
            opt.step()

        if epoch % save_interval == 0:
            torch.save(network.state_dict(), os.path.join(save_path, f"yolo_{epoch}.pth"))
            print(f"Loss: {loss.item()}")

    torch.save(network.state_dict(), os.path.join(save_path, f"yolo.pth"))


def main(img_h5_dir, ann_dir):
    device = torch.device("cuda")

    network = yolo.Network.NetWork(80)
    network.train().to(device, non_blocking=True)

    dataset = H5Dataset(img_h5_dir, ann_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, collate_fn=H5Dataset.collate_fn)

    train(network, dataloader, 1, "weight", 1)

    print(network)
    pass


if __name__ == '__main__':
    main("dataset/cocos/data.h5", "dataset/cocos/obj_record.pkl")
