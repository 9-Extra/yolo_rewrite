import math
from typing import Optional
import numpy
import torch
import cv2
import os
import tqdm
import random

from config import Config
from yolo.network import Yolo
import preprocess


class ImageReader:
    directory: str
    limit: int

    def __init__(self, directory: str, limit: Optional[int] = 50):
        self.directory = directory
        self.limit = limit
        pass

    def get_image_paths(self):
        suffix = (".jpg", ".png")

        names = os.listdir(self.directory)
        random.shuffle(names)

        result = []
        for name in names:
            if any(s in name for s in suffix):
                result.append(os.path.join(self.directory, name))
                if self.limit is not None and len(result) >= self.limit:
                    break

        return result


def _bbox_remap(
    bbox: numpy.ndarray,
    src_size: tuple[int, int],
    tar_size: tuple[int, int],
    scaleup=False,
):
    """
    将基于像素大小的prediction映射回原图像坐标
    """

    # 因为进行过加灰边，需要还原
    r = min(tar_size[0] / src_size[0], tar_size[1] / src_size[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = round(src_size[1] * r), round(src_size[0] * r)
    dw, dh = tar_size[1] - new_unpad[1], tar_size[0] - new_unpad[0]  # wh padding
    pad = numpy.array((dh, dw, dh, dw)) / 2

    return bbox / r - pad


def _draw_prediction(
    src_img_path: str,
    prediction: torch.Tensor,
    reshaped_img_shape: tuple[int, int],
    label_names: list[str],
):

    prediction: numpy.ndarray = prediction.numpy(force=True)
    img = cv2.imread(src_img_path)
    ori_img_shape = img.shape[:2]
    for obj in prediction:
        x1, y1, x2, y2 = _bbox_remap(
            obj[0:4], ori_img_shape, reshaped_img_shape
        ).astype(int)
        conf = obj[4]
        cls = int(obj[10])

        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label_names[cls]}:{conf:%}"
        img = cv2.putText(
            img,
            text,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return img


def detect(config: Config, input_images: list[str]):
    torch.set_float32_matmul_precision("highest")
    device = config.device

    label_names: list[str] = eval(open(config.run_path / "train_label.txt", "r").read())

    network: Yolo = Yolo(config.num_class)
    network.load_state_dict(torch.load(config.file_yolo_weight, weights_only=True))
    network.eval().to(device, non_blocking=True)

    def get_batch():
        for i in range(len(input_images) // config.batch_size):
            img_list = input_images[i * config.batch_size : (i + 1) * config.batch_size]
            images = []
            for img_path in img_list:
                img, _ = preprocess.process_data(img_path, [], config.img_size)
                images.append(img)

            img_batch = torch.from_numpy(numpy.stack(images))
            yield img_batch.to(device, non_blocking=True).float() / 255, img_list

        # 余下的
        img_list = input_images[(i + 1) * config.batch_size :]
        images = []
        for img_path in img_list:
            img, _ = preprocess.process_data(img_path, [], config.img_size)
            images.append(img)

        img_batch = torch.from_numpy(numpy.stack(images))
        img_batch = torch.nn.functional.pad(
            img_batch,
            (0, 0, 0, 0, 0, 0, 0, config.batch_size - img_batch.shape[0]),
            "constant",
            0,
        )
        yield img_batch.to(device, non_blocking=True).float() / 255, img_list

    save_dir = config.run_path / "result"
    os.makedirs(save_dir, exist_ok=True)

    for batch, img_list in tqdm.tqdm(
        get_batch(), total=math.ceil(len(input_images) / config.batch_size)
    ):
        predictions = network.inference(batch)
        for prediction, img_path in zip(predictions, img_list):
            img = _draw_prediction(img_path, prediction, config.img_size, label_names)
            cv2.imwrite(save_dir / os.path.basename(img_path), img)

    pass


if __name__ == "__main__":
    path = "/mnt/panpan/datasets/coco2017/val2017"
    detect(Config(), ImageReader(path).get_image_paths())
