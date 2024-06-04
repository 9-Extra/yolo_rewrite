import functools
import os

import cv2
import h5py
import numpy
import tqdm

from dataset.CocoBird import CocoBird
from dataset.CocoDataset import CocoDataset
from dataset.DroneDataset import DroneDataset, DroneTestDataset
from dataset.RawDataset import RawDataset, mix_raw_dataset
from dataset.BirdVSDroneBird import BirdVSDroneBird
import utils


def display(img, objs, label_names):
    h, w, _ = img.shape
    for obj in objs:
        cls, x, y, width, height = obj
        cls = int(cls)
        x1 = int((x - width / 2) * w)
        y1 = int((y - height / 2) * h)
        x2 = int((x + width / 2) * w)
        y2 = int((y + height / 2) * h)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.putText(img, label_names[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                          cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)


def process_data(origin_img: str, objs: list, target_size: tuple[int, int]):
    origin_img = cv2.imread(origin_img)
    img, ratio, (top, left) = utils.letterbox_fixed_size(origin_img, target_size)
    mapped_objs = numpy.empty((len(objs), 5), dtype=numpy.float32)
    for i, obj in enumerate(objs):
        x, y, width, height = obj[1]
        # center x, center y, width, height
        x = ((x + width / 2) * ratio[0] + left) / target_size[0]
        y = ((y + height / 2) * ratio[1] + top) / target_size[1]
        width *= ratio[0] / target_size[0]
        height *= ratio[1] / target_size[1]

        mapped_objs[i] = [obj[0], x, y, width, height]

    cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL, img)
    # cv2.cvtColor(img, cv2.COLOR_HSV2BGR_FULL, img)
    # cv2.imshow("show", img)
    # cv2.waitKey()
    img = img.transpose(2, 0, 1)

    return img, mapped_objs


def main(dist: str, data: RawDataset):
    os.makedirs(os.path.dirname(dist), exist_ok=True)
    target_size = [640, 640]

    process = functools.partial(process_data, target_size=target_size)

    image_count = len(data)
    bbox_count = sum(len(d.objs) for d in data)
    with h5py.File(dist, "w") as h5f:
        images: h5py.Dataset = h5f.create_dataset("image", (image_count, 3, *target_size), dtype=numpy.uint8)
        bbox_idx: h5py.Dataset = h5f.create_dataset("bbox_idx", (image_count, 2), dtype=numpy.uint32)
        bbox: h5py.Dataset = h5f.create_dataset("bbox", (bbox_count, 5), dtype=numpy.float32)

        h5f.create_dataset("obj_name", data=data.get_label_names(), dtype=h5py.special_dtype(vlen=str))

        bbox_idx_offset = 0
        for i, d in enumerate(tqdm.tqdm(data, total=image_count, desc="预处理图像")):
            # data.display(i)
            img, mapped_objs = process(d.img, d.objs)
            bbox_num = mapped_objs.shape[0]

            images.write_direct(numpy.ascontiguousarray(img), dest_sel=i)
            if bbox_num != 0:
                slice_ = slice(bbox_idx_offset, bbox_idx_offset + bbox_num)
                bbox_idx.write_direct(numpy.array([slice_.start, slice_.stop], dtype=numpy.uint32),
                                      dest_sel=i)
                bbox.write_direct(mapped_objs, dest_sel=slice_)
                bbox_idx_offset += bbox_num

    pass


if __name__ == '__main__':
    # drone = DroneDataset("G:/datasets/DroneTrainDataset", split="val")
    # coco_bird = CocoBird(r"D:\迅雷下载\train2017", r"D:\迅雷下载\annotations\instances_train2017.json")
    # # bird = BirdVSDroneBird("G:/datasets/BirdVsDrone/Birds")
    # print(f"包含 {len(drone)} 无人机，{len(coco_bird)}鸟")
    # mixed = mix_raw_dataset([drone, coco_bird])
    # # mixed = mix_raw_dataset([drone, bird, coco_bird])
    # # mixed = mix_raw_dataset(drone)
    # # main("preprocess/pure_bird.h5", bird)
    # # 10289 无人机，320鸟
    # main("preprocess/ood_val.h5", mixed)

    # drone_test = DroneTestDataset(r"G:\datasets\DroneTestDataset")
    # print("总图像数=", len(drone_test))
    # main("preprocess/test_pure_drone.h5", drone_test)

    # drone_test = DroneTestDataset(r"G:\datasets\DroneTestDataset")
    # print("总图像数=", len(drone_test))
    # coco = CocoDataset(r"D:\迅雷下载\train2017", r"D:\迅雷下载\annotations\instances_train2017.json")
    # coco = coco.ramdom_sample(len(drone_test))
    # for item in coco.items:
    #     item.objs.clear()
    # coco.label_names.clear()
    # mixed = mix_raw_dataset([drone_test, coco])
    # print("混合总图像数=", len(mixed))
    # main("preprocess/test_drone_with_coco.h5", mixed)

    drone_test = DroneTestDataset(r"G:\datasets\DroneTestDataset")
    print("总图像数=", len(drone_test))
    bird = BirdVSDroneBird("G:/datasets/BirdVsDrone/Birds")
    print("鸟图像数=", len(bird))
    mixed = mix_raw_dataset([drone_test, bird])
    print("混合总图像数=", len(mixed))
    main("preprocess/test_drone_with_bird.h5", mixed)



