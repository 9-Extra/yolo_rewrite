import functools
import os

import cv2
import h5py
import numpy
import tqdm

from dataset.DroneDataset import DroneTestDataset
from dataset.RawDataset import RawDataset, mix_raw_dataset
from dataset.BirdVSDroneBird import BirdVSDroneBird


def letterbox_fixed_size(im, new_shape: tuple[int, int], color=(114, 114, 114), scaleup=False):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    height, width = im.shape[:2]  # current shape [height, width]
    # Scale ratio (new / old)
    r = min(new_shape[0] / height, new_shape[1] / width)
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = round(width * r), round(height * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    ratio = new_unpad[0] / width, new_unpad[1] / height

    if (width, height) != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh // 2, dh - (dh // 2)  # divide padding into 2 parts
    left, right = dw // 2, dw - (dw // 2)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (top, left)


def letterbox_stride(im: numpy.ndarray, stride: int = 32, color=(114, 114, 114)):
    height, width, channel = im.shape
    target_width = width + stride - 1 - (width - 1) % stride
    target_height = height + stride - 1 - (height - 1) % stride
    if target_width == width and target_height == height:
        return im  # skip

    result = numpy.full_like(im, color, shape=(target_height, target_width, channel), subok=False)
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    result[top:top + height, left: left + width] = im

    return result


def crop_image(img: numpy.ndarray, bbox, size: tuple[int, int]):
    x, y, w, h = bbox
    img = img[y:y + h, x:x + w, :]
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img


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
    img, ratio, (top, left) = letterbox_fixed_size(origin_img, target_size)
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
    drone_test = DroneTestDataset(r"G:\datasets\DroneTestDataset")
    print("总图像数=", len(drone_test))
    bird = BirdVSDroneBird("G:/datasets/BirdVsDrone/Birds")
    print("鸟图像数=", len(bird))
    mixed = mix_raw_dataset([drone_test, bird])
    print("混合总图像数=", len(mixed))
    main("preprocess/test_drone_with_bird.h5", mixed)
