import os

import cv2.gapi
import h5py
import numpy
import tqdm

from dataset.DroneDataset import DroneDataset
from dataset.RawDataset import RawDataset
from dataset.h5Dataset import ObjectRecord


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


def letterbox(im, new_shape: list, color=(114, 114, 114), scaleup=True):
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


def process_data(img: str, objs: list, target_size: list) -> tuple[numpy.ndarray, numpy.ndarray]:
    img = cv2.imread(img)
    img, ratio, (top, left) = letterbox(img, target_size)
    mapped_objs = numpy.empty((len(objs), 5), dtype=numpy.float32)
    for i, obj in enumerate(objs):
        x, y, width, height = obj[1]
        # center x, center y, width, height
        x = ((x + width / 2) * ratio[0] + left) / target_size[0]
        y = ((y + height / 2) * ratio[1] + top) / target_size[1]
        width *= ratio[0] / target_size[0]
        height *= ratio[1] / target_size[1]

        mapped_objs[i] = [obj[0], x, y, width, height]

    return img, mapped_objs


def main(dest_dir: str, data: RawDataset):
    os.makedirs(dest_dir, exist_ok=True)
    target_size = [640, 640]

    # data.items = data.items[:1000]
    obj_record = ObjectRecord(data.get_label_names(), [])
    with h5py.File(os.path.join(dest_dir, "data.h5"), "w") as h5f:
        h5f.create_dataset("image", (len(data), 3, *target_size), dtype=numpy.uint8)
        images: h5py.Dataset = h5f["image"]
        for i, d in enumerate(tqdm.tqdm(data)):
            img, mapped_objs = process_data(d.img, d.objs, target_size)

            obj_record.objs.append(mapped_objs)
            # tasks.append(delayed(process_data)(path, d.objs, target_size))
            # display(img, mapped_objs, data.label_names)

            img = numpy.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
            images.write_direct(img, dest_sel=i)

    # obj_record.objs = Parallel(n_jobs=6, batch_size=16, verbose=10)(tasks)

    obj_record.dump(os.path.join(dest_dir, "obj_record.pkl"))

    pass


if __name__ == '__main__':
    main("dataset/cocos", DroneDataset("G:/datasets/DroneTrainDataset"))
