import io
import json
import os
import random

import cv2
import numpy as np
from datasets import tqdm
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch
from PIL import Image

import struct

from util import gpu_transform, numpy_transform
from torchvision import transforms


def collate_fn(batch):
    images = []
    labels = []

    # print(len(batch))

    for image, label in batch:
        images.append(image)
        labels.append(label)

    # print(images)
    # print(labels)

    images = torch.stack(images)
    labels = torch.stack(labels)

    return images, labels


def tensor_process(batch, device):
    images = []
    labels = []

    for image, label in batch:
        image = torch.from_numpy(np.array(image))
        image.to(device)
        label.to(device)

        images.append(gpu_transform(image))
        labels.append(label)

    images = torch.stack(images)
    labels = torch.stack(labels)

    return images, labels


def write_compressed_images(json_path, folder_name, file_name, split=False):
    with open(json_path, 'rt') as file:
        dataset = json.load(file)

    output_file = open(file_name, 'wb')

    total_count = 0

    image_names = list(dataset.keys())
    if split:
        image_names = image_names[:10]

    random.shuffle(image_names)

    for image_name in tqdm(image_names, ncols=75):
        # 데이터 전처리
        image_path = os.path.join('../datasets/incidents', folder_name, image_name)
        if not os.path.exists(image_path):  # 이미지가 존재하지 않으면 스킵
            continue

        image = Image.open(image_path)
        image = image.convert('RGB')
        image = np.array(image)
        image = cv2.resize(image, (224, 224))
        image = Image.fromarray(image)

        image_stream = io.BytesIO()
        image.save(image_stream, format='JPEG')  # Save as compressed JPEG with 85% quality
        compressed_image = image_stream.getvalue()

        image_size = len(compressed_image)
        output_file.write(struct.pack('I', image_size))  # Store the compressed image size
        output_file.write(compressed_image)

        label = next(iter(set(map(lambda x: x[1], dataset[image_name]['incidents'].items()))))

        print('info:', image_size, label)

        output_file.write(struct.pack('B', label))  # 라벨 (1바이트, unsigned char)

        total_count += 1

    output_file.close()
    print('{}: {}'.format(file_name, total_count))


def load_compressed_images(file_name):
    transform = transforms.ToTensor()

    with open(file_name, 'rb') as f:
        while True:
            # 이미지 읽기
            size_data = f.read(4)
            if not size_data:
                break
            image_size = struct.unpack('I', size_data)[0]

            image_data = f.read(image_size)
            if not image_data:
                break

            image_stream = io.BytesIO(image_data)
            image = Image.open(image_stream)
            image = image.convert('RGB')  # Convert to RGB format
            image = transform(image)

            # 라벨 읽기
            label = f.read(1)
            if not label:
                break
            label = struct.unpack('B', label)[0]
            label = torch.tensor(label)

            # 4. 제네레이터로 반환
            yield image, label


# 반드시 worker = 1, shuffle=False 이어야 함
class IncidentsDataset(IterableDataset):
    def __init__(self, path, length):
        self.path = path
        self.length = length
        self.num_classes = 2
        self.classes = ['class_negative', 'class_positive']
        self.data = []

    def __len__(self):
        return self.length

    def __iter__(self):
        return load_compressed_images(self.path)

# 100%|█████████████████████████| 1029726/1029726 [11:04:32<00:00, 25.83it/s]c
# cached_train.bin: 632516
# 100%|████████████████████████████████| 57207/57207 [33:55<00:00, 28.10it/s]
# cached_val.bin: 35117

train_filename = 'cached_train.bin'
val_filename = 'cached_val2.bin'

train_dataset = IncidentsDataset(train_filename, 632516)
val_dataset = IncidentsDataset(val_filename, 35117)


def get_train_loader(batch_size):
    # num_workers=0: 메인 프로세스 사용
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        # pin_memory=True,
    )

    return train_loader


def get_val_loader(batch_size):
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )

    return val_loader


if __name__ == '__main__':
    # 이미지 크롤링이 완료돼야 캐시할 수 있음
    # write_compressed_images('dataset/eccv_train.json', 'images', train_filename, split=True)
    # write_compressed_images('dataset/eccv_val.json', 'images_val', val_filename)

    for image, label in get_train_loader(batch_size=8):
        print(image.shape, label.shape)