import json
import os
import random

import numpy as np
from datasets import tqdm
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch
from PIL import Image

import struct

from util import gpu_transform, numpy_transform


def collate_fn(batch):
    images = []
    labels = []

    for image, label in batch:
        images.append(image)
        labels.append(label)

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


def write_compressed_images(json_path, folder_name, file_name):
    with open(json_path, 'rt') as file:
        dataset = json.load(file)

    output_file = open(file_name, 'wb')

    total_count = 0

    image_names = list(dataset.keys())
    random.shuffle(image_names)

    for image_name in tqdm(image_names, ncols=75):
        # 데이터 전처리
        image_path = os.path.join('../datasets/incidents', folder_name, image_name)
        if not os.path.exists(image_path):  # 이미지가 존재하지 않으면 스킵
            continue

        image = Image.open(image_path)
        image = image.convert('RGB')
        image = np.array(image)
        image = numpy_transform(image)

        label = next(iter(set(map(lambda x: x[1], dataset[image_name]['incidents'].items()))))

        # 1. 경로 정보 (문자열)을 바이너리로 저장
        # encoded_path = json_path.encode('utf-8')  # 문자열을 바이트로 인코딩
        # path_length = len(encoded_path)  # 경로 문자열의 길이
        # output_file.write(struct.pack('I', path_length))  # 경로 길이(4바이트, unsigned int) 저장
        # output_file.write(encoded_path)  # 경로 정보 저장

        # 2. 라벨 (정수)을 바이너리로 저장
        output_file.write(struct.pack('B', label))  # 라벨 (1바이트, unsigned char)

        # 3. 이미지 (numpy array)를 바이너리로 저장
        image_shape = image.shape  # 이미지의 형상 (height, width, channels)
        output_file.write(struct.pack('III', *image_shape))  # 이미지의 형상 정보 (4바이트씩 3개, unsigned int)
        output_file.write(image.tobytes())  # 이미지 데이터를 바이트로 변환하여 저장

        total_count += 1

    output_file.close()
    print('{}: {}'.format(file_name, total_count))


def load_compressed_images(file_name):
    with open(file_name, 'rb') as f:
        while True:
            # 1. 경로 정보 읽기
            # path_length_data = f.read(4)
            # if not path_length_data:  # EOF 체크
            #     break
            # path_length = struct.unpack('I', path_length_data)[0]
            # path = f.read(path_length).decode('utf-8')

            # 2. 라벨 읽기
            label = struct.unpack('B', f.read(1))[0]

            # 3. 이미지 데이터 읽기
            image_shape = struct.unpack('III', f.read(12))  # 4바이트씩 3개
            image_size = image_shape[0] * image_shape[1] * image_shape[2]
            image_data = f.read(image_size)
            image = np.frombuffer(image_data, dtype=np.uint8).reshape(image_shape)

            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label)

            # 4. 제네레이터로 반환
            yield image, label


# 반드시 worker = 1, shuffle=False 이어야 함
class IncidentsDataset(IterableDataset):
    def __init__(self, path):
        self.path = path
        self.num_classes = 2
        self.classes = ['class_negative', 'class_positive']
        self.data = []

    def __iter__(self):
        return load_compressed_images(self.path)


train_filename = 'cached_train.bin'
val_filename = 'cached_val.bin'

train_dataset = IncidentsDataset(train_filename)
val_dataset = IncidentsDataset(val_filename)


def get_train_loader(batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)  # num_workers=0: 메인 프로세스 사용

    return train_loader


def get_val_loader(batch_size):
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    return val_loader


if __name__ == '__main__':
    # 이미지 크롤링이 완료돼야 캐시할 수 있음
    write_compressed_images('dataset/eccv_train.json', 'images', train_filename)
    write_compressed_images('dataset/eccv_val.json', 'images_val', val_filename)