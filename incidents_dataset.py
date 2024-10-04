import json
import os

import numpy as np
from datasets import tqdm
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.nn.functional as F
import torch
from PIL import Image

from resnest import numpy_transform
import struct


def collate_fn(batch):
    images = []
    labels = []

    for image, label in batch:
        images.append(image)
        labels.append(label)

    return images, labels


def gpu_transform(images, size=(224, 224)):
    """
    images: GPU에 로드된 이미지 Tensor
    size: Resize할 이미지의 크기
    """
    # 크기 조정
    images = F.interpolate(images, size=size, mode='bilinear', align_corners=False)

    # 정규화 (mean, std는 일반적인 RGB 이미지의 기준값)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    images = (images - mean) / std

    return images


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


def write_compressed_images(json_path, file_name):
    with open(json_path, 'rt') as file:
        dataset = json.load(file)

    output_file = open(file_name, 'wb')

    total_count = 0

    for image_name in tqdm(dataset.keys(), ncols=75):
        # 데이터 전처리
        url = dataset[image_name]['url']
        image_path = os.path.join('../datasets/incidents/images', url)

        if not os.path.exists(image_path):  # 이미지가 존재하지 않으면 스킵
            continue

        image = Image.open(image_path)
        image = np.array(image)
        image = numpy_transform(image)

        label = set(map(lambda x: x[1], dataset[image_name]['incidents'].items()))[0]

        # 1. 경로 정보 (문자열)을 바이너리로 저장
        encoded_path = json_path.encode('utf-8')  # 문자열을 바이트로 인코딩
        path_length = len(encoded_path)  # 경로 문자열의 길이
        output_file.write(struct.pack('I', path_length))  # 경로 길이(4바이트, unsigned int) 저장
        output_file.write(encoded_path)  # 경로 정보 저장

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
            path_length_data = f.read(4)
            if not path_length_data:  # EOF 체크
                break
            path_length = struct.unpack('I', path_length_data)[0]
            path = f.read(path_length).decode('utf-8')

            # 2. 라벨 읽기
            label = struct.unpack('B', f.read(1))[0]

            # 3. 이미지 데이터 읽기
            image_shape = struct.unpack('III', f.read(12))  # 4바이트씩 3개
            image_size = image_shape[0] * image_shape[1] * image_shape[2]
            image_data = f.read(image_size)
            image = np.frombuffer(image_data, dtype=np.uint8).reshape(image_shape)

            # 4. 제네레이터로 반환
            yield path, label, image


# 반드시 worker = 1, shuffle=False 이어야 함
class IncidentsDataset(IterableDataset):
    def __init__(self, path, length):
        self.path = path
        self.length = length
        self.num_classes = 2
        self.classes = ['class_positive', 'class_negative']
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
    write_compressed_images('dataset/eccv_train.json', train_filename)
    write_compressed_images('dataset/eccv_val.json', val_filename)