
import numpy as np
import torch
from PIL import Image
from datasets import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import InterpolationMode

from util import numpy_transform, convert_tensor


class MemoryCachedDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = datasets.ImageFolder(root=dataset_path, transform=None)  # Transform 없이 데이터 로드

        # 메모리에 모든 이미지와 라벨을 캐싱
        print(self.data.classes, len(self.data))
        print("Caching images in memory...")

        self.cached_images = []
        for img_path, label in tqdm(self.data.samples, ncols=75):
            image = Image.open(img_path)
            image = image.convert('RGB')
            image = convert_tensor(image)
            self.cached_images.append((image, label))
        print(f"Cached {len(self.cached_images)} images.")

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, index):
        # 메모리에 캐싱된 이미지와 라벨을 가져옴
        image, label = self.cached_images[index]

        return image, label


# Dataloader 생성 함수
def get_dataloader(dataset_path, split_ratio, batch_size=32, shuffle=True):
    # 메모리 캐시된 데이터셋 생성
    full_dataset = MemoryCachedDataset(dataset_path=dataset_path)

    # 데이터셋 길이 및 분할 계산
    train_size = int(len(full_dataset.data.samples) * split_ratio)
    val_size = len(full_dataset.data.samples) - train_size

    # 클래스 개수 구하기
    num_classes = len(full_dataset.data.classes)

    # 데이터셋 분할
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes
