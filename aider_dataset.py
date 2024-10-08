import random
import numpy as np
import torch
from PIL import Image
from datasets import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import InterpolationMode

from util import numpy_transform, convert_tensor


# 학습 데이터 변환
train_transforms = transforms.Compose([
    # transforms.Resize((256, 256), interpolation=InterpolationMode.LANCZOS),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.25),   # 25% 확률로 상하 반전
    transforms.RandomRotation(30),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    # transforms.Resize((224, 224), interpolation=InterpolationMode.LANCZOS),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class MemoryCachedDataset(Dataset):
    def __init__(self, cached_images, transform=None):
        self.cached_images = cached_images
        self.transform = transform
        # 클래스 이름 및 클래스 수
        # self.classes = self.data.classes
        # self.class_to_idx = self.data.class_to_idx

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, index):
        # 메모리에 캐싱된 이미지와 라벨을 가져옴
        image, label = self.cached_images[index]
        # Transform이 설정되어 있을 경우, 변환 적용
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# Dataloader 생성 함수
def get_dataloader(dataset_path, split_ratio, batch_size=32, shuffle=True):
    # Transform 정의
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])

    data = datasets.ImageFolder(root=dataset_path, transform=None)  # Transform 없이 데이터 로드

    samples = data.samples
    random.shuffle(samples)
    print(data.classes, len(data))

    # 데이터셋 길이 및 분할 계산
    train_size = int(len(samples) * split_ratio)
    # val_size = len(samples) - train_size

    # 메모리에 모든 이미지와 라벨을 캐싱
    print("Caching images in memory...")

    cached_images = []
    for img_path, label in tqdm(samples, ncols=75):
        image = Image.open(img_path)
        image = image.convert('RGB')
        if len(cached_images) < train_size:
            image = image.resize((256, 256), Image.LANCZOS)
        else:
            image = image.resize((224, 224), Image.LANCZOS)
        cached_images.append((image, label))
    print(f"Cached {len(cached_images)} images.")

    # 메모리 캐시된 데이터셋 생성
    train_dataset = MemoryCachedDataset(
        cached_images=cached_images[:train_size],
        transform=train_transforms
    )
    val_dataset = MemoryCachedDataset(
        cached_images=cached_images[train_size:],
        transform=val_transforms
    )

    # 클래스 개수 구하기
    num_classes = len(data.classes)


    # 데이터셋 분할
    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes
