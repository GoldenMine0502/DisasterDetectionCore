import numpy as np
import torch
from PIL import Image
from datasets import tqdm
from torch import nn, optim

import incidents_dataset
from models.resnest.resnest import resnest50, resnest269

import argparse
import yaml
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

import torch.nn.functional as F

import cv2

from util import numpy_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training configuration")
    parser.add_argument('--config', type=str, default='models/resnest/config.yaml', help='Path to the YAML configuration file')
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# 메모리에 미리 이미지를 로드하여 캐시하는 Dataset 클래스
class MemoryCachedDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        root: ImageFolder와 동일하게 루트 디렉토리 경로를 지정
        transform: 이미지 변환을 위한 torchvision.transforms 객체
        """
        # ImageFolder로 데이터를 로드하여 캐싱
        self.data = datasets.ImageFolder(root=root, transform=None)  # Transform 없이 데이터 로드
        self.transform = transform

        # 메모리에 모든 이미지와 라벨을 캐싱
        print("Caching images in memory...")
        self.cached_images = []
        for img_path, label in tqdm(self.data.samples, ncols=75):
            image = Image.open(img_path)
            self.cached_images.append((convert_tensor(image), label))
        print(f"Cached {len(self.cached_images)} images.")

        # 클래스 이름 및 클래스 수
        self.classes = self.data.classes
        self.class_to_idx = self.data.class_to_idx

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, index):
        # 메모리에 캐싱된 이미지와 라벨을 가져옴
        image, label = self.cached_images[index]
        # Transform이 설정되어 있을 경우, 변환 적용
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def convert_tensor(image):
    image = image.convert('RGB')  # 이미지를 RGB로 변환하여 로드
    image = np.array(image)
    image = numpy_transform(image)
    image_tensor = torch.from_numpy(image).float()  # Tensor로 변환

    return image_tensor

# Dataloader 생성 함수
def get_dataloader(dataset_path, split_ratio, batch_size=32, shuffle=True):
    # Transform 정의
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])

    # 메모리 캐시된 데이터셋 생성
    full_dataset = MemoryCachedDataset(root=dataset_path)

    # 클래스 개수 구하기
    num_classes = len(full_dataset.classes)

    # 데이터셋 길이 및 분할 계산
    train_size = int(len(full_dataset) * split_ratio)
    val_size = len(full_dataset) - train_size

    # 데이터셋 분할
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes


# 모델 학습 함수
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    count = 0
    for inputs, labels in (pgbar := tqdm(train_loader, ncols=75)):
        inputs, labels = inputs.to(device), labels.to(device)

        # inputs = gpu_transform(inputs)

        # Optimizer 초기화
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 손실 및 정확도 계산
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        count += 1

        epoch_loss = running_loss / count
        accuracy = 100. * correct / total

        pgbar.set_description('{:.2f}%, {:.4f}'.format(accuracy, epoch_loss))


    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return epoch_loss, accuracy


# 모델 검증 함수
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, ncols=75):
            inputs, labels = inputs.to(device), labels.to(device)

            # inputs = gpu_transform(inputs)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 손실 및 정확도 계산
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = 100. * correct / total

    print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return epoch_loss, accuracy


# 체크포인트 저장 함수
def save_checkpoint(model, optimizer, epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, f'chkpt_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def main():
    # 인자 파싱
    args = parse_args()

    # 설정 파일 로드
    config = load_config(args.config)

    # yaml 설정 값 가져오기
    # dataset_path = config['dataset_path']
    learning_rate = config['learning_rate']
    # split_ratio = config['split_ratio']
    num_epochs = config['epoch']
    checkpoint_dir = config['checkpoint_dir']
    checkpoint_epoch = config['checkpoint_epoch']
    batch_size = config['batch_size']

    # 잘못된 이미지 제거
    # remove_corrupt_images(dataset_path)

    # Dataloader 및 클래스 수 구하기
    # train_loader, val_loader, num_classes = get_dataloader(
    #     dataset_path,
    #     split_ratio,
    #     batch_size=8,
    # )

    train_loader = incidents_dataset.get_train_loader(batch_size=batch_size)
    val_loader = incidents_dataset.get_val_loader(batch_size=batch_size)
    num_classes = 2

    # 출력 확인
    # print(f'Train DataLoader has {len(train_loader.dataset)} samples')
    # print(f'Validation DataLoader has {len(val_loader.dataset)} samples')
    print(f'Number of classes: {num_classes}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnest269(
        num_classes=num_classes
    )
    model.to(device)

    # 손실 함수 및 최적화 도구 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 학습 및 검증 반복문
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch [{epoch}/{num_epochs}]")

        # 학습
        train(model, train_loader, criterion, optimizer, device)

        # 검증
        validate(model, val_loader, criterion, device)

        # n 에포크마다 체크포인트 저장
        if epoch % 1 == checkpoint_epoch:
            save_checkpoint(model, optimizer, epoch, checkpoint_dir)


if __name__ == "__main__":
    main()
