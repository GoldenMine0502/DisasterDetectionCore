import numpy as np
import torch
from PIL import Image
from datasets import tqdm
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import incidents_dataset
import argparse
import yaml
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

from transformers import AutoImageProcessor, ViTForImageClassification
import torch.nn.functional as F


from util import numpy_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training configuration")
    parser.add_argument('--config', type=str, default='models/vision_transformer/config.yaml', help='Path to the YAML configuration file')
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# 학습 및 검증 함수 정의
def train(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    count = 0

    for images, labels in (pgbar := tqdm(trainloader, ncols=75)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images).logits  # ViT의 경우 `logits`가 예측값
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 손실 계산
        running_loss += loss.item()

        # 정확도 계산
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        count += 1

        pgbar.set_description("{:.2f}%, {:.4f}".format(100 * correct / total, running_loss / count))

    accuracy = 100 * correct / total
    return running_loss / len(trainloader), accuracy


def validate(model, validationloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in validationloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 정확도 계산
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return val_loss / len(validationloader), accuracy


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
    print(f'Train DataLoader has {len(train_loader.dataset)} samples')
    print(f'Validation DataLoader has {len(val_loader.dataset)} samples')
    print(f'Number of classes: {num_classes}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ViT 모델 설정 (이미지 분류용)
    model_pretrained = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        # num_labels=num_classes
    )
    config = model_pretrained.config
    config.num_labels = 2
    model = ViTForImageClassification(config)

    # CUDA 사용 여부 확인 및 설정
    model.to(device)

    # 손실 함수 및 최적화 도구 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    # Define the cosine learning rate scheduler
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

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

        # scheduler.step()

if __name__ == "__main__":
    main()
