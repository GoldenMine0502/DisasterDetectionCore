
import torch
from datasets import tqdm
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import incidents_dataset
from models.resnest.resnest import resnest50, resnest269

import argparse
import yaml
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training configuration")
    parser.add_argument('--config', type=str, default='models/resnest/config.yaml', help='Path to the YAML configuration file')
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


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
    count = 0
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
            count += 1

    epoch_loss = running_loss / count
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


def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

    print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {start_epoch}")
    return model, optimizer, start_epoch


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
    start_epoch = config['start_epoch']

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
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    # Define the cosine learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    if start_epoch > 1:
        load_checkpoint(model, optimizer, os.path.join(checkpoint_dir, 'chkpt_{}.pt'.format(start_epoch - 1)))


    # 학습 및 검증 반복문
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch [{epoch}/{num_epochs}]")

        # 학습
        train(model, train_loader, criterion, optimizer, device)

        # 검증
        validate(model, val_loader, criterion, device)

        # n 에포크마다 체크포인트 저장
        if epoch % 1 == checkpoint_epoch:
            save_checkpoint(model, optimizer, epoch, checkpoint_dir)

        scheduler.step()


if __name__ == "__main__":
    main()
