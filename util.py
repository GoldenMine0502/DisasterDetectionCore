import cv2
import numpy as np
import torch

import torch.nn.functional as F
from torch import nn


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


# NumPy 기반 Transform 함수 정의
def numpy_transform(image, size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    NumPy를 이용한 이미지 변환 함수
    - 이미지 크기 조정 및 정규화를 NumPy로 처리
    - mean과 std는 정규화에 사용될 각 채널(RGB)의 평균 및 표준 편차 값
    """
    # 크기 조정
    image = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)

    # 정규화: (H, W, C) 순서를 (C, H, W)로 변환하고, 정규화 수행
    image = image.astype(np.float32) / 255.0  # 픽셀 값을 [0, 1] 범위로 조정
    image = (image - np.array(mean)) / np.array(std)  # 채널 별 정규화

    # (H, W, C) -> (C, H, W)로 변경
    image = np.transpose(image, (2, 0, 1))  # PyTorch의 Tensor 형태로 변환 (C, H, W)

    return image


# Define the mean and std for normalization (standard for ImageNet)
mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]  # R, G, B channel means
std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]   # R, G, B channel std deviations


def regularization(image):
    return (image - mean) / std


def convert_tensor(image):
    image = image.convert('RGB')  # 이미지를 RGB로 변환하여 로드
    image = np.array(image)
    image = numpy_transform(image)
    image_tensor = torch.from_numpy(image).float()  # Tensor로 변환

    return image_tensor


def collate_images_labels(batch):
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


class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, weight=None, reduction='mean'):
        """
        :param alpha: 양성 클래스의 가중치 (보통 0.25에서 0.75 사이).
        :param gamma: 초점 조정 매개변수 (보통 2.0).
        :param weight: 클래스 균형을 위한 가중치, 텐서 형태 [2] (이진 분류의 경우).
        :param reduction: 출력에 적용할 감소 방식: 'none' | 'mean' | 'sum'.
        """
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        # BCE 손실 계산
        # bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=self.weight, reduction='none')
        ce_loss = self.ce(inputs, targets)

        # # 확률 예측 값 계산
        # probs = torch.sigmoid(inputs)
        #
        # # Focal Loss 구성 요소 계산
        # pt = probs * targets + (1 - probs) * (1 - targets)  # pt = p (y=1일 때), 아니면 1-p
        # focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * ((1 - pt) ** self.gamma)

        # 예측 확률 계산 (Softmax를 통해)
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            # ce_loss *= self.alpha.gather(0, targets[:, 1])
            ce_loss *= self.alpha.gather(0, targets)

        # Focal Loss 계산
        loss = (1 - pt) ** self.gamma * ce_loss

        # BCE와 Focal Loss 결합
        # loss = focal_weight * bce_loss
        # loss = bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss