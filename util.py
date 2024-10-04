import cv2
import numpy as np
import torch

import torch.nn.functional as F


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
    image = cv2.resize(image, size)

    # 정규화: (H, W, C) 순서를 (C, H, W)로 변환하고, 정규화 수행
    image = image.astype(np.float32) / 255.0  # 픽셀 값을 [0, 1] 범위로 조정
    image = (image - np.array(mean)) / np.array(std)  # 채널 별 정규화

    # (H, W, C) -> (C, H, W)로 변경
    image = np.transpose(image, (2, 0, 1))  # PyTorch의 Tensor 형태로 변환 (C, H, W)

    return image

