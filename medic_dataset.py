import numpy as np
import torch
from PIL import Image
from datasets import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

from util import numpy_transform


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