import io
import json
import os

import numpy as np
import torch
from PIL import Image
from datasets import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

from util import numpy_transform, convert_tensor, collate_images_labels, regularization


def load_json(path):
    res = []
    with open(path, 'rt') as f:
        for line in f.readlines():
            res.append(json.loads(line))

    json_data = []
    labels = []

    for data in res:
        path = os.path.join('../datasets/MEDIC', data['image_path'])
        label = data['label']

        json_data.append((path, label))

        if label not in labels:
            labels.append(label)

    print(labels, len(json_data))

    return json_data, labels


labels = ['not_disaster', 'hurricane', 'earthquake', 'other_disaster', 'flood', 'fire', 'landslide']


class MedicDataset(Dataset):
    def __init__(self, json_path):
        data, _ = load_json(json_path)
        print(labels)
        # print(self.labels.index('not_disaster'))

        label_count = [0 for i in range(0, len(labels))]
        for _, label in tqdm(data, ncols=75):
            label_count[labels.index(label)] += 1

        print(label_count)

        print("Caching images in memory...")
        self.cached_images = []
        for img_path, label in tqdm(data, ncols=75):
            if not os.path.exists(img_path):
                print('not exist:', img_path, label)
                continue

            image = Image.open(img_path)
            image = image.convert('RGB')  # Convert to RGB format
            image = image.resize((224, 224), Image.LANCZOS)

            image_stream = io.BytesIO()
            image.save(image_stream, format='JPEG')  # Save as compressed JPEG with 85% quality

            label = torch.tensor(labels.index(label))
            self.cached_images.append((image_stream, label))

            del image

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, index):
        # 메모리에 캐싱된 이미지와 라벨을 가져옴
        image_stream, label = self.cached_images[index]

        # 스트림에서 읽고 초기 상태로 되돌림
        image = Image.open(image_stream)
        image_stream.seek(0)

        # 정규화
        image = self.transform(image)
        image = regularization(image)

        return image, label


train_dataset = MedicDataset('../datasets/MEDIC/multilabel/disaster_train.json')
val_dataset = MedicDataset('../datasets/MEDIC/multilabel/disaster_dev.json')


def get_train_loader(batch_size=32, num_workers=0):
    # num_workers=0: 메인 프로세스 사용
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_images_labels,
        shuffle=True,
        pin_memory=True,
    )

    return train_loader


def get_val_loader(batch_size=32, num_workers=0):
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_images_labels
    )

    return val_loader


if __name__ == '__main__':
    train_loader = MedicDataset('../datasets/MEDIC/multilabel/disaster_train.json')
    # train_data = load_json('../datasets/MEDIC/multilabel/disaster_train.json')
    #
    # labels = set()
    #
    # for data in train_data:
    #     path = os.path.join('../datasets/MEDIC/data', data['image_path'])
    #     label = data['label']
    #
    #     labels.add(label)
    #
    # print(labels)
    #

    # AIDER: ['collapsed_building', 'fire', 'flooded_areas', 'normal', 'traffic_incident']
    # MEDIC: {'hurricane', 'not_disaster', 'flood', 'fire', 'earthquake', 'landslide', 'other_disaster'}
    # AIDER['normal'] = MEDIC['not_disaster']
    # AIDER['fire'] = MEDIC['fire']
    # 나머지는안겹침
    # print(train_json)