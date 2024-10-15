import os

import torch
from PIL import Image
from transformers import ViTForImageClassification

from aider_dataset import val_transforms, pre_val_transforms
from models.resnest.resnest import resnest269
from util import convert_tensor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# 체크포인트 로드 함수
def load_checkpoint(model, epoch, save_dir):
    checkpoint_path = os.path.join(save_dir, f'chkpt_{epoch}.pt')

    # 파일 존재 여부 확인
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # 모델과 옵티마이저 상태 복원
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 시작 에포크 설정
    start_epoch = checkpoint['epoch'] + 1

    print(f"Checkpoint loaded from '{checkpoint_path}', epoch {checkpoint['epoch']}.")
    return model, None, start_epoch


model_pretrained = ViTForImageClassification.from_pretrained(
    # 'google/vit-base-patch16-224',
    # 'openai/clip-vit-large-patch14',
    'google/vit-large-patch32-384',
    # num_labels=num_classes
)
config = model_pretrained.config
config.num_labels = 5
model = ViTForImageClassification(config)
model, optimizer, start_epoch = load_checkpoint(model, 36, 'chkpt/vision_transformer')


def inference(image):
    image = image.convert('RGB')
    # image = image.resize((224, 224), Image.LANCZOS)
    image = pre_val_transforms(image)
    image = val_transforms(image)
    image = image.unsqueeze(0)
    # image = convert_tensor(image).to(DEVICE)
    outputs = model(image).logits
    _, predicted = outputs.max(1)

    return to_numpy_tensor(outputs), to_numpy_tensor(predicted)


def inferencev2(image):
    image = image.convert('RGB')

def to_numpy_tensor(tensor):
    return tensor.squeeze().cpu().detach().numpy()