import os

import torch

from models.resnest.resnest import resnest269
from resnest import convert_tensor


DEVICE = 'cuda' if torch.cuda_is_available() else 'cpu'


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
    return model, optimizer, start_epoch


model = resnest269(
    num_classes=6
)
model, optimizer, start_epoch = load_checkpoint(model, 20, 'chkpt/resnest')


def inference(image):
    image = convert_tensor(image).to(DEVICE)
    outputs = model(image)
    _, predicted = outputs.max(1)

    return predicted.tensor.cpu().detach().numpy()