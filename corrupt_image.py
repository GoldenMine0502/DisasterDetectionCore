import os

from datasets import tqdm
from PIL import Image


def remove_corrupt_images(directory):
    """
    지정한 디렉토리 내에서 손상된 이미지를 찾아 삭제하는 함수
    """
    print('removing corrupt images...')
    removed = 0
    total = 0
    for subdir, _, files in os.walk(directory):
        print(subdir, len(files))
        for file in (pgbar := tqdm(files, ncols=75)):
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path).convert('RGB') as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                # print(f"Removing corrupt image: {file_path} - {e}")
                removed += 1
                os.remove(file_path)

            total += 1
            pgbar.set_description('({}/{})'.format(removed, total))


if __name__ == "__main__":
    remove_corrupt_images('../datasets/incidents/images')