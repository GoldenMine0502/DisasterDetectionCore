import json
import pickle
import os
import pprint
import random
from collections import defaultdict
import numpy as np
import requests
from tqdm import tqdm
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
# from IPython.display import IFrame


with open("dataset/eccv_train.json", "r") as fp:
    dataset = json.load(fp)


def extract_small():
    # 2. 키 값 추출 및 천 개만 선택
    keys = list(dataset.keys())[:1000]  # 첫 천 개의 키만 선택

    # 3. 선택한 키 기반 데이터 필터링
    filtered_data = {key: dataset[key] for key in keys}

    # 4. 새로운 JSON 파일로 저장
    with open('output_file.json', 'w', encoding='utf-8') as outfile:
        json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)


def download_image(url, image_path, timeout):
    try:
        if os.path.exists(image_path):
            return 0, 0

        # 상위 폴더 자동 생성
        folder_path = os.path.dirname(image_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(image_path, 'wb') as _:
            pass

        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code == 200:
            # print(f"Created directories: {folder_path}")

            # 이미지 파일 저장
            with open(image_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            # print(f"Image successfully downloaded: {image_name}")
            return 1, 0
        else:
            print(f"Failed to download {url} (status code: {response.status_code})")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

    return 0, 1


def download_images_in_parallel(image_name_and_urls, max_threads=64, timeout=16):
    # 1. ThreadPoolExecutor로 병렬 다운로드 실행
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # 2. 각 URL에 대해 다운로드 작업 할당
        futures = []
        for index, (image_name, url) in enumerate(tqdm(image_name_and_urls, ncols=75)):
            # 각 이미지마다 다른 저장 경로를 설정
            futures.append(
                executor.submit(download_image, url, os.path.join('../datasets/incidents/images', image_name), timeout)
            )

        # 3. 모든 작업이 완료될 때까지 대기
        succeed = 0
        failed = 0
        for future in (progressbar := tqdm(as_completed(futures), total=len(futures), ncols=75)):
            succeed_one, failed_one = future.result()  # 각 스레드 작업의 결과를 기다림

            succeed += succeed_one
            failed += failed_one

            progressbar.set_description('({}/{})'.format(succeed, succeed + failed))
    print("All images are downloaded using multithreading.")


counters = {
    "class_positive": defaultdict(int),
    "class_negative": defaultdict(int)
}

image_name_and_urls = []

for image_name in tqdm(dataset.keys(), ncols=75):
    url = dataset[image_name]['url']

    for category, label in dataset[image_name]['incidents'].items():
        if label == 1:
            counters["class_positive"][category] += 1
        elif label == 0:
            counters["class_negative"][category] += 1

    image_name_and_urls.append((image_name, url))
    # download_image(url, os.path.join('images', image_name))

    # pprint.pprint(dataset[image_name])
    # break


positive_sum = sum(counters['class_positive'].values())
negative_sum = sum(counters['class_negative'].values())

print(positive_sum, negative_sum, positive_sum + negative_sum)

download_images_in_parallel(image_name_and_urls, max_threads=64)
