"""
학습 데이터 증강을 위한 코드
기존 이미지와 mask값을 0도, 90도, 180도, 270도 회전
"""


import os
import cv2
import pandas as pd
import numpy as np
import torch
import albumentations as A

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE 값을 계산하는 함수 (이미지 크기를 입력받음)
def calculate_mask_rle(mask, shape):
    mask_rle_calculated = rle_encode(mask)
    return mask_rle_calculated

# RLE 값을 계산하는 함수 (이미지 크기를 입력받음)
def calculate_mask_rle(mask, shape):
    mask_rle_calculated = rle_encode(mask)
    return mask_rle_calculated

# 이미지 회전 및 mask_rle 계산 함수 (시계 방향 회전)
def rotate_image_and_calculate_mask_rle(img_path, mask_rle, angle):
    # 이미지를 읽어옴
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 이미지와 mask를 주어진 각도로 시계 방향으로 회전
    if angle == 0:
        transform = A.Compose([])
    elif angle == 1:
        transform = A.Rotate(limit=(90, 90), p=1)  # 원본 -> b 방향으로 회전 (270도)
    elif angle == 2:
        transform = A.Rotate(limit=(90, 90), p=1)  # 원본 -> c 방향으로 회전 (180도)
    elif angle == 3:
        transform = A.Rotate(limit=(90, 90), p=1)  # 원본 -> d 방향으로 회전 (90도)
    else:
        transform = A.Compose([])

    rotated_data = transform(image=image)
    rotated_image = rotated_data['image']

    # Mask도 동일하게 시계 방향으로 회전
    mask = rle_decode(mask_rle, image.shape[:2])
    if angle == 1:
        mask = np.rot90(mask, k=3, axes=(1, 0))  # 원본 -> b 방향으로 회전 (270도)
    elif angle == 2:
        mask = np.rot90(mask, k=2, axes=(1, 0))  # 원본 -> c 방향으로 회전 (180도)
    elif angle == 3:
        mask = np.rot90(mask, k=1, axes=(1, 0))  # 원본 -> d 방향으로 회전 (90도)

    # 회전한 이미지에 대한 mask_rle 계산
    rotated_mask_rle = rle_encode(mask)

    return rotated_image, rotated_mask_rle

# 이미지를 저장할 디렉토리 생성 및 이미지 저장 함수
def save_rotated_image(image, output_dir, img_id):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"TRAIN_{img_id:05d}.png")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def main():
    csv_file_path = './train.csv'  # 기존의 csv 파일 경로
    output_csv_file_path = './train_ch.csv'  # 출력할 csv 파일 경로
    output_image_dir = './train_img_ch'  # 이미지를 저장할 디렉토리 경로

    # 기존 csv 파일에서 이미지 경로와 mask_rle 값을 가져옴
    data = pd.read_csv(csv_file_path)

    # DataFrame에 저장할 데이터를 담을 리스트 초기화
    data_rotated = []

    for idx, row in data.iterrows():
        img_path = row['img_path']  # 이미지 경로 칼럼명을 맞게 변경해주세요
        mask_rle = row['mask_rle']  # mask_rle 칼럼명을 맞게 변경해주세요

        for angle in [0, 1, 2, 3]:  # 각도를 설정하여 반복
            # 이미지 회전 및 mask_rle 계산
            rotated_image, rotated_mask_rle = rotate_image_and_calculate_mask_rle(img_path, mask_rle, angle)

            # 이미지 저장
            img_id = len(data_rotated)  # 이미지 번호를 설정 (순차적으로 증가)
            save_rotated_image(rotated_image, output_image_dir, img_id)

            # 파일 경로, mask_rle 값, 이미지 ID를 리스트에 추가
            img_id = f"TRAIN_{img_id:05d}.png"
            img_path = os.path.join(output_image_dir, img_id)
            data_rotated.append({
                'img_id': img_id,
                'img_path': os.path.relpath(img_path),  # 상대 경로로 변환
                'mask_rle': rotated_mask_rle
            })

    # 리스트를 DataFrame으로 변환
    data_rotated = pd.DataFrame(data_rotated)

    # DataFrame을 csv 파일로 저장 (인덱스 번호를 제거하지 않고, 컬럼 이름을 포함)
    data_rotated.to_csv(output_csv_file_path, index=False, header=True)

if __name__ == "__main__":
    main()
