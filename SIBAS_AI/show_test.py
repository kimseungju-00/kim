import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# 저장된 test.csv 파일과 submit.csv 파일을 읽어옵니다.
test_data = pd.read_csv('./test.csv')
submit_data = pd.read_csv('./submit.csv')

# show 폴더가 없으면 생성합니다.
os.makedirs('./show', exist_ok=True)

# 이미지와 마스크 시각화 및 저장
for i in range(len(submit_data)):
    mask_rle = submit_data['mask_rle'][i]
    if mask_rle == -1:
        continue

    # 마스크 디코딩
    img_id = submit_data['img_id'][i]
    img_path = test_data[test_data['img_id'] == img_id]['img_path'].values[0]
    image = cv2.imread(img_path)
    mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

    # 분할 영역의 윤곽선 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 분할 영역을 원본 이미지 위에 표시
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)

    # 분할 영역의 윤곽선을 원본 이미지 위에 그리기
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

    # 시각화 및 저장
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Segmented Image')
    axes[1].axis('off')

    # 이미지를 show 폴더에 저장
    save_path = os.path.join('./show', f'{img_id}.png')
    plt.savefig(save_path)
    plt.close()

print("이미지와 분할된 마스크를 show 폴더에 저장하였습니다.")
