import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

image_files = ['101201_01.bmp','104545_01.bmp','132344_01.bmp']
# 입력 폴더와 출력 폴더 경로 설정
input_folder = 'testset/normal/upper'
output_folder = 'infer/Wing'

for image_file in image_files:
        # 이미지 파일 불러오기
        image = cv2.imread(image_file)
        # 이미지 높이, 너비 계산
        height, width = image.shape[:2]

        # 이미지 세로로 2등분
        lower_image = image[height//2:, :]

        # # 이미지를 그레이스케일로 변환
        # gray_image = cv2.cvtColor(lower_image, cv2.COLOR_BGR2GRAY)

        print(np.max(lower_image),np.sum(lower_image >= np.max(lower_image)))
        
        # # 히스토그램 계산
        # hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        # # 히스토그램 그리기
        # plt.figure(figsize=(10, 5))

        # # 원본 이미지 출력
        # plt.subplot(1, 2, 1)
        # plt.imshow(cv2.cvtColor(lower_image, cv2.COLOR_BGR2RGB))
        # plt.title('Original Image')

        # # 히스토그램 출력
        # plt.subplot(1, 2, 2)
        # plt.plot(hist)
        # plt.title('Histogram')
        # plt.xlabel('Bins')
        # plt.ylabel('Frequency')

        # # 결과 이미지 파일 경로 생성
        # output_file = os.path.splitext(image_file)[0] + '.png'
        # output_path = os.path.join(output_folder, output_file)

        # # 이미지 저장
        # plt.savefig(output_path)
        # plt.close()

        # print(f'Saved histogram for {image_file} as {output_file}.')

