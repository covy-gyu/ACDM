from PIL import Image

# 24비트 BMP 이미지를 엽니다.
input_image = Image.open('infer/Wing/101201_01.bmp')
# input_image = Image.open('infer/Wing/104545_01.bmp')
# input_image = Image.open('infer/Wing/132344_01.bmp')

# 이미지를 16색 비트맵 형식으로 변환합니다.
# dither 매개변수를 사용하여 색상 감지를 활성화합니다.
# output_image = input_image.convert("P", palette=Image.ADAPTIVE, colors=10,dither=1)
quantized_image = input_image.quantize(colors=8)
# 16색 비트맵 이미지를 저장합니다.
quantized_image.save("101201_01.bmp")

print("이미지 변환이 완료되었습니다.")

# import cv2
# import numpy as np

# # 이미지 로드
# # image = cv2.imread('infer/Wing/101201_01.bmp')
# # image = cv2.imread('infer/Wing/104545_01.bmp')
# image = cv2.imread('infer/Wing/132344_01.bmp')

# # 이미지 양자화
# quantized_image = cv2.convertScaleAbs((image // 80) * 75)

# # 결과 이미지 저장
# cv2.imwrite('132344_01.bmp', quantized_image)
