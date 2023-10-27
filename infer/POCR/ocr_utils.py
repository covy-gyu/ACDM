import cv2
import numpy as np
import easyocr

from conf import *
from jamo import h2j, j2hcj
from difflib import SequenceMatcher

import infer.POCR.pocr_conf as OCR_conf
import infer.POCR.filter_common as filter_common
import infer.POCR.warp_common as warp_common
import logs.logmg as logmg


def image_read(img_paths):
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        imgs.append(img)
    return imgs


def transform_image(imgs, pts):
    arr = []
    for img in imgs:
        rows, cols = img.shape[:2]
        # 변환 전 4개 좌표
        pts1 = np.float32(pts)
        # 변환 후 영상에 사용할 서류의 폭과 높이 계산
        w1 = abs(pts[2][0] - pts[3][0])
        w2 = abs(pts[1][0] - pts[0][0])
        h1 = abs(pts[1][1] - pts[2][1])
        h2 = abs(pts[0][1] - pts[3][1])
        width = max([w1, w2])  # 두 좌우 거리간의 최대값이 서류의 폭
        height = max([h1, h2])  # 두 상하 거리간의 최대값이 서류의 높이
        # 변환 후 4개 좌표
        pts2 = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        # 변환 행렬 계산
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        # 원근 변환 적용
        result = cv2.warpPerspective(img, mtrx, (int(width), int(height)))
        arr.append(result)
    return arr


def concat_images(images):
    # 이미지의 최대 높이와 전체 너비 계산
    max_height = max(image.shape[0] for image in images)
    total_width = sum(image.shape[1] for image in images)
    # 결과 이미지를 담을 빈 캔버스 생성
    concatenated_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    x_offset = 0
    for image in images:
        # 이미지를 결과 이미지에 복사하여 이어붙임
        concatenated_image[
            0 : image.shape[0], x_offset : x_offset + image.shape[1]
        ] = image
        x_offset += image.shape[1]
    # print("이미지가 성공적으로 이어붙여졌습니다!")
    return concatenated_image


def get_ordered_images(start_index, images):
    ordered_images = []
    num_images = len(images)
    for i in range(start_index, start_index + num_images):
        image_index = i % num_images  # 순환적으로 인덱스 선택
        ordered_images.append(images[image_index])
    return ordered_images


def start_index(transformed_images):
    best_index = -1  # 가장 작은 누적 값의 이미지 인덱스
    best_cumsum = float("inf")  # 누적 값의 초기값을 무한대로 설정

    # 이미지들에 대해 반복하여 가장 작은 누적 값의 이미지 찾기
    for index, image in enumerate(transformed_images):
        hist = cv2.calcHist(
            [image], [0], None, [256], [100, 256]
        )  # 명도임계범위: 글자추측명도 100~256
        cumsum = np.cumsum(hist)
        # 명도값이 100 이상인 범위에서 누적 값 계산
        valid_cumsum = np.sum(cumsum[:])
        # print(f"이미지 {index} 누적 값:", valid_cumsum)
        if valid_cumsum < best_cumsum:
            best_cumsum = valid_cumsum
            best_index = index
    # print("가장 작은 누적 값의 이미지 인덱스:", best_index)
    return best_index + 1


def apply_filter(image, k):
    # 샤프닝 필터 커널 정의
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    filtered = cv2.filter2D(image, -1, k * sharpening_kernel)
    return filtered


def extract_text(std, ocr_res):
    std_len = len(std)
    before_diff = float('inf')
    now_diff = 0
    res = ""

    for text in ocr_res:
        text = text.strip()
        res += text + " "
        res = res.strip()
        now_diff = abs(std_len - len(res))
        if now_diff >= before_diff:
            break
        else:
            before_diff = now_diff
        # print(f"Text: {text}")
        # print(f"ocr: {res}")
    return res


def text_matching_rate(std, target):
    jamo_std = j2hcj(h2j(std))
    jamo_target = j2hcj(h2j(target))
    match_rate = SequenceMatcher(None, jamo_std, jamo_target).ratio()
    return match_rate


def len_match_rate(std, target):
    std_len = len(std)
    target_len = len(target)

    if std_len < target_len:
        p = target_len
        c = std_len
    else:
        p = std_len
        c = target_len

    return float(c / p)


def find_similar_word(words, target, threshold):
    for word in words:
        wmt = text_matching_rate(word, target)
        if wmt >= threshold:
            return word

def chk_include(text, target):
    for s in target:
        if s in text:
            return True
    return False


def common_err_correction(orig, corr_err_list):
    for corr, err_list in corr_err_list:
        for err in err_list:
            orig = orig.replace(err, corr)
    return orig