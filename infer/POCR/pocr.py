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
        pts2 = np.float32([[0, 0], [width - 1, 0],
                        [width - 1, height - 1], [0, height - 1]])
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
        concatenated_image[0:image.shape[0], x_offset:x_offset+image.shape[1]] = image
        x_offset += image.shape[1]
    # print("이미지가 성공적으로 이어붙여졌습니다!")
    return concatenated_image

def get_ordered_images(start_index,images):
    ordered_images = []
    num_images = len(images)
    for i in range(start_index, start_index + num_images):
        image_index = i % num_images  # 순환적으로 인덱스 선택
        ordered_images.append(images[image_index])
    return ordered_images

def start_index(transformed_images):
    best_index = -1  # 가장 작은 누적 값의 이미지 인덱스
    best_cumsum = float('inf')  # 누적 값의 초기값을 무한대로 설정

    # 이미지들에 대해 반복하여 가장 작은 누적 값의 이미지 찾기
    for index, image in enumerate(transformed_images):
        hist = cv2.calcHist([image], [0], None, [256], [100, 256]) # 명도임계범위: 글자추측명도 100~256
        cumsum = np.cumsum(hist)
        # 명도값이 100 이상인 범위에서 누적 값 계산
        valid_cumsum = np.sum(cumsum[:])
        # print(f"이미지 {index} 누적 값:", valid_cumsum)
        if valid_cumsum < best_cumsum:
            best_cumsum = valid_cumsum
            best_index = index
    # print("가장 작은 누적 값의 이미지 인덱스:", best_index)
    return best_index+1

def apply_filter(image,k):
     # 샤프닝 필터 커널 정의
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    filtered = cv2.filter2D(image, -1, k*sharpening_kernel)
    return filtered

# def extract_text(std, image):
#     std = std.replace(" ", "")
#     std_len = len(std)
#     before_diff = float('inf')
#     now_diff = 0
#     ocr_result = ""

#     reader = easyocr.Reader(['ko', 'en'])  # Language parameter can be adjusted as per your requirement
#     result = reader.readtext(image)
#     for detection in result:
#         text = detection[1]
#         text = text.replace(" ", "")
#         ocr_result = ocr_result + text
#         now_diff = abs(std_len - len(ocr_result))
#         if now_diff >= before_diff:
#             break
#         else:
#             before_diff = now_diff
#         # print(f"Text: {text}")
#         # print(f"ocr: {ocr_result}")
#     return ocr_result

def text_matching_rate(std,target):    
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
        
        return float(c/p)

def find_similar_word(words, target, threshold):
    for word in words:
        wmt = text_matching_rate(word, target)
        if wmt >= threshold:
            return word

def init():
    pocr_obj = PororoOcr()
    return pocr_obj


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


def infer_7(bomb, ocr_obj):
    logmg.i.log("# 비인가된 신관 결합")
    conf = OCR_conf.infer_7

    img_paths = bomb.img_path['CAM4']
    images = list(map(cv2.imread, img_paths))

    # ===================================================================
    # Orig : Concat warped image(1 image) -> Apply filter(1 image) -> Run OCR(1 Text)
    # New : Apply filter(6 images) -> Run OCR(6 Texts)
    '''
    # Apply transforms
    cropped_img = list(map(fuse_warp.crop_img, images))
    warped_img = list(map(fuse_warp.warp_img, cropped_img))
    cc_img = fuse_warp.concat_images(warped_img)
    filtered_img = filter_common.apply_filters(cc_img)

    # Run OCR
    ocr_res = ocr_obj['reader'](filtered_img)
    ocr_text = ocr_obj['join'](ocr_res)
    ocr_text = common_err_correction(ocr_text, conf['common_ocr_error'])
    '''
    # Apply transforms
    cropped_img = list(map(lambda src: warp_common.crop_img(src, conf['crop_rect']), images))
    warped_img = list(map(lambda image: warp_common.warp_img(image, conf), cropped_img))
    cc_img = warp_common.concat_images(warped_img, conf)
    filtered_img = filter_common.apply_filters(cc_img, 1.5)

    # Run OCR
    ocr_result = ocr_obj.run_ocr(filtered_img)
    ocr_result = ''.join(ocr_result).replace(' ','')
    ocr_result = ocr_result.replace("'","")
    ocr_result = common_err_correction(ocr_result, conf['common_ocr_error'])
    # ===================================================================

    res = chk_include(ocr_result, conf['target'])
    result = "정상"
    if res is False:
        result = "비인가된 신관 결합"
        bomb.defect['head']['res'][6].append(DEFECT_CODE['head']['match'])
   
    logmg.i.log("기준표기 : %s", conf['target'])
    logmg.i.log("OCR : %s, 결과 : %s", ocr_result, result)

    bomb.update_infer_stat('head', 'match', res)
    return res


# def infer_12_13_19_22(bomb, ocr_obj):
#     img_paths = bomb.img_path['CAM3']
#     images = list(map(cv2.imread, img_paths))


def infer_12_13_19_22(bomb, ocr_obj):
    logmg.i.log("# 도색표기착오, 탄종혼합, 도색표기 불량, 흐림")
    conf = OCR_conf.infer_12_13_19_22
    cam = bomb.img_path['CAM3']
        # 미리 정의한 4개의 고정 좌표
    fixed_pts = [[521, 358], [821, 352], [760, 933], [570, 944]]
    # 이미지 파일 경로 리스트
    imgs = image_read(cam)
    # 이미지 변환 수행
    transformed_images = transform_image(imgs, fixed_pts)
    # 이미지 배열 반환 함수 호출
    ordered_images = get_ordered_images(start_index(transformed_images),transformed_images)
    concatenated_image = concat_images(ordered_images)
    
    image = cv2.cvtColor(concatenated_image, cv2.COLOR_BGR2GRAY)
    filted = apply_filter(image,k=2)
    gray = cv2.inRange(filted, 200, 255)
    kernel = np.ones((2, 1), np.uint8)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # 텍스트 추출
    std = "81MM COMP B KM374 고폭탄"
    ocr_result = ocr_obj.run_ocr(closed)
    ocr_result = ''.join(ocr_result).replace(' ','')
    if len(ocr_result) > 26:
        ocr_result = ocr_result[:24]
    ocr_result = common_err_correction(ocr_result, conf['common_ocr_error'])
    # ocr_result = extract_text(std, closed)

    # 글자수; 기준치, 일치율
    len_match_threshold = 0.5
    lmr = len_match_rate(std, ocr_result)
    # cv2.imshow("crossed.jpg",filted)
    # cv2.waitKey(0)
    
    # 정확도; 기준치, 일치율
    text_match_threshold = 0.5
    tmr = text_matching_rate(std, ocr_result)

    # 포함된특정단어; 기준치, 일치율
    words = ["고폭탄", "연습탄", "훈련탄", "조명탄", "백린연막탄"]
    word = find_similar_word(words, ocr_result, threshold=0.25)

    t = [True, True, True, True]
    t2 = ['paint_3','type','paint_2','paint_1']
    #고정 문구 나오는 값 확인, 표기착오, 탄종혼합 분류
    if lmr > len_match_threshold:
        if tmr > text_match_threshold:
            if word == "고폭탄":
                result = "정상"
            else:
                result = "도색표기착오"
                bomb.defect['body']['bot']['res'][6].append(DEFECT_CODE['body']['bot']['paint_3'])
                t[0] = False

        elif tmr <= text_match_threshold and word == ("연습탄" or "훈련탄" or "조명탄" or "백린연막탄"):
            result = "탄종 혼합"            
            bomb.defect['body']['bot']['res'][6].append(DEFECT_CODE['body']['bot']['type'])
            t[1] = False
        
        else:
            result = "도색표기불량"            
            bomb.defect['body']['bot']['res'][6].append(DEFECT_CODE['body']['bot']['paint_2'])
            t[2] = False

    else:
        result = "도색표기흐림"            
        bomb.defect['body']['bot']['res'][6].append(DEFECT_CODE['body']['bot']['paint_1'])
        t[3] = False

    logmg.i.log("기준표기 : %s",std)
    logmg.i.log("OCR : %s, 글자수일치율 : %f, 정확도 : %f, 포함특정단어 : %s, 결과 : %s",ocr_result,lmr,tmr,word,result)

    for i in range(4):
        bomb.update_infer_stat('body', t2[i], t[i])
    # print(f"OCR : {ocr_result}, 일치율 : {match_rate}, 결과 : {result}")
    # logmg.i.log("기준표기 : %s",std)
    # logmg.i.log("OCR : %s, 글자수일치율 : %f, 정확도 : %f, 포함특정단어 : %s, 결과 : %s",ocr_result,lmr,tmr,word,result)
    return True
    
def infer_14(bomb, ocr_obj):
    logmg.i.log("# 적색 경고표지 식별불가")
    conf = OCR_conf.infer_14

    img_paths = bomb.img_path['CAM4']
    images = list(map(cv2.imread, img_paths))

    # Apply transforms
    cropped_img = list(map(lambda src: warp_common.crop_img(src, conf['crop_rect']), images))
    filtered_img = list(map(lambda image: filter_common.apply_filters(image, 1.3), cropped_img))

    # Reduce images to run faster
    text_test = [np.sum(img) for img in filtered_img]  # High value = there is no text
    for _ in range(3):
        del_idx = text_test.index(max(text_test))
        del text_test[del_idx]
        del filtered_img[del_idx]

    # Rotate images
    rot_angle = list(range(-15, 16, 5))
    augmented_img = []
    for img in filtered_img:
        for angle in rot_angle:
            augmented_img.append(warp_common.rotate_img(img, angle))

    # Run OCR
    ocr_result = ''
    for img in augmented_img:
        ocr_res = ocr_obj.run_ocr(img)
        ocr_result += ''.join(ocr_res).replace(' ','')
    ocr_result = common_err_correction(ocr_result, conf['common_ocr_error'])


    cnt = 0
    for target in conf['target']:
        if target in ocr_result:
            cnt += 1
    res = cnt >= conf['threshold']
    result = "정상"

    if res is False:
        result = "적색 경고표지 식별불가"    
        bomb.defect['body']['bot']['res'][6].append(DEFECT_CODE['body']['top']['warn'])
    logmg.i.log("기준표기 : %s", conf['target'])
    logmg.i.log("OCR : %s 결과 : %s", ocr_result, result)

    bomb.update_infer_stat('body', 'warn', res)
    return res

import cv2
from infer.POCR.pororo import Pororo
from infer.POCR.pororo.pororo import SUPPORTED_TASKS
import warnings

warnings.filterwarnings('ignore')

class PororoOcr:
    def __init__(self, model: str = "brainocr", lang: str = "ko", **kwargs):
        self.model = model
        self.lang = lang
        self._ocr = Pororo(task="ocr", lang=lang, model=model, **kwargs)
        self.img_path = None
        self.ocr_result = {}

    def run_ocr(self, img_path: str, debug: bool = False):
        self.img_path = img_path
        self.ocr_result = self._ocr(img_path, detail=True)

        if self.ocr_result['description']:
            ocr_text = self.ocr_result["description"]
        else:
            ocr_text = ""

        return ocr_text

    @staticmethod
    def get_available_langs():
        return SUPPORTED_TASKS["ocr"].get_available_langs()

    @staticmethod
    def get_available_models():
        return SUPPORTED_TASKS["ocr"].get_available_models()

    def get_ocr_result(self):
        return self.ocr_result

    def get_img_path(self):
        return self.img_path

def do_infer(bomb, ocr_obj):
    infer_7(bomb, ocr_obj)
    infer_12_13_19_22(bomb, ocr_obj)
    infer_14(bomb, ocr_obj)