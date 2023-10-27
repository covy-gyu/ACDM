from ocr_utils import *

def infer_7(bomb, ocr_obj):
    logmg.i.log("# 비인가된 신관 결합")
    conf = OCR_conf.infer_7

    img_paths = bomb.img_path["CAM4"]
    images = list(map(cv2.imread, img_paths))

    # ===================================================================
    # Orig : Concat warped image(1 image) -> Apply filter(1 image) -> Run OCR(1 Text)
    # New : Apply filter(6 images) -> Run OCR(6 Texts)
    """
    # Apply transforms
    cropped_img = list(map(fuse_warp.crop_img, images))
    warped_img = list(map(fuse_warp.warp_img, cropped_img))
    cc_img = fuse_warp.concat_images(warped_img)
    filtered_img = filter_common.apply_filters(cc_img)

    # Run OCR
    ocr_res = ocr_obj['reader'](filtered_img)
    ocr_text = ocr_obj['join'](ocr_res)
    ocr_text = common_err_correction(ocr_text, conf['common_ocr_error'])
    """
    # Apply transforms
    cropped_img = list(
        map(lambda src: warp_common.crop_img(src, conf["crop_rect"]), images)
    )
    warped_img = list(map(lambda image: warp_common.warp_img(image, conf), cropped_img))
    cc_img = warp_common.concat_images(warped_img, conf)
    filtered_img = filter_common.apply_filters(cc_img, 1.5)

    # Run OCR
    ocr_result = ocr_obj.run_ocr(filtered_img)
    logmg.i.log("Origin OCR result : %s", ocr_result)
    ocr_result = "".join(ocr_result).replace(" ", "")
    ocr_result = ocr_result.replace("'", "")
    ocr_result = common_err_correction(ocr_result, conf["common_ocr_error"])
    # ===================================================================

    res = chk_include(ocr_result, conf["target"])
    result = "정상"
    if res is False:
        result = "비인가된 신관 결합"
        bomb.defect["head"]["res"][6].append(DEFECT_CODE["head"]["match"])

    logmg.i.log("기준표기 : %s", conf["target"])
    logmg.i.log("Processed OCR : %s, 결과 : %s", ocr_result, result)

    bomb.update_infer_stat("head", "match", res)
    return res


# def infer_12_13_19_22(bomb, ocr_obj):
#     img_paths = bomb.img_path['CAM3']
#     images = list(map(cv2.imread, img_paths))


# def infer_12_13_19_22(bomb, ocr_obj):
#     logmg.i.log("# 도색표기착오, 탄종혼합, 도색표기 불량, 흐림")
#     conf = OCR_conf.infer_12_13_19_22
#     cam = bomb.img_path["CAM3"]
#     # 미리 정의한 4개의 고정 좌표
#     fixed_pts = [[521, 358], [821, 352], [760, 933], [570, 944]]
#     # 이미지 파일 경로 리스트
#     imgs = image_read(cam)
#     # 이미지 변환 수행
#     transformed_images = transform_image(imgs, fixed_pts)
#     # 이미지 배열 반환 함수 호출
#     ordered_images = get_ordered_images(
#         start_index(transformed_images), transformed_images
#     )
#     concatenated_image = concat_images(ordered_images)

#     image = cv2.cvtColor(concatenated_image, cv2.COLOR_BGR2GRAY)
#     filted = apply_filter(image, k=2)
#     gray = cv2.inRange(filted, 200, 255)
#     kernel = np.ones((2, 1), np.uint8)
#     closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

#     # 텍스트 추출
#     std = "81MM COMP B KM374 고폭탄"
#     ocr_result = ocr_obj.run_ocr(closed)
#     logmg.i.log("Origin OCR result : %s", ocr_result)

#     ocr_result = "".join(ocr_result).replace(" ", "")
#     if len(ocr_result) > 26:
#         ocr_result = ocr_result[:24]
#     ocr_result = common_err_correction(ocr_result, conf["common_ocr_error"])
#     # ocr_result = extract_text(std, closed)

#     # 글자수; 기준치, 일치율
#     len_match_threshold = 0.5
#     lmr = len_match_rate(std, ocr_result)
#     # cv2.imshow("crossed.jpg",filted)
#     # cv2.waitKey(0)

#     # 정확도; 기준치, 일치율
#     text_match_threshold = 0.5
#     tmr = text_matching_rate(std, ocr_result)

#     # 포함된특정단어; 기준치, 일치율
#     words = ["CTG", "고폭탄", "연습탄", "훈련탄", "조명탄", "백린연막탄"]
#     word = find_similar_word(words, ocr_result, threshold=0.25)

#     t = [True, True, True, True]
#     t2 = ["paint_3", "type", "paint_2", "paint_1"]
#     # 고정 문구 나오는 값 확인, 표기착오, 탄종혼합 분류
#     if lmr > len_match_threshold:
#         if tmr > text_match_threshold:
#             if word == "고폭탄" or word == "CTG":
#                 result = "정상"
#             else:
#                 result = "도색표기착오"
#                 bomb.defect["body"]["bot"]["res"][6].append(
#                     DEFECT_CODE["body"]["bot"]["paint_3"]
#                 )
#                 t[0] = False

#         elif tmr <= text_match_threshold and word == (
#             "연습탄" or "훈련탄" or "조명탄" or "백린연막탄"
#         ):
#             result = "탄종 혼합"
#             bomb.defect["body"]["bot"]["res"][6].append(
#                 DEFECT_CODE["body"]["bot"]["type"]
#             )
#             t[1] = False

#         else:
#             result = "도색표기불량"
#             bomb.defect["body"]["bot"]["res"][6].append(
#                 DEFECT_CODE["body"]["bot"]["paint_2"]
#             )
#             t[2] = False

#     else:
#         result = "도색표기흐림"
#         bomb.defect["body"]["bot"]["res"][6].append(
#             DEFECT_CODE["body"]["bot"]["paint_1"]
#         )
#         t[3] = False

#     logmg.i.log("기준표기 : %s", std)
#     logmg.i.log(
#         "Processed OCR : %s, 글자수일치율 : %f, 정확도 : %f, 포함특정단어 : %s, 결과 : %s",
#         ocr_result,
#         lmr,
#         tmr,
#         word,
#         result,
#     )

#     for i in range(4):
#         bomb.update_infer_stat("body", t2[i], t[i])

#     return True



def infer_14(bomb, ocr_obj):
    logmg.i.log("# 적색 경고표지 식별불가")
    conf = OCR_conf.infer_14

    img_paths = bomb.img_path["CAM4"]
    images = list(map(cv2.imread, img_paths))

    # Apply transforms
    cropped_img = list(
        map(lambda src: warp_common.crop_img(src, conf["crop_rect"]), images)
    )
    filtered_img = list(
        map(lambda image: filter_common.apply_filters(image, 1.3), cropped_img)
    )

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
    ocr_result = ""
    for img in augmented_img:
        ocr_res = ocr_obj.run_ocr(img)
        logmg.i.log("Origin OCR result : %s", ocr_res)
        ocr_result += "".join(ocr_res).replace(" ", "")
    ocr_result = common_err_correction(ocr_result, conf["common_ocr_error"])

    cnt = 0
    for target in conf["target"]:
        if target in ocr_result:
            cnt += 1
    res = cnt >= conf["threshold"]
    result = "정상"

    if res is False:
        result = "적색 경고표지 식별불가"
        bomb.defect["body"]["bot"]["res"][6].append(DEFECT_CODE["body"]["top"]["warn"])
    logmg.i.log("기준표기 : %s", conf["target"])
    logmg.i.log("Processed OCR : %s 결과 : %s", ocr_result, result)

    bomb.update_infer_stat("body", "warn", res)
    return res


from infer.POCR.pororo import Pororo
from infer.POCR.pororo.pororo import SUPPORTED_TASKS
import warnings

warnings.filterwarnings("ignore")


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

        if self.ocr_result["description"]:
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


def detecting_text(bomb, ocr_obj):
    logmg.i.log("# 도색 표기 확인")
    conf = OCR_conf.infer_12_13_19_22
    cam = bomb.img_path["CAM3"]
    # 미리 정의한 4개의 고정 좌표
    fixed_pts = [[521, 358], [821, 352], [760, 933], [570, 944]]
    # 이미지 파일 경로 리스트
    imgs = image_read(cam)
    # 이미지 변환 수행
    transformed_images = transform_image(imgs, fixed_pts)
    # 이미지 배열 반환 함수 호출
    ordered_images = get_ordered_images(
        start_index(transformed_images), transformed_images
    )
    concatenated_image = concat_images(ordered_images)

    image = cv2.cvtColor(concatenated_image, cv2.COLOR_BGR2GRAY)
    filted = apply_filter(image, k=2)
    gray = cv2.inRange(filted, 200, 255)
    kernel = np.ones((2, 1), np.uint8)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 텍스트 추출
    res = ocr_obj.run_ocr(closed)
    logmg.i.log("Origin OCR result : %s", res)


    return res
    pass

def do_infer(bomb, ocr_obj):


    infer_7(bomb, ocr_obj)

    ocr_res = detecting_text(bomb, ocr_obj)
    # infer_12_13_19_22(bomb, ocr_obj)
    infer_12(bomb, ocr_res)
    infer_13(bomb, ocr_res)
    infer_19(bomb, ocr_res)
    infer_22(bomb, ocr_res)
   
    infer_14(bomb, ocr_obj)

    # return True
