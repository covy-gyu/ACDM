from ocr_utils import *

import infer.POCR.pocr_conf as OCR_conf
import infer.POCR.filter_common as filter_common
import infer.POCR.warp_common as warp_common
import logs.logmg as logmg

def infer_7(bomb, ocr_obj):
    logmg.i.log("# 비인가된 신관 결합")
    conf = OCR_conf.infer_7

    img_paths = bomb.img_path["CAM4"]
    images = list(map(cv2.imread, img_paths))

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

def infer_12(bomb, ocr_res):
    logmg.i.log("# 도색 표기 착오")
    std = "81MM COMP B KM374 고폭탄"

    processed_text = extract_text(std, ocr_res)
    



    pass

def infer_13(bomb, ocr_res):
    pass

def infer_19(bomb, ocr_res):
    pass

def infer_22(bomb, ocr_res):
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