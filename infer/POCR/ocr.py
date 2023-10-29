from infer.POCR.ocr_utils import *

import infer.POCR.pocr_conf as OCR_conf
import infer.POCR.filter_common as filter_common
import infer.POCR.warp_common as warp_common


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
        map(lambda image: filter_common.apply_filters(image, 1.4), cropped_img)
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
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        ocr_res = ocr_obj.run_ocr(img)
        logmg.i.log("Origin OCR result : %s", ocr_res)
        ocr_result += "".join(ocr_res).replace(" ", "")
    ocr_result = common_err_correction(ocr_result, conf["common_ocr_error"])

    ocr_result = ocr_result.lower()

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
    ocr_res = ocr_obj.run_ocr(closed)
    logmg.i.log("Origin OCR result : %s", ocr_res)

    res = ""
    for text in ocr_res:
        if "4" in text:
            text = text.replace("4", "4 ")
        res += " " + text
    res = common_err_correction(res, conf["common_ocr_error"])
    res = res.strip().split()
    logmg.i.log("OCR res : %s", res)

    return res


# 영어가 들어있는경우 20
# 한글인 경우 21


def infer_12(bomb, ocr_res):
    conf = OCR_conf.infer_12_13

    logmg.i.log("# 도색 표기 착오")
    is_ok = True

    target = "연습탄 훈련탄 고폭탄 CTG"
    logmg.i.log("target : %s", target)

    text_list = extract_text(conf["len"], ocr_res)
    logmg.i.log("Text_list : %s", text_list)
    target_list = [word for word in target.split()]

    matching_dict = find_word(target_list, text_list)
    if matching_dict:
        max_key = max(matching_dict, key=matching_dict.get)

        # check
        if max_key in ["고폭탄", "CTG"]:
            is_ok = True
        else:
            is_ok = False
            bomb.defect["body"]["bot"]["res"][6].append(
                DEFECT_CODE["body"]["bot"]["paint_3"]
            )

        logmg.i.log("일치율 dict: %s", matching_dict)
        logmg.i.log("최고 일치 단어 : %s", max_key)
    logmg.i.log("결과 : %s", is_ok)

    bomb.update_infer_stat("body", "paint_3", is_ok)
    return is_ok


def infer_13(bomb, ocr_res):
    conf = OCR_conf.infer_12_13
    logmg.i.log("# 탄종 혼합")
    is_ok = True

    target = "조명탄 연막탄 백린연막탄 고폭탄 CTG"
    logmg.i.log("target : %s", target)

    text_list = extract_text(conf["len"] + 2, ocr_res)
    logmg.i.log("Text_list : %s", text_list)
    target_list = [word for word in target.split()]

    matching_dict = find_word(target_list, text_list)
    if matching_dict:
        max_key = max(matching_dict, key=matching_dict.get)

        # check
        if max_key in ["고폭탄", "CTG"]:
            is_ok = True
        else:
            is_ok = False
            bomb.defect["body"]["bot"]["res"][6].append(
                DEFECT_CODE["body"]["bot"]["type"]
            )

        logmg.i.log("일치율 dict: %s", matching_dict)
        logmg.i.log("최고 일치 단어 : %s", max_key)
    logmg.i.log("결과 : %s", is_ok)

    bomb.update_infer_stat("body", "type", is_ok)
    return is_ok


def infer_19(bomb, ocr_res):
    logmg.i.log("# 도색 표기 불량")
    is_ok = True

    std = "81mm comp b km374 고폭탄 ctg"
    logmg.i.log("std : %s", std)

    text_list = extract_text(21, ocr_res)
    logmg.i.log("Text_list : %s", text_list)
    std_list = [word for word in std.split()]

    matching_dict = find_word(std_list, text_list)

    cnt = len(matching_dict)

    if matching_dict:
        min_key = min(matching_dict, key=matching_dict.get)

        # check
        if matching_dict[min_key] > 0.5:
            is_ok = True
        else:
            is_ok = False
            bomb.defect["body"]["bot"]["res"][6].append(
                DEFECT_CODE["body"]["bot"]["paint_2"]
            )

    logmg.i.log("일치율 dict: %s", matching_dict)
    logmg.i.log("find word count : %s", cnt)
    logmg.i.log("결과 : %s", is_ok)

    bomb.update_infer_stat("body", "paint_2", is_ok)
    return is_ok


def infer_22(bomb, ocr_res):
    logmg.i.log("# 도색 표기 흐림")
    is_ok = True

    std = "81MM COMP B 고폭탄 KM374"
    logmg.i.log("std : %s", std)

    text_list = extract_text(len(std), ocr_res)
    logmg.i.log("Text_list : %s", text_list)
    text = " ".join(text_list)

    lmr = len_match_rate(std, text)

    # check
    if lmr >= 0.7:
        is_ok = True
    else:
        is_ok = False
        bomb.defect["body"]["bot"]["res"][6].append(
            DEFECT_CODE["body"]["bot"]["paint_1"]
        )

    logmg.i.log("text : %s", text)
    logmg.i.log("글자수 일치율: %s", lmr)
    logmg.i.log("결과 : %s", is_ok)

    bomb.update_infer_stat("body", "paint_1", is_ok)
    return is_ok


def do_infer(bomb, ocr_obj):
    infer_7(bomb, ocr_obj)
    infer_14(bomb, ocr_obj)

    ocr_res = detecting_text(bomb, ocr_obj)
    infer_12(bomb, ocr_res)
    infer_13(bomb, ocr_res)

    infer_19(bomb, ocr_res)
    infer_22(bomb, ocr_res)

    # infer_12_13_19_22(bomb, ocr_obj)

    # return True
