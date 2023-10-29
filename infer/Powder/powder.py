import math

from conf import *
from util import *

import infer.Powder.powder_conf as powder_conf
import logs.logmg as logmg


def infer_30(bomb):
    """
    추진 장약 부분 망실
    """

    logmg.i.log("# 추진장약 부분망실")
    conf = powder_conf.infer_30

    images = load_images(bomb.img_path["CAM1"], cv2.IMREAD_GRAYSCALE)
    img_h, img_w = images[0].shape

    val_threshold = conf["brightness_threshold"]
    crop_rect = conf["crop_rect"]

    is_ok = True
    for i, img in enumerate(images):
        upper_img = img  # Use upper half of image

        # Mask pixels have brightness between values of val_thresholds
        cropped_img = upper_img[
            crop_rect["top"] : crop_rect["bottom"],
            crop_rect["left"] : crop_rect["right"],
        ]
        value_mask = cv2.inRange(
            cropped_img, val_threshold["lower"], val_threshold["upper"]
        )

        # Calculate masked pixel density
        pixel_cnt = cv2.countNonZero(value_mask)
        total_pixel = value_mask.shape[0] * value_mask.shape[1]
        value_density = pixel_cnt / total_pixel
        contours, _ = cv2.findContours(
            value_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        result = "정상"

        if conf["density_threshold"] < value_density and conf["pixel_cnt"] < pixel_cnt:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if h > 110:
                    result = "부분 망실"
                    bomb.defect["powder"]["bot"]["res"][i].append(
                        DEFECT_CODE["powder"]["bot"]["exist"]
                    )
                    is_ok = False
        logmg.i.log("%d pixel_cnt: %s 밝기 픽셀 밀집도: %.3f 결과: %s", i, pixel_cnt, value_density, result)

    bomb.update_infer_stat("powder", "exist", is_ok)
    return is_ok


def infer_31(bomb):
    """
    추진 장약 부적절한 위치
    """

    logmg.i.log("# 추진장약 부적절한 위치")
    conf = powder_conf.infer_31

    images = load_images(bomb.img_path["CAM1"], cv2.IMREAD_GRAYSCALE)
    img_h, img_w = images[0].shape

    template_img = cv2.imread(conf["template_img"], cv2.IMREAD_GRAYSCALE)

    mask_rect = conf["mask_rect"]
    is_ok = True
    for i, img in enumerate(images):
        upper_img = img[: img_h // 2]
        binary = get_binary(upper_img, mask_rect, 30, True)

        # Do template matching
        infer = cv2.matchTemplate(binary, template_img, cv2.TM_CCOEFF_NORMED)
        result = "정상"
        if abs(infer[0][0]) > conf["infer_threshold"]:
            result = "부적절한 위치"
            bomb.defect["powder"]["top"]["res"][i].append(
                DEFECT_CODE["powder"]["top"]["position"]
            )
            is_ok = False
        logmg.i.log("%d 마스크 유사도 : %.8f 결과 : %s", i, infer[0][0], result)

    bomb.update_infer_stat("powder", "position", is_ok)
    return is_ok


def infer_32(bomb):
    """
    추진 장약 장력 상실 또는 파손
    """

    logmg.i.log("# 추진장약 장력상실 또는 파손")
    conf = powder_conf.infer_32

    images = load_images(bomb.img_path["CAM2"])

    mask_rect = conf["mask_rect"]
    is_ok = True
    for i, img in enumerate(images):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(gray_img)
        mask = cv2.rectangle(
            mask,
            mask_rect["top_left"],
            mask_rect["bottom_right"],
            color=(255, 255, 255),
            thickness=cv2.FILLED,
        )
        hist = cv2.calcHist([img], [0], mask, [30], [0, 30])

        max_idx = np.argmax(hist)
        result = "정상"
        if max_idx < 7:
            result = "장력상실 또는 파손"
            bomb.defect["powder"]["bot"]["res"][i].append(
                DEFECT_CODE["powder"]["bot"]["condition"]
            )
            is_ok = False
        logmg.i.log("%d 결함 구분 : %s", i, result)

    bomb.update_infer_stat("powder", "condition", is_ok)
    return is_ok


def infer_33(bomb):
    """
    추진 장약 변색
    """
    logmg.i.log("# 추진장약 변색")
    conf = powder_conf.infer_33

    images = load_images(bomb.img_path["CAM1"], cv2.IMREAD_GRAYSCALE)
    img_h, img_w = images[0].shape

    mask_rect = conf["mask_rect"]
    is_ok = True
    for i, img in enumerate(images):
        upper_img = img[: img_h // 2]
        binary = get_binary(upper_img, mask_rect, 180)
        masked_img = cv2.bitwise_and(upper_img, upper_img, mask=binary)

        area_val = cv2.countNonZero(masked_img)

        # 상대적 평균 이용
        rel_dev = area_val - conf["rel_avg"]
        rel_var = rel_dev * rel_dev / conf["rel_avg"]
        rel_std = math.sqrt(rel_var)

        # 절대적 임의값 이용
        dev = area_val - conf["threshold"]
        var = dev * dev / conf["threshold"]
        std = math.sqrt(var)

        result = "정상"
        # 결함 점수
        score = 1
        if dev < -95000:
            score = score + 0.7 * score
        if var > 90000:
            score = score + 0.7 * score
        if area_val < 5000:
            score = score + 0.6 * score
        if std > 300:
            score = score + 0.6
        if rel_var < 4000:
            score = score + 0.3 * score
        if rel_dev > -3000:
            score = score + 0.2 * score
        if rel_std < 60:
            score = score + 0.2 * score

        if score > 5:
            result = "약포 변색"
            bomb.defect["powder"]["bot"]["res"][i].append(
                DEFECT_CODE["powder"]["bot"]["discolor"]
            )
            is_ok = False
        logmg.i.log("%d 결함 점수 : %f 결과 : %s", i, score, result)

    bomb.update_infer_stat("powder", "discolor", is_ok)
    return is_ok


def infer_34(bomb):
    """
    추진 장약 이물질
    """
    logmg.i.log("# 추진장약 이물질")
    conf = powder_conf.infer_34

    images = load_images(bomb.img_path["CAM1"])
    img_h, img_w = images[0].shape[:2]

    mask_rect = conf["mask_rect"]
    fo_detect_area = conf["fo_detect_area"]
    is_ok = True
    for i, img in enumerate(images):
        upper_img = img[: img_h // 2]

        mask = np.zeros_like(upper_img)
        mask = cv2.rectangle(
            mask,
            mask_rect["top_left"],
            mask_rect["bottom_right"],
            color=(255, 255, 255),
            thickness=cv2.FILLED,
        )
        masked_img = cv2.bitwise_and(upper_img, mask)
        _, thresh_img = cv2.threshold(masked_img, 180, 255, cv2.THRESH_BINARY)

        thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(
            thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 감지된 윤곽선을 순회 하며 필터링 및 표시
        brightnesses = []
        means = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if fo_detect_area["min"] < area < fo_detect_area["max"]:
                x, y, w, h = cv2.boundingRect(contour)
                if abs(w - h) < 3:
                    for point in contour:
                        x, y = point[0]
                        brightness = img[y, x]
                        brightnesses.append(brightness)
                    mean_brightness = np.mean(brightnesses)  # 감지된 영역 내부의 밝기값 평균 계산
                    max_brightness = np.max(brightnesses)

                    if mean_brightness > 191.7 and max_brightness < 206 and w * h > 60:
                        means.append(mean_brightness)
        result = "정상"
        if len(means) != 0:
            result = "이물질 감지"
            bomb.defect["powder"]["bot"]["res"][i].append(
                DEFECT_CODE["powder"]["bot"]["fo"]
            )
            is_ok = False
        logmg.i.log("%d 결함 : %s %s", i, result, means)

    bomb.update_infer_stat("powder", "fo", is_ok)
    return is_ok


def do_infer(bomb):
    infer_30(bomb)
    infer_31(bomb)
    infer_32(bomb)
    infer_33(bomb)
    infer_34(bomb)

    # return True
