import math

from conf import *
from util import *

import infer.Wing.wing_conf as wing_conf
import logs.logmg as logmg


def infer_26(bomb):
    """
    날개 굴곡 및 파손
    """
    logmg.i.log("# 날개 굴곡 및 파손")
    conf = wing_conf.infer_26
    images = load_images(bomb.img_path['CAM1'])

    mask_rect = conf['mask_rect']
    is_ok = True
    for i, img in enumerate(images):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_img, 50, 200, apertureSize=3)
        mask = np.zeros_like(edges)
        cv2.rectangle(mask, mask_rect['top_left'], mask_rect['bottom_right'],
                      (255, 255, 255), thickness=cv2.FILLED)
        masked_edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLines(masked_edges, 1, np.pi / 500, threshold=160)

        if lines is None:
            lines = np.array([])

        line_cnt = lines.size // 2
        leans = []
        x_at_y_530s = []
        x_at_y_960s = []
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

            x_at_y_530 = (rho - 530 * b) / a
            x_at_y_960 = (rho - 960 * b) / a

            logmg.i.log("x_at_y_530 : %s", x_at_y_530)
            logmg.i.log("x_at_y_960 : %s", x_at_y_960)

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            angle_deg = math.degrees(theta)

            leans.append(angle_deg)
            x_at_y_530s.append(x_at_y_530)
            x_at_y_960s.append(x_at_y_960)

        bad_line_cnt = 0
        bad_perc = 0
        result_type = ["결과 : 굴곡 및 파손 ", "결과 : 정상 "]
        result = ""
        if line_cnt == 0:
            result = result_type[0]
            bomb.defect['wing']['res'][i].append(DEFECT_CODE['wing']['bent'])
            is_ok = False
            continue

        for j, lean in enumerate(leans):
            result = result_type[1]
            x530 = x_at_y_530s[j]
            x960 = x_at_y_960s[j]
            if  not(177 < lean <= 180 or 0 <= lean <= 0.5) and not(620 < x530 < 660 and 620 < x960 < 660):
                bad_line_cnt += 1

        if bad_line_cnt != 0:
            bad_perc = bad_line_cnt / line_cnt * 100

        if bad_perc > 25:
            result = result_type[0]
            bomb.defect['wing']['res'][i].append(DEFECT_CODE['wing']['bent'])
            is_ok = False
        logmg.i.log("감지 직선 : %d개 %s 기울기 : %r", line_cnt, result, leans)
        logmg.i.log("bad_perc : %s", bad_perc)

    bomb.update_infer_stat('wing', 'bent', is_ok)
    return is_ok


def infer_27(bomb):
    """
    날개 파손 및 절단
    """
    logmg.i.log("# 날개 파손 및 절단")
    conf = wing_conf.infer_27

    images = load_images(bomb.img_path['CAM1'], cv2.IMREAD_GRAYSCALE)
    img_h, img_w = images[0].shape[:2]

    template_img = cv2.imread(conf['template_path'], cv2.IMREAD_GRAYSCALE)

    mask_rect = conf['mask_rect']
    is_ok = True
    result = ""
    for i, img in enumerate(images):
        lower_img = img[img_h // 2:]
        binary = get_binary(lower_img, mask_rect, 30)

        res = cv2.matchTemplate(binary, template_img, cv2.TM_CCOEFF_NORMED)
        result = "정상"
        if res <= conf['infer_threshold']:
            bomb.defect['wing']['res'][i].append(DEFECT_CODE['wing']['damage'])
            result = "파손, 절단 의심"
            is_ok = False
        logmg.i.log("%d 마스크 유사도 : %f 결과 : %s", i, res, result)

    bomb.update_infer_stat('wing', 'damage', is_ok)
    return is_ok


def infer_28(bomb):
    """
    날개 부식
    """
    logmg.i.log("# 날개 부식")
    conf = wing_conf.infer_28

    images = load_images(bomb.img_path['CAM1'], cv2.IMREAD_GRAYSCALE)
    img_h, img_w = images[0].shape

    threshold = conf['infer_threshold']
    is_ok = True

    for i, img in enumerate(images):
        quantized_image =  cv2.convertScaleAbs((img // 85) * 85)
        lower_img = quantized_image[img_h // 2:]

        max_sum = np.sum(lower_img >= np.max(lower_img))

        result = '정상'
        if max_sum < threshold:
            result = '부식'
            bomb.defect['wing']['res'][i].append(DEFECT_CODE['wing']['corr'])
            is_ok = False
        
        logmg.i.log('%d max_sum : %f 기준 : 6000 결과 : %s', i, max_sum, result)
    bomb.update_infer_stat('wing', 'corr', is_ok)
    return is_ok

def do_infer(bomb):
    infer_26(bomb)
    infer_27(bomb)
    infer_28(bomb)