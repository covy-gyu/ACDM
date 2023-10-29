# from PIL import Image
import cv2, copy
from ultralytics import YOLO

# from data import logmg
from conf import *
import numpy as np
from pathlib import Path
import logs.logmg as logmg


def get_cls_pos(res_CAM, target):
    positions = []
    for i, sublist in enumerate(res_CAM["cls"]):
        for j, subelement in enumerate(sublist):
            if subelement == target:
                positions.append((i, j))
    return positions


# bomb.defect['powder']['bot']['res'][i].append(DEFECT_CODE['powder']['bot']['exist'])
def add_defects(code, cls, bomb_defect):
    for pos in cls:
        bomb_defect[pos[0]].append(code)


def infer_2(bomb, res_CAM4):
    logmg.i.log("# 신관 안전핀 불완전 결합")
    is_ok = True
    a1 = get_cls_pos(res_CAM4, "a1")
    a2 = get_cls_pos(res_CAM4, "a2")
    # if a1 is None and a2 is None:
    #     return
    if len(a1) > 0 and len(a2) == 0:
        # logmg.i.log("2 : 신관안전핀불완전결합")
        # 신관안전핀불완전결합 추가 - 결함리스트의 위치에 맞는 곳에 추가
        # bomb.defect['anchor']['bot']['res'][6].append(DEFECT_CODE['anchor']['exist'])
        add_defects(DEFECT_CODE["head"]["pin_exist"], a2, bomb.defect["head"]["res"])
        is_ok = False
    logmg.i.log("a1 : %s", a1)
    logmg.i.log("a2 : %s", a2)

    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")
    bomb.update_infer_stat("head", "pin_attach", is_ok)
    return is_ok


def infer_3(bomb, res_CAM4):
    logmg.i.log("# 신관 신관 구성품 손상 및 망실")
    is_ok = True
    b1 = get_cls_pos(res_CAM4, "b1")
    b2 = get_cls_pos(res_CAM4, "b2")
    b5 = get_cls_pos(res_CAM4, "b5")

    if len(b1) == 0:
        # 신관정상결함코드 추가 - 결함리스트의 위치에 맞는 곳에 추가
        add_defects(DEFECT_CODE["head"]["comp"], b1, bomb.defect["head"]["res"])
        is_ok = False
    if len(b2) == 0:
        # 신관정상결함코드 추가 - 결함리스트의 위치에 맞는 곳에 추가
        add_defects(DEFECT_CODE["head"]["comp"], b2, bomb.defect["head"]["res"])
        is_ok = False
    if len(b5) > 0:
        # 신관정상결함코드 추가 - 결함리스트의 위치에 맞는 곳에 추가
        add_defects(DEFECT_CODE["head"]["comp"], b5, bomb.defect["head"]["res"])
        is_ok = False
    logmg.i.log("b1 : %s", b1)
    logmg.i.log("b2 : %s", b2)
    logmg.i.log("b5 : %s", b5)

    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")

    bomb.update_infer_stat("head", "comp", is_ok)
    return is_ok


def infer_4(bomb, res_CAM4):
    logmg.i.log("# 신관 파손")
    is_ok = True
    c1 = get_cls_pos(res_CAM4, "c1")
    if len(c1) > 0:
        # 파손 추가 - 결함리스트의 위치에 맞는 곳에 추가
        add_defects(DEFECT_CODE["head"]["damage"], c1, bomb.defect["head"]["res"])
        is_ok = False
        pass
    logmg.i.log("c1 : %s", c1)
    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")
    bomb.update_infer_stat("head", "damage", is_ok)
    return is_ok


def infer_5(bomb, res_CAM4):
    logmg.i.log("# 신관 안전핀 망실 또는 절단")
    is_ok = True
    a1 = get_cls_pos(res_CAM4, "a1")
    a2 = get_cls_pos(res_CAM4, "a2")
    a4 = get_cls_pos(res_CAM4, "a4")
    if len(a2) == 0 and (len(a1) > 0 or len(a4) > 0):
        # 안전핀망실또는절단 추가 - 결함리스트의 위치에 맞는 곳에 추가
        add_defects(DEFECT_CODE["head"]["pin_exist"], a2, bomb.defect["head"]["res"])
        is_ok = False
        pass
    logmg.i.log("a1 : %s", a1)
    logmg.i.log("a2 : %s", a2)
    logmg.i.log("a4 : %s", a4)
    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")
    bomb.update_infer_stat("head", "pin_exist", is_ok)
    return is_ok


# def infer_6_8(bomb, res_CAM4):
#     logmg.i.log("# 신관 부식")
#     is_ok1 = True
#     is_ok2 = True

#     c2 = get_cls_pos(res_CAM4, "c2")
#     c2p = get_cls_pos(res_CAM4, "c2+")
#     if len(c2) > 6:
#         is_ok1 = False
#         add_defects(DEFECT_CODE["head"]["light_corr"], c2, bomb.defect["head"]["res"])

#     if len(c2p) > 0:
#         add_defects(DEFECT_CODE["head"]["heavy_corr"], c2p, bomb.defect["head"]["res"])
#         is_ok2 = False

#     bomb.update_infer_stat("head", "light_corr", is_ok1)
#     bomb.update_infer_stat("head", "heavy_corr", is_ok2)

#     logmg.i.log("c2 : %s", c2)
#     logmg.i.log("c2p : %s", c2p)

#     logmg.i.log("정상 여부")
#     logmg.i.log("소부식 : %s", is_ok1)
#     logmg.i.log("중부식 : %s", is_ok2)

#     return is_ok1


def infer_6(bomb, res_CAM4):
    logmg.i.log("# 신관 소부식")
    is_ok = True

    c2 = get_cls_pos(res_CAM4, "c2")

    if len(c2) > 0:
        is_ok = False
        add_defects(DEFECT_CODE["head"]["light_corr"], c2, bomb.defect["head"]["res"])

    bomb.update_infer_stat("head", "light_corr", is_ok)

    logmg.i.log("c2 : %s", c2)

    logmg.i.log("정상 여부 : %s", is_ok)

    return is_ok


def infer_8(bomb, res_CAM4):
    logmg.i.log("# 신관 중부식")
    is_ok = True

    c2p = get_cls_pos(res_CAM4, "c2+")

    if len(c2p):
        is_ok = False
        add_defects(DEFECT_CODE["head"]["heavy_corr"], c2p, bomb.defect["head"]["res"])

    bomb.update_infer_stat("head", "heavy_corr", is_ok)

    logmg.i.log("c2p : %s", c2p)

    logmg.i.log("정상 여부 : %s", is_ok)

    return is_ok


def infer_9(bomb, res_CAM4):
    logmg.i.log("# 안전핀구부러짐")
    is_ok = True
    a3 = get_cls_pos(res_CAM4, "a3")
    if len(a3) > 0:
        # 안전핀구부러짐 추가 - 결함리스트의 위치에 맞는 곳에 추가
        add_defects(DEFECT_CODE["head"]["pin_bent"], a3, bomb.defect["head"]["res"])
        is_ok = False
        pass
    logmg.i.log("a3 : %s", a3)
    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")
    bomb.update_infer_stat("head", "pin_bent", is_ok)
    return is_ok


def infer_10(bomb, res_CAM4):
    logmg.i.log("# 안전핀 소부식")
    is_ok = True
    a5 = get_cls_pos(res_CAM4, "a5")
    if len(a5) > 0:
        # 안전핀 소부식 추가 - 결함리스트의 위치에 맞는 곳에 추가
        add_defects(DEFECT_CODE["head"]["pin_corr"], a5, bomb.defect["head"]["res"])
        is_ok = False
        pass
    logmg.i.log("a5 : %s", a5)
    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")
    bomb.update_infer_stat("head", "pin_corr", is_ok)
    return is_ok


def infer_14(bomb, res_CAM4):
    logmg.i.log("# 탄체 적색 경고 표지 망실 또는 식별불가")
    is_ok = True
    d2 = get_cls_pos(res_CAM4, "d2")
    if len(d2) == 0:
        # 적색경고표지 망실 추가 - 결함리스트의 위치에 맞는 곳에 추가, 식별불가랑 합집합해야함
        is_ok = False
        bomb.defect["body"]["top"]["res"][6].append(DEFECT_CODE["body"]["top"]["warn"])
        pass
    logmg.i.log("d2 : %s", d2)
    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")
    bomb.update_infer_stat("body", "warn", is_ok)
    return is_ok


def infer_15(bomb, res_CAM4):
    logmg.i.log("# 탄두 균열 또는 뒤틀림")
    is_ok = True
    d1 = get_cls_pos(res_CAM4, "d1")
    d8 = get_cls_pos(res_CAM4, "d8")
    res = d1 + d8
    if len(res) > 0:
        # 탄두균열 또는 뒤틀림 추가 - 결함리스트의 위치에 맞는 곳에 추가
        add_defects(
            DEFECT_CODE["body"]["bot"]["crack_shift"],
            res,
            bomb.defect["body"]["top"]["res"],
        )
        is_ok = False
    logmg.i.log("d1 : %s", d1)
    logmg.i.log("d8 : %s", d8)

    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")

    bomb.update_infer_stat("body", "crack_shift", is_ok)
    return is_ok


def infer_16(bomb, res_CAM3):
    logmg.i.log("# 폐쇄링 망실")
    is_ok = True
    d3 = get_cls_pos(res_CAM3, "d3")
    if len(d3) == 0:
        # 폐쇄링 망실 추가 - 결함리스트의 위치에 맞는 곳에 추가
        add_defects(
            DEFECT_CODE["body"]["bot"]["ring_exist"],
            d3,
            bomb.defect["body"]["bot"]["res"],
        )
        is_ok = False
        pass
    logmg.i.log("d3 : %s", d3)
    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")
    bomb.update_infer_stat("body", "ring_exist", is_ok)
    return is_ok


def infer_18(bomb, res_CAM3):
    logmg.i.log("# 링게이지 불합격")
    is_ok = True
    d3 = get_cls_pos(res_CAM3, "d3")
    cam = bomb.img_path["CAM3"]
    masks = res_CAM3["masks"]
    imgs = []
    for i in range(6):
        imgs.append(cv2.imread(cam[i]))
    w_arr = []
    if len(d3) != 0:
        for pos in d3:
            row, col = pos
            mask = masks[row][col]
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for cnt in contours:
                _, _, w, _ = cv2.boundingRect(cnt)
                if 500 < w < 550:
                    w_arr.append(w)
    logmg.i.log("d3 : %s", d3)
    logmg.i.log("w_arr : %s", w_arr)

    if len(w_arr) > 0:
        is_ok = True
    else:
        is_ok = False

    bomb.update_infer_stat("body", "gauge", is_ok)

    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")

    # for image_path in cam:
    #     item = {'filename':str, 'w': 0}
    #     img = cv2.imread(image_path)
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     _, thresh = cv2.threshold(img_gray,20,255,0)
    #     contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #     for cnt in contours:
    #         _, _, w, _ = cv2.boundingRect(cnt)
    #         if w > 500:
    #             # img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    #             w_arr.append(w)
    # avg = np.average(w_arr)
    # result = "링게이지 불합격"
    # # if 515 < avg < 530:
    # if 511 < avg < 535:
    #     result = "정상"
    # else:
    #     is_ok = False
    #     bomb.defect['body']['bot']['res'][6].append(DEFECT_CODE['body']['bot']['gauge'])
    # logmg.i.log("링게이지 수치: %s 결과: %s", avg, result)
    # bomb.update_infer_stat('body', 'gauge', is_ok)
    return is_ok


# def infer_17_24(bomb, res_CAM3):
#     logmg.i.log("# 탄체 부식")
#     is_ok1 = True
#     is_ok2 = True
#     d4 = get_cls_pos(res_CAM3, "d4")
#     d4p = get_cls_pos(res_CAM3, "d4+")
#     logmg.i.log("d4 : %s", d4)
#     logmg.i.log("d4p : %s", d4p)

#     if len(d4) > 0:
#         # 이미지 픽셀 개수 이상 확인
#         # 탄체 중부식추가 - 결함리스트의 위치에 맞는 곳에 추가
#         add_defects(
#             DEFECT_CODE["body"]["bot"]["light_corr"],
#             d4,
#             bomb.defect["body"]["bot"]["res"],
#         )
#         is_ok1 = False

#     if len(d4p) > 0:
#         add_defects(
#             DEFECT_CODE["body"]["bot"]["heavy_corr"],
#             d4p,
#             bomb.defect["body"]["bot"]["res"],
#         )
#         is_ok2 = False

#     bomb.update_infer_stat("body", "light_corr", is_ok1)
#     bomb.update_infer_stat("body", "heavy_corr", is_ok2)

#     logmg.i.log("정상 여부")
#     logmg.i.log("소부식 : %s", is_ok1)
#     logmg.i.log("중부식 : %s", is_ok2)

#     return is_ok1


def infer_17(bomb, res_CAM3):
    logmg.i.log("# 탄체 중부식")
    is_ok = True
    d4p = get_cls_pos(res_CAM3, "d4+")
    logmg.i.log("d4p : %s", d4p)

    if len(d4p) > 0:
        add_defects(
            DEFECT_CODE["body"]["bot"]["heavy_corr"],
            d4p,
            bomb.defect["body"]["bot"]["res"],
        )
        is_ok = False

    bomb.update_infer_stat("body", "heavy_corr", is_ok)

    logmg.i.log("정상 여부: %s", is_ok)

    return is_ok


def infer_24(bomb, res_CAM3):
    logmg.i.log("# 탄체 소부식")
    is_ok = True
    d4 = get_cls_pos(res_CAM3, "d4")
    logmg.i.log("d4 : %s", d4)

    # img_idx = {}
    # for row, col in d4:
    #     if row not in img_idx:
    #         img_idx[row] = []
    #     img_idx[row].append(col)
    # logmg.i.log("img_idx : %s", img_idx)
    # logmg.i.log("res_CAM3[masks] : %s", res_CAM3["masks"])

    # # 각 이미지 인덱스에 대한 데이터 접근
    # for row in img_idx:
    #     final_mask = None
    #     for col in img_idx[row]:
    #         logmg.i.log(col)
    #         mask = res_CAM3["masks"][row][col]
    #         if np.any(mask):
    #             if final_mask is None:
    #                 final_mask = mask
    #             else:
    #                 final_mask = cv2.bitwise_and(final_mask, mask)
    #     if final_mask is not None:  # 수정된 부분
    #         img_idx[row] = cv2.countNonZero(final_mask)  # 수정된 부분

    # defect_idx = []
    # for i in range(6):
    #     if i in img_idx:
    #         mask_cnt = img_idx[i]
    #         mask_perc = mask_cnt / 288000 * 100
    #         if mask_perc >= 30:
    #             defect_idx.append((i, mask_perc))

    # if defect_idx:
    #     is_ok = False
    #     logmg.i.log("defect_idx : %s", defect_idx)

    #     add_defects(
    #         DEFECT_CODE["body"]["bot"]["light_corr"],
    #         defect_idx,
    #         bomb.defect["body"]["bot"]["res"],
    #     )

    if len(d4) > 0:
        for pos in d4:
            row, col = pos
            mask = res_CAM3["masks"][row][col]
            cv2.imwrite(f"data/d4/{bomb.lot.name}_{bomb.num}.png", mask)

            # 이미지 픽셀 개수 이상 확인
            # 탄체 중부식추가 - 결함리스트의 위치에 맞는 곳에 추가
        add_defects(
            DEFECT_CODE["body"]["bot"]["light_corr"],
            d4,
            bomb.defect["body"]["bot"]["res"],
        )
        is_ok = False

    bomb.update_infer_stat("body", "light_corr", is_ok)

    logmg.i.log("정상 여부: %s", is_ok)

    return is_ok


def infer_20(bomb, res_CAM4):
    logmg.i.log("# 충전물 누출 및 흔적")
    is_ok = True
    d5 = get_cls_pos(res_CAM4, "d5")
    if len(d5) > 0:
        # 충전물 누출 및 흔적 추가 - 결함리스트의 위치에 맞는 곳에 추가
        add_defects(
            DEFECT_CODE["body"]["top"]["filling_leak"],
            d5,
            bomb.defect["body"]["top"]["res"],
        )
        is_ok = False
        pass
    logmg.i.log("d5 : %s", d5)
    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")
    bomb.update_infer_stat("body", "filling_leak", is_ok)
    return is_ok


def infer_21(bomb, res_CAM4):
    logmg.i.log("# 폐쇄링 손상")
    is_ok = True
    d7 = get_cls_pos(res_CAM4, "d7")
    if len(d7) > 0:
        # 폐쇄링 손상 추가 - 결함리스트의 위치에 맞는 곳에 추가
        add_defects(
            DEFECT_CODE["body"]["bot"]["ring_damage"],
            d7,
            bomb.defect["body"]["bot"]["res"],
        )
        is_ok = False
        pass
    logmg.i.log("d7 : %s", d7)
    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")
    bomb.update_infer_stat("body", "ring_damage", is_ok)
    return is_ok


def infer_23(bomb, res_CAM4):
    logmg.i.log("# 탄체 이물질")
    is_ok = True
    d6 = get_cls_pos(res_CAM4, "d6")
    if len(d6) > 0:
        # 이물질 추가 - 결함리스트의 위치에 맞는 곳에 추가
        is_ok = False
        add_defects(
            DEFECT_CODE["body"]["bot"]["fo"],
            d6,
            bomb.defect["body"]["bot"]["res"],
        )
        # bomb.defect["body"]["bot"]["res"][6].append(DEFECT_CODE["body"]["bot"]["fo"])

    logmg.i.log("d6 : %s", d6)
    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")
    bomb.update_infer_stat("body", "fo", is_ok)
    return is_ok


# def infer_34(bomb, res_CAM1, res_CAM2):
#     """
#     추진 장약 이물질
#     """
#     logmg.i.log("# 추진장약 이물질")
#     is_ok = True

#     wpf = get_cls_pos(res_CAM1, "pf")
#     bpf = get_cls_pos(res_CAM2, "pf")

#     logmg.i.log("wpf : %s", wpf)
#     logmg.i.log("bpf : %s", bpf)

#     def is_defect(cls, position):
#         if len(cls) > 1:
#             for i in range(6):
#                 cnt = get_counter(cls).get(i, 0)
#                 if cnt == 0:
#                     add_defects(DEFECT_CODE['powder']['bot']['fo'], cls, bomb.defect['powder'][position]['res'])
#             return True
#         return False

#     res_wpf = is_defect(wpf,'bot')
#     res_bpf = is_defect(bpf, 'top')

#     if res_wpf or res_bpf == True:
#         logmg.i.log("results : wpf : %s, bpf : %s", res_wpf, res_bpf)
#         is_ok = False

#     if is_ok:
#         logmg.i.log("정상")
#     else:
#         logmg.i.log("결함")

#     bomb.update_infer_stat('powder', 'fo', is_ok)
#     return is_ok


from collections import Counter


def get_counter(cls_list):
    elements = [x[0] for x in cls_list]
    cnt = Counter(elements)

    return cnt


def infer_36(bomb, res_CAM1, res_CAM2):
    logmg.i.log("# 추진약멈치 망실")
    is_ok = True

    wc = get_cls_pos(res_CAM1, "wc")
    bc = get_cls_pos(res_CAM2, "bc")

    logmg.i.log("wc : %s", wc)
    logmg.i.log("bc : %s", bc)

    # 각 멈치 개수가 2개 이하인 경우
    def is_defect(cls, position):
        if len(cls) <= 2:
            for i in range(6):
                cnt = get_counter(cls).get(i, 0)
                if cnt == 0:
                    add_defects(
                        DEFECT_CODE["anchor"]["exist"],
                        cls,
                        bomb.defect["anchor"][position]["res"],
                    )
            return True
        return False

    res_wc = is_defect(wc, "bot")
    res_bc = is_defect(bc, "top")

    if res_wc or res_bc == True:
        logmg.i.log("results : wc : %s, bc : %s", res_wc, res_bc)
        is_ok = False

    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")

    bomb.update_infer_stat("anchor", "exist", is_ok)
    return is_ok

    # # 1: cam1, 2: cam2
    # cls1 = res_CAM1["cls"]
    # cls2 = res_CAM2["cls"]
    # # print(cls1)
    # # print(cls2)
    # sum1, sum2 = 0, 0
    # for i in range(6):
    #     sum1 += len(cls1[i])
    #     sum2 += len(cls2[i])
    # if sum1 <= 4:
    #     is_ok = False
    #     bomb.defect['anchor']['bot']['res'][6].append(DEFECT_CODE['anchor']['exist'])
    #     logmg.i.log("날개 멈치 결함")
    # if sum2 <= 11:
    #     is_ok = False
    #     bomb.defect['anchor']['top']['res'][6].append(DEFECT_CODE['anchor']['exist'])
    #     logmg.i.log("탄체 멈치 결함")
    # logmg.i.log("정상")

    # bomb.update_infer_stat('anchor', 'exist', is_ok)
    # return is_ok


def get_mask_img(path, mask):
    # plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    # plt.show()
    # cv2.imshow("img", mask)
    # cv2.waitKey(0)
    pass


def infer_37(bomb, res_CAM1, res_CAM2):
    logmg.i.log("# 추진약멈치 부식")
    is_ok = True

    wc = get_cls_pos(res_CAM1, "wc")
    bc = get_cls_pos(res_CAM2, "bc")

    logmg.i.log("wc : %s", wc)
    logmg.i.log("bc : %s", bc)

    def is_defect(target, cam):
        pixel_max_list = []
        pixel_avg_list = []
        pixel_frq_list = []

        if len(target) > 0:
            for pos in target:
                row, col = pos
                mask = cam["masks"][row][col]
                # logmg.i.log(mask)
                # # 현재 시간을 가져오기
                # now = datetime.now()

                # # 문자열로 변환하기
                # current_time = now.strftime("%Y-%m-%d%H:%M:%S.%f")

                # quantized_mask = cv2.convertScaleAbs((mask // 50) * 50)
                # max_sum = np.sum(quantized_mask >= np.max(quantized_mask))
                # unique_values, counts = np.unique(
                #     quantized_mask[quantized_mask > 0], return_counts=True
                # )
                # logmg.i.log("pos: %s", pos)
                # logmg.i.log("max: %s max_sum: %s", np.max(quantized_mask), max_sum)
                # logmg.i.log("unique_values: %s counts: %s", unique_values, counts)
                # logmg.i.log("most_frequent_val: %s", most_frequent_val)
                # cv2.imshow("mask", mask)
                # cv2.waitKey(0)
                if np.any(mask):
                    maxval = np.max(mask[mask > 0])
                    pixel_max_list.append(maxval)
                    avgval = np.mean(mask[mask > 0])
                    pixel_avg_list.append(avgval)

                    unique_values, counts = np.unique(
                        mask[mask > 0], return_counts=True
                    )
                    most_frequent_val = unique_values[np.argmax(counts)]
                    pixel_frq_list.append(most_frequent_val)
                    logmg.i.log(
                        "pos: %s maxval: %s avgval: %.2f, most_frequent_val: %s",
                        pos,
                        maxval,
                        avgval,
                        most_frequent_val,
                    )
                    # # logmg.i.log("unique_values: %s", unique_values)
                    # logmg.i.log("most_frequent_val: %s", most_frequent_val)
        avg_mean = np.mean(pixel_avg_list)
        max_mean = np.mean(pixel_max_list)
        frq_mean = np.mean(pixel_frq_list)
        logmg.i.log(
            "max_mean: %s avg_mean: %s frq_mean: %s", max_mean, avg_mean, frq_mean
        )
        if max_mean > 120:
            return False
        else:
            if avg_mean < 55 and frq_mean < 50:
                logmg.i.log("부식============================================")
                return True

    logmg.i.log("wc:")
    res_wc = is_defect(wc, res_CAM1)
    logmg.i.log("bc:")
    res_bc = is_defect(bc, res_CAM2)

    if res_wc or res_bc == True:
        logmg.i.log("results : wc : %s, bc : %s", res_wc, res_bc)
        is_ok = False
        if res_wc:
            # wc인 결함 추가
            bomb.defect["anchor"]["bot"]["res"][6].append(
                DEFECT_CODE["anchor"]["heavy_corr"]
            )
            logmg.i.log("날개 멈치 부식")
        if res_bc:
            # bc인 결함 추가
            bomb.defect["anchor"]["top"]["res"][6].append(
                DEFECT_CODE["anchor"]["heavy_corr"]
            )
            logmg.i.log("탄체 멈치 부식")

    if is_ok:
        logmg.i.log("정상")
    else:
        logmg.i.log("결함")

    bomb.update_infer_stat("anchor", "heavy_corr", is_ok)
    cv2.destroyAllWindows()
    return is_ok

    # mask_info = {
    #     "max": [[], [], [], [], [], []],
    #     "avg": [[], [], [], [], [], []],
    #     "fqt": [[], [], [], [], [], []],
    # }

    # def check_mask(mask):
    #     # cv2.imshow("mask",mask)
    #     # cv2.waitKey(0)
    #     maxval = np.max(mask[mask > 0])
    #     avgval = np.average(mask[mask > 0])

    #     unique_values, counts = np.unique(mask[mask > 0], return_counts=True)
    #     # logmg.i.log("check_mask unique_val : %s", unique_values)
    #     # logmg.i.log("check_mask counts : %s", counts)
    #     most_frequent_val = unique_values[np.argmax(counts)]
    #     return maxval, avgval, most_frequent_val

    # def check_defect(idx, info_mask, mask):

    #     maxval, avgval, most_frequent_val = check_mask(mask)
    #     info_mask["max"][idx].append(maxval)
    #     info_mask["avg"][idx].append(avgval)
    #     info_mask["fqt"][idx].append(most_frequent_val)
    #     # logmg.i.log("maxval : %s, avgval : %s, most_frequent_val : %s", maxval, avgval, most_frequent_val)
    #     if most_frequent_val < 20:
    #         return True
    #     else:
    #         return False

    # mask1 = res_CAM1["masks"]
    # mask2 = res_CAM2["masks"]
    # # print(mask1)
    # # print(mask2)
    # info_mask1 = copy.deepcopy(mask_info)
    # info_mask2 = copy.deepcopy(mask_info)

    # corr_cnt = 0
    # corr_cnt2 = 0
    # for i in range(6):
    #     lm1 = len(mask1[i])
    #     lm2 = len(mask2[i])
    #     if lm1 > 0:
    #         for j in range(lm1):
    #             logmg.i.log("날개 멈치 : ")
    #             if check_defect(i, info_mask1, mask1[i][j]):
    #                 corr_cnt += 1
    #                 bomb.defect['anchor']['bot']['res'][i].append(DEFECT_CODE['anchor']['heavy_corr'])
    #                 logmg.i.log("날개 멈치 %d 부식",i)
    #     if lm2 > 0:
    #         for k in range(lm2):
    #             logmg.i.log("탄체 멈치 : ")
    #             if check_defect(i, info_mask2, mask2[i][k]):
    #                 corr_cnt2 += 1
    #                 bomb.defect['anchor']['top']['res'][i].append(DEFECT_CODE['anchor']['heavy_corr'])
    #                 logmg.i.log("탄체 멈치 %d 부식",i)

    # if not(corr_cnt < 3 and corr_cnt2 < 3):
    #     is_ok = False

    # bomb.update_infer_stat('anchor', 'heavy_corr', is_ok)
    # return is_ok
    # print(info_mask1)
    # print(info_mask2)


def get_mask(mask, org_img):
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    mask_img = np.zeros((480, 640), dtype=np.uint8)
    mask = mask.data.cpu().numpy().astype("uint8")
    mask[mask > 0] = 255
    mask_img = cv2.bitwise_and(mask_img, mask_img, mask=mask)
    mask_img = cv2.resize(mask, (1280, 960))
    # cv2.imshow("r", mask)
    # cv2.waitKey(0)
    mask_img = cv2.bitwise_and(org_img, mask_img)
    return mask_img


class YOLOInfer:
    def __init__(self) -> None:
        self.model1 = YOLO("data/model/anchor.pt")
        self.model2 = YOLO("data/model/head.pt")

    def head_infer(self, bomb):
        res_CAM3 = self.detecing(bomb.img_path["CAM3"], self.model2)
        res_CAM4 = self.detecing(bomb.img_path["CAM4"], self.model2)

        infer_2(bomb, res_CAM4)
        infer_3(bomb, res_CAM4)
        infer_4(bomb, res_CAM4)
        infer_5(bomb, res_CAM4)
        # infer_6_8(bomb, res_CAM4)
        infer_6(bomb, res_CAM4)
        infer_8(bomb, res_CAM4)
        infer_9(bomb, res_CAM4)
        infer_10(bomb, res_CAM4)
        infer_14(bomb, res_CAM4)
        infer_15(bomb, res_CAM4)
        infer_16(bomb, res_CAM3)
        infer_18(bomb, res_CAM3)
        infer_20(bomb, res_CAM4)
        infer_21(bomb, res_CAM3)
        infer_23(bomb, res_CAM3)
        # infer_17_24(bomb, res_CAM3)
        infer_17(bomb, res_CAM3)
        infer_24(bomb, res_CAM3)

        return True

    def anchor_infer(self, bomb):
        res_CAM1 = self.detecing(bomb.img_path["CAM1"], self.model1)
        res_CAM2 = self.detecing(bomb.img_path["CAM2"], self.model1)

        # infer_34(bomb, res_CAM1, res_CAM2)
        infer_36(bomb, res_CAM1, res_CAM2)
        infer_37(bomb, res_CAM1, res_CAM2)

        return True

    def detecing(self, cam, model, savemode=False):
        res_info = {"cls": [[], [], [], [], [], []], "masks": [[], [], [], [], [], []]}

        def get_mask_and_area(mask, org_img):
            get_mask(mask, org_img)
            area = np.argwhere(mask_img > 0)
            # cv2.imshow("r",mask_img)
            # cv2.waitKey(0)
            return mask_img, len(area)

        def get_imgval(img):
            return len(np.argwhere(img > 0))

        imgs = []
        res = copy.deepcopy(res_info)
        for i in range(6):
            imgs.append(cv2.imread(cam[i]))
        results = model.predict(imgs, verbose=False, save=savemode)
        for i in range(6):
            r = results[i]
            class_ids = np.array(r.boxes.cls.cpu()).astype(int)
            # logmg.i.log("detecting cls len : %s", len(class_ids))

            for id in class_ids:
                name = r.names[id]
                res["cls"][i].append(name)
            masks = r.masks
            # idx = len(res["cls"][i])
            # logmg.i.log("cls idx: %s", len(res["cls"][i]))
            if masks is not None:
                # logmg.i.log("detecting masks len : %s", len(masks))
                mask_img = np.zeros_like(imgs[i], dtype=np.uint8)
                for mask in masks:
                    mask_img = get_mask(mask, imgs[i])
                    res["masks"][i].append(mask_img)

                # if idx >= 2:  # 마스크 중복 제거
                #     for j in range(idx - 1):
                #         cls_cnt = sum(sublist.count(res['cls'][i]) for sublist in res['cls'])
                #         if res['cls'][j] == res['cls'][j+i]:
                #             logmg.i.log("마스크 중복 제거")

                #             mask_img1, area1 = get_mask_and_area(masks[j], imgs[i])
                #             mask_img2, area2 = get_mask_and_area(masks[j + 1], imgs[i])
                #             sum_area = area1 + area2
                #             union_area = get_imgval(cv2.bitwise_or(mask_img1, mask_img2))
                #             ratio = (sum_area - union_area) / union_area
                #             # print(sum_area,union_area,ratio)
                #             logmg.i.log("cls mask1, mask2: %s, %s", res["cls"][j],res["cls"][j+1])
                #             if ratio >= 0.5:  # 중복처리 범위 기준치
                #                 # print(ratio)
                #                 logmg.i.log("총 cls 개수: %s", len(res["cls"]))
                #                 res["cls"][i].pop()

                #                 res["masks"][i].pop()
        logmg.i.log("cls : %s", res["cls"])

        return res
