import time
import traceback
from PIL import ImageGrab

import TUI
from common import *
from util import *
from conf import *

from file_scan import get_next_bomb
from excel import ExcelWriter
import logs.logmg as logmg
import infer.YOLOS.yolos as yolos
import infer.POCR.pocr as POCR
import infer.POCR.ocr as OCR
import infer.Powder.powder as powder
import infer.Wing.wing as wing
import infer.Laser.laser as laser

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action="ignore")


def get_infer_list(bomb, ocr_obj, yolo_obj):
    return [
        [yolo_obj.head_infer, [bomb], "신관, 탄체 결함 판별중"],
        [OCR.do_infer, [bomb, ocr_obj], "도색표기 결함 판별중"],
        [laser.infer17, [bomb], "변위센서 결함 판별중"],
        [wing.do_infer, [bomb], "날개 결함 판별중"],
        [powder.do_infer, [bomb], "추진장약 결함 판별중"],
        [yolo_obj.anchor_infer, [bomb], "추진약멈치 결함 판별중"],
    ]


def queue_push(queue, data, max_len):
    queue.append(data)
    if len(queue) > max_len:
        queue.pop(0)


def init_infer_obj():
    print("Initializing Model...")
    yolo_obj = yolos.YOLOInfer()

    print("\nInitializing OCR...")
    ocr_obj = POCR.init()

    return ocr_obj, yolo_obj


class Count:
    def __init__(self):
        self.curr_lot_stat = None
        self.cnt = {"lot": {"total": 0, "defect": 0}, "bomb": {"total": 0, "defect": 0}}

    def __getitem__(self, item):
        return self.cnt[item]

    def new_res(self, prev_bomb_lot, curr_lot, is_bomb_ok, is_lot_ok):
        self["bomb"]["total"] += 1  # New bomb
        if prev_bomb_lot != curr_lot:  # New lot
            self["lot"]["total"] += 1
            self.curr_lot_stat = True

        if is_bomb_ok is False:
            self["bomb"]["defect"] += 1

        if self.curr_lot_stat is True and is_lot_ok is False:
            self["lot"]["defect"] += 1
            self.curr_lot_stat = False


def loop(scan_freq=1.0):
    try:
        cls()

        cnt_obj = Count()
        ocr_obj, yolo_obj = init_infer_obj()
        time.sleep(1)
        cls()

        prev_bomb_lot = None
        is_lot_ok = True
        prev_res_q = []
        infer_time_q = []
        ew = None

        stat_flags["start"] = True
        while True:
            # Get next bomb
            # ===========================================================
            bomb = get_next_bomb(prev_bomb_lot, scan_freq=scan_freq)
            if bomb is None:
                break
            st_tm = time.time()

            if prev_bomb_lot != bomb.lot:  # New lot
                is_lot_ok = True
                logmg.i.log("!!!!!!!%s 로트", bomb.lot.name)
                ew = ExcelWriter(f"{bomb.lot.name}.xlsx")

                n_prev_bomb = len(ew.workbook.worksheets)
                if n_prev_bomb != 0:
                    for i in range(n_prev_bomb):
                        bomb.lot.bombs.insert(0, None)
                    bomb.num = n_prev_bomb

            # Do inferences
            # ===========================================================
            cls()
            TUI.print_status(bomb, cnt_obj, prev_res_q, infer_time_q)
            is_bomb_ok = True
            for func, args, stat_str in get_infer_list(bomb, ocr_obj, yolo_obj):
                TUI.print_status_msg(stat_str)
                res = func(*args)

                is_bomb_ok = res if is_bomb_ok is True else False
                is_lot_ok = res if is_lot_ok is True else False

                TUI.print_status(bomb, cnt_obj, prev_res_q, infer_time_q)
            TUI.erase_line()
            cnt_obj.new_res(prev_bomb_lot, bomb.lot, is_bomb_ok, is_lot_ok)

            # Write on .xlsx file & finish
            # ===========================================================
            logmg.i.log(
                "=======================================================%s 탄 %d 추가",
                bomb.lot.name,
                bomb.num,
            )
            ew.add_bomb(bomb)
            ew.save_excel()
            ew.close_excel()
            bomb.done()
            prev_bomb_lot = bomb.lot

            queue_push(prev_res_q, [bomb, is_bomb_ok], max_len=PREV_RES_QUEUE_LEN)
            queue_push(
                infer_time_q, time.time() - st_tm, max_len=NUM_BOMB_AVG_INFER_TIME
            )
            # 화면 캡처
            screenshot = ImageGrab.grab()
            # bbox=(x1, y1, x2, y2)

            # 이미지 파일로 저장
            screenshot.save(f"data/tui_result/{bomb.lot.name}_{bomb.num}.png")

            time.sleep(int(CONFIG["LOOP"]["DELAY"]))
            cls()
    except KeyboardInterrupt:
        logmg.i.log("키 인터럽트 : 작동 중지 ...")
    except Exception as e:
        stat_flags["waiting"] = True
        stat_flags["done"] = True
        print("\n" + str(e) + "\n")
        if PRINT_ERR_TRACEBACK:
            traceback.print_exc()
            print()
        print("Press Any Key To Quit...")
