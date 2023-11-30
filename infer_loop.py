import time
import traceback

from PIL import ImageGrab

import pygetwindow as gw
import pyautogui

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


def capture_program_window(program_name, save_path):
    # 프로그램 창 찾기
    program_window = gw.getWindowsWithTitle(program_name)

    if not program_window:
        print(f"프로그램 '{program_name}'을 찾을 수 없습니다.")
        return

    program_window = program_window[0]

    # 창을 활성화하고 잠시 대기
    program_window.activate()
    time.sleep(1)

    # 현재 화면 캡쳐
    screenshot = pyautogui.screenshot()

    # 프로그램 창의 좌표와 크기 가져오기
    left, top, width, height = (
        program_window.left,
        program_window.top,
        program_window.width,
        program_window.height,
    )

    # 창 부분만 잘라내기
    window_capture = screenshot.crop((left, top, left + width, top + height))

    # 이미지 저장
    window_capture.save(save_path)
    print(f"프로그램 창을 '{save_path}'에 저장했습니다.")


def get_infer_list(bomb, ocr_obj, yolo_obj):
    return [
        [yolo_obj.head_infer, [bomb], "신관, 탄체 결함 판별중"],
        [OCR.do_infer, [bomb, ocr_obj, yolo_obj], "도색표기 결함 판별중"],
        [laser.infer17, [bomb], "변위센서 결함 판별중"],
        [wing.do_infer, [bomb], "날개 결함 판별중"],
        [powder.do_infer, [bomb], "추진장약 결함 판별중"],
        [yolo_obj.powder_infer, [bomb], "추진장약 결함 판별중"],
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
            time.sleep(2)

            # 화면 캡처

            # 프로그램 이름과 저장 경로 설정
            # program_name = "Visual Studio Code"  # 대상 프로그램의 창 제목
            # save_path = f"data/tui_result/{bomb.lot.name}"  # 저장할 이미지 경로 및 이름
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # # 이미지 파일로 저장
            # save_path = save_path + f"{bomb.num}.png"

            # # 함수 호출
            # capture_program_window(program_name, save_path)

            screenshot = ImageGrab.grab()
            # bbox=(x1, y1, x2, y2)

            screenshot_path = f"data/tui_result/{bomb.lot.name}"
            if not os.path.exists(screenshot_path):
                os.makedirs(screenshot_path)
            # 이미지 파일로 저장
            screenshot.save(screenshot_path + f"/{bomb.num}.png")

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
