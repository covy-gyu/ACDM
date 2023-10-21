import os, sys
import time, datetime
import math
import numpy as np
import cv2


def rep_sl(path):
    return path.replace('\\', '/')


def rep_sl_join(p1, p2):
    return rep_sl(os.path.join(rep_sl(p1), rep_sl(p2)))


def get_abs_path(path):
    return rep_sl(os.path.abspath(path))


def cls():
    os.system('cls')


def set_cursor_pos(x, y):
    print('\033[{};{}H'.format(y, x), end='')


def load_images(image_paths, flag=cv2.IMREAD_COLOR):
    return list(map(lambda path: cv2.imread(path, flag), image_paths))


def get_binary(img, mask_rect, thresh, comp_mask=False):
    mask = np.zeros_like(img)
    mask = cv2.rectangle(mask, mask_rect['top_left'], mask_rect['bottom_right'],
                         color=(255, 255, 255), thickness=cv2.FILLED)
    if comp_mask is True:
        mask = cv2.bitwise_not(mask)
    masked_img = cv2.bitwise_and(img, mask)
    _, binary = cv2.threshold(masked_img, thresh, 255, cv2.THRESH_BINARY)

    return binary


def get_created_time(path):
    mod_time = time.localtime(os.path.getmtime(path))
    mod_time = time.strftime("%Y-%m-%d %H:%M:%S", mod_time)

    # given_datetime = datetime.strptime(mod_time, '%Y-%m-%d %H:%M:%S')
    # base_datetime = datetime(1900,1,1,0,0,0)

    # difference = given_datetime - base_datetime
    # result = difference.days + difference.seconds / (24 * 60 * 60)

    # result = math.trunc((result+2) * (10 ** 10)) / (10 ** 10)

    return mod_time


def merge_lists(*args):
    return [sum(rows, []) for rows in zip(*args)]


def path_abs2rel(abs_path):
    if abs_path == '':
        return ''

    return rep_sl(os.path.relpath(abs_path))


if os.name == 'nt':
    import msvcrt
    import ctypes

    class _CursorInfo(ctypes.Structure):
        _fields_ = [("size", ctypes.c_int),
                    ("visible", ctypes.c_byte)]


def hide_cursor():
    if os.name == 'nt':
        ci = _CursorInfo()
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
        ci.visible = False
        ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
    elif os.name == 'posix':
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()


def show_cursor():
    if os.name == 'nt':
        ci = _CursorInfo()
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
        ci.visible = True
        ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
    elif os.name == 'posix':
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
