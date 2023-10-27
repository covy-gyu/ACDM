import msvcrt
import time
import sys
from common import *


def loop():
    while not stat_flags['done']:
        try:
            key = msvcrt.getch().decode()
            if key == chr(27):  # ASCII 코드 27은 Esc 키에 대응합니다.
                print("프로그램을 종료합니다.")
                sys.exit()
            if stat_flags['waiting']:
                break
        except UnicodeDecodeError:
            continue

    stat_flags['quit'] = True
    while not stat_flags['done']:
        time.sleep(0.5)

    exit(0)
