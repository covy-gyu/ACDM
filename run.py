import threading
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from infer_loop import loop as infer_loop
from quit_loop import loop as quit_input_loop
from date_loop import loop as date_loop

from util import *

def run():
    hide_cursor()
    print('\033[97m', end='')
    time.sleep(1)
    th_date_loop = threading.Thread(target=date_loop)
    th_date_loop.daemon = True
    th_date_loop.start()

    th_infer_loop = threading.Thread(target=infer_loop, args=(1.0, ))
    th_infer_loop.daemon = True
    th_infer_loop.start()

    quit_input_loop()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        exit(0)
