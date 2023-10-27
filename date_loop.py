import datetime

import TUI
from util import *
from conf import *
from common import *


def loop():
    while stat_flags['start'] is False:
        time.sleep(1)

    while stat_flags['quit'] is False:
        print_lock.acquire()

        set_cursor_pos(1 + COL1_LEFT_MARGIN, 1)
        now = datetime.datetime.now()
        print(now.strftime('%Y-%m-%d %H:%M:%S'), end='', flush=True)
        TUI.mv2col1bot(y_bias=1)

        print_lock.release()
        time.sleep(0.3)
