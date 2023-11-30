from common import *
from conf import *
from util import *
from logs import logmg


def print_status(bomb, cnt, prev_res_q, infer_time_q):
    print_lock.acquire()

    col2_st_x = COL1_LEFT_MARGIN + COL1_H_BAR_LEN + COL2_LEFT_MARGIN + 1
    col3_st_x = col2_st_x + COL2_H_BAR_LEN

    # Print infer time
    # ==========================================================================================
    if len(infer_time_q) != 0:
        avg_infer_time = np.mean(infer_time_q)
        infer_time_str = "Avg {:.2f}s/Bomb".format(avg_infer_time)
        set_cursor_pos(col3_st_x - len(infer_time_str), 0)
        print(infer_time_str)

    # Print COL1
    # ==========================================================================================
    set_cursor_pos(0, 0)

    print()
    print()
    print(
        " " * COL1_LEFT_MARGIN
        + " " * int((COL1_H_BAR_LEN - len("Current Bomb")) / 2)
        + "Current Bomb"
    )
    print(" " * COL1_LEFT_MARGIN + "─" * COL1_H_BAR_LEN)
    print(" " * COL1_LEFT_MARGIN + "      LOT : {}".format(bomb.lot.name))
    print(" " * COL1_LEFT_MARGIN + " BOMB NUM : {}".format(bomb.num))
    print(" " * COL1_LEFT_MARGIN + "─" * COL1_H_BAR_LEN)
    print()
    print(bomb, end="")
    print()
    print(" " * COL1_LEFT_MARGIN + "─" * COL1_H_BAR_LEN)

    # Print COL2
    # ==========================================================================================

    n_lot = {
        "total": cnt["lot"]["total"],
        "defect": cnt["lot"]["defect"],
        "normal": cnt["lot"]["total"] - cnt["lot"]["defect"],
    }

    n_bomb = {
        "total": cnt["bomb"]["total"],
        "defect": cnt["bomb"]["defect"],
        "normal": cnt["bomb"]["total"] - cnt["bomb"]["defect"],
    }

    col2_lines = [
        "",
        "",
        " " * int((COL2_H_BAR_LEN - len("Total")) / 2) + "Total",
        "─" * COL2_H_BAR_LEN,
        "  LOTS : {} (\033[92m{}\033[97m/\033[91m{}\033[97m)".format(
            n_lot["total"], n_lot["normal"], n_lot["defect"]
        ),
        " BOMBS : {} (\033[92m{}\033[97m/\033[91m{}\033[97m)".format(
            n_bomb["total"], n_bomb["normal"], n_bomb["defect"]
        ),
        "─" * COL2_H_BAR_LEN,
        "",
        " " * int((COL2_H_BAR_LEN - len("Prev Results")) / 2) + "Prev Results",
        "─" * COL2_H_BAR_LEN,
    ]

    for i, prev_res in enumerate(prev_res_q):
        lot_name = prev_res[0].lot.name
        if i != 0:
            if prev_res_q[i - 1][0].lot.name == lot_name:
                lot_name = ""
        test_res = "\033[92m정상\033[97m" if prev_res[1] is True else "\033[91m결함\033[97m"

        prev_infer_stat = prev_res[0].infer_stat
        total = 0
        ok_cnt = 0
        for cat1 in prev_infer_stat.keys():
            for cat2 in prev_infer_stat[cat1].keys():
                total += 1
                if prev_infer_stat[cat1][cat2]["res"] is True:
                    ok_cnt += 1
        ok_perc = round(ok_cnt / total * 100, 1)

        col2_lines.append(
            "{:<15s}  {:2d} {} {:>5.1f}%".format(
                lot_name, prev_res[0].num, test_res, ok_perc
            )
        )
    col2_lines += [""] * (PREV_RES_QUEUE_LEN - len(prev_res_q))
    col2_lines += ["─" * COL2_H_BAR_LEN]

    for y, line in enumerate(col2_lines):
        set_cursor_pos(col2_st_x, y + 1)
        print(line, end="")

    # Move cursor to bottom of COL1
    mv2col1bot()

    print_lock.release()


def print_status_msg(msg):
    print_lock.acquire()

    mv2col1bot()
    erase_line()
    print(" " * COL1_LEFT_MARGIN + msg + "...", end="", flush=True)

    print_lock.release()


def erase_line():
    l_col1 = COL1_LEFT_MARGIN + COL1_H_BAR_LEN + COL1_INNER_COL_SPACE
    l_col2 = COL2_LEFT_MARGIN + COL2_H_BAR_LEN
    l_total = l_col1 + l_col2
    print(" " * l_col1 + "\r", end="")


def mv2col1bot(y_bias=0):
    set_cursor_pos(0, 25 + y_bias)
