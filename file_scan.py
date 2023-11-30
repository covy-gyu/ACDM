import shutil
import time

import mortar
from util import *
from conf import *
from common import *
from logs import logmg


class LotsStat:
    """
            Case No. │ 1  2  3  4  5  6  7  8  9 10  | Total 10 Cases
    ─────────────────┼──────────────────────────────
           LOT_EXIST │ 0  1  1  1  1  1  1  1  1  1
      NEXT_LOT_EXIST │ 0  0  1  0  1  0  1  1  1  1
         DATA_IN_LOT │ 0  0  0  1  1  1  1  0  1  1
         BOMB_IN_LOT │ 0  0  0  0  0  1  1  0  0  1
    DATA_IN_NEXT_LOT │ 0  0  0  0  0  0  0  1  1  1
    ─────────────────┴──────────────────────────────
    """

    def __init__(self):
        self.LOT_EXIST = False
        self.NEXT_LOT_EXIST = False

        self.DATA_IN_LOT = False
        self.BOMB_IN_LOT = False

        self.DATA_IN_NEXT_LOT = False


def get_lots_info():
    """
    get list of lots ascending sorted by modified time

    :return: list of lots
    """
    img_path = rep_sl(DIR_PATH["pre"]["camera"])
    # lot_names = sorted(
    #     os.listdir(img_path), key=lambda x: os.path.getmtime(os.path.join(img_path, x))
    # )
    lot_names = os.listdir(img_path)
    img_paths = [rep_sl_join(img_path, lot) for lot in lot_names]
    lot_mtimes = [os.stat(path).st_mtime for path in img_paths]
    lots = list(zip(lot_names, img_paths, lot_mtimes))
    lots.sort(key=lambda lot: lot[2])  # Sort by last modified time
    for idx, (name, path, m_time) in enumerate(lots):
        lots[idx] = {"name": name}
    return lots


def chk_lots_stat():
    res = LotsStat()

    # Check lot existence
    lots = get_lots_info()
    logmg.m.log("get lots info : %r", lots)
    if len(lots) == 0:
        return res
    if len(lots) >= 1:
        res.LOT_EXIST = True
    if len(lots) >= 2:
        res.NEXT_LOT_EXIST = True

    # Check data stat
    img_dir = rep_sl(DIR_PATH["pre"]["camera"])
    sensor_data_dir = rep_sl(DIR_PATH["pre"]["sensor"])

    num_sensor_data = len(os.listdir(sensor_data_dir))

    img_cnt = 0
    for path, dirs, files in os.walk(rep_sl_join(img_dir, lots[0]["name"])):
        img_cnt += len(files)

    if img_cnt > 0:
        res.DATA_IN_LOT = True
    if img_cnt >= 24 and num_sensor_data > 0:
        res.BOMB_IN_LOT = True

    if res.NEXT_LOT_EXIST is True:
        # Check is next lot has data
        img_cnt = 0

        for path, dirs, files in os.walk(rep_sl_join(img_dir, lots[1]["name"])):
            img_cnt += len(files)

        if img_cnt > 0:
            res.DATA_IN_NEXT_LOT = True

    return res


def chk_conditions(conditions):
    for c in conditions:
        if c is not False:
            return True
    return False


def get_img_data_paths(lot_name):
    # img_dir_path = rep_sl_join(DIR_PATH['pre']['camera'], lot_name)

    # img_paths = {'CAM' + str(i): [] for i in range(1, 5)}

    # for (path, dirs, files) in os.walk(img_dir_path):
    #     # key = rep_sl(path).split('/')[-1]
    #     if key in img_paths.keys():
    #         img_paths[key] += [rep_sl_join(path, f) for f in files]
    #         logmg.m.log("paths : %s",img_paths)

    # for key in img_paths.keys():
    #     img_paths[key].sort()
    #     img_paths[key] = [get_abs_path(img_path) for img_path in img_paths[key][:6]]

    # return img_paths

    # img_base_dir = rep_sl_join(DIR_PATH['pre']['camera'], lot_name)

    # img_paths = {'CAM' + str(i): [] for i in range(1, 5)}

    # for cam_num in range(1, 5):
    #     cam_dir = os.path.join(img_base_dir, f'CAM{cam_num}')
    #     for (path, dirs, files) in os.walk(cam_dir):
    #         img_paths['CAM' + str(cam_num)] += [rep_sl_join(path, f) for f in files]

    # for key in img_paths.keys():
    #     img_paths[key].sort()
    #     img_paths[key] = [get_abs_path(img_path) for img_path in img_paths[key][:6]]

    # return img_paths

    img_base_dir = rep_sl_join(DIR_PATH["pre"]["camera"], lot_name)
    img_paths = {"CAM" + str(i): [] for i in range(1, 5)}

    for path, dirs, files in os.walk(img_base_dir):
        if any(not img_paths[key] for key in img_paths):
            for cam_num in range(1, 5):
                cam_dir = os.path.join(path, f"CAM{cam_num}")
                if os.path.exists(cam_dir):
                    for cam_path, cam_dirs, cam_files in os.walk(cam_dir):
                        img_paths["CAM" + str(cam_num)] += [
                            rep_sl_join(cam_path, f) for f in cam_files
                        ]
        else:
            break

    for key in img_paths.keys():
        img_paths[key].sort()
        img_paths[key] = [get_abs_path(img_path) for img_path in img_paths[key][:6]]

    return img_paths


def get_sensor_data_file_path():
    data_path = rep_sl(DIR_PATH["pre"]["sensor"])

    file_names = os.listdir(data_path)
    file_paths = list(map(lambda s: rep_sl_join(data_path, s), file_names))

    file_mtimes = list(stat.st_mtime_ns for stat in map(os.stat, file_paths))
    file_idx = file_mtimes.index(min(file_mtimes))

    return get_abs_path(file_paths[file_idx])


def clear_lots():
    lot_names = [lot["name"] for lot in get_lots_info()]
    lot_paths = [
        rep_sl_join(DIR_PATH["pre"]["camera"], lot_name) for lot_name in lot_names
    ]
    for lot_path in lot_paths:
        shutil.rmtree(lot_path)


def get_next_bomb(prev_lot, scan_freq=0.5):
    while True:
        stat = chk_lots_stat()

        # ===================================================================================
        conditions_chg_lot = [  # Change to next lot
            not stat.DATA_IN_LOT and stat.NEXT_LOT_EXIST  # Case 3, 8
        ]

        if chk_conditions(conditions_chg_lot) is True:
            stat_flags["waiting"] = False
            shutil.rmtree(
                rep_sl_join(DIR_PATH["pre"]["camera"], get_lots_info()[0]["name"])
            )
            continue

        # ===================================================================================
        conditions_exit_able = [  # Waiting next bomb
            not stat.LOT_EXIST,  # Case 1
            not stat.DATA_IN_LOT and not stat.NEXT_LOT_EXIST,  # Case 2
        ]

        if chk_conditions(conditions_exit_able) is True:
            print(
                "\r"
                + " " * COL1_LEFT_MARGIN
                + "Waiting Next Bomb... (press any key to quit)",
                end="",
            )
            if stat_flags["quit"] is True:
                clear_lots()
                stat_flags["done"] = True
                return None

            stat_flags["waiting"] = True
            time.sleep(scan_freq)
            continue

        # ===================================================================================
        conditions_data_receiving = [  # Receiving data
            stat.DATA_IN_LOT and not stat.BOMB_IN_LOT  # Case 4, 5, 9
        ]
        try:
            if chk_conditions(conditions_data_receiving) is True:
                print("\r" + " " * (50 + COL1_LEFT_MARGIN), end="")
                print("\r" + " " * COL1_LEFT_MARGIN + "Receiving Data...", end="")
                stat_flags["waiting"] = False
                time.sleep(scan_freq)
                continue
        except KeyboardInterrupt as k:
            print(k)
        # ===================================================================================
        """
        conditions_bomb_in_lot = [
            (stat & DirStat.BOMB_IN_LOT)  # Case 6, 7, 10
        ]

        # Don't need to check condition
        """

        stat_flags["waiting"] = False
        break

    lot_info = get_lots_info()[0]
    logmg.m.log("lot_info : %s", lot_info)
    img_paths = get_img_data_paths(lot_info["name"])
    logmg.m.log("imgs_info : %s", img_paths)
    sensor_data_path = get_sensor_data_file_path()
    logmg.m.log("data_info : %s", sensor_data_path)

    if prev_lot is None:  # New lot
        lot = mortar.Lot(name=lot_info["name"])
    else:
        if prev_lot.name != lot_info["name"]:  # New lot
            lot = mortar.Lot(name=lot_info["name"])
        else:  # Lot isn't changed
            lot = prev_lot

    res_bomb = mortar.Bomb(data_paths={"camera": img_paths, "sensor": sensor_data_path})

    lot.add_bomb(res_bomb)

    return res_bomb
