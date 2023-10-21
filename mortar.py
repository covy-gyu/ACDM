import shutil
from wcwidth import wcswidth

from util import *
from conf import *
from logs import logmg


class Lot:
    def __init__(self, name):
        self.name = name
        self.bombs = []

    def __len__(self):
        return len(self.bombs)

    def __getitem__(self, item):
        return self.bombs[item]

    def add_bomb(self, bomb):
        bomb.lot = self
        bomb.num = len(self.bombs)

        self.bombs.append(bomb)


class Bomb:
    class Defect:
        def __init__(self):
            self.defect = {  # Defect Code
                'head': {'res': [[] for _ in range(7)], 'cam': 'CAM4'},
                'body': {
                    'top': {'res': [[] for _ in range(7)], 'cam': 'CAM4'},
                    'bot': {'res': [[] for _ in range(7)], 'cam': 'CAM3'}
                },
                'wing': {'res': [[] for _ in range(7)], 'cam': 'CAM1'},
                'powder': {
                    'top': {'res': [[] for _ in range(7)], 'cam': 'CAM2'},
                    'bot': {'res': [[] for _ in range(7)], 'cam': 'CAM1'}
                },
                'anchor': {
                    'top': {'res': [[] for _ in range(7)], 'cam': 'CAM2'},
                    'bot': {'res': [[] for _ in range(7)], 'cam': 'CAM1'}
                }
            }

        def __getitem__(self, item):
            return self.defect[item]

        @staticmethod
        def is_normal(defect_list):
            for cam_defect in defect_list:
                if cam_defect:
                    return False
            return True

        def get_defect(self):
            d = self.defect

            res = {
                'CAM1': [[] for _ in range(7)],
                'CAM2': [[] for _ in range(7)],
                'CAM3': [[] for _ in range(7)],
                'CAM4': [[] for _ in range(7)]
            }

            for part in d.keys():
                if 'res' in d[part].keys():
                    if self.is_normal(d[part]['res']) is True:
                        d[part]['res'][6].append(DEFECT_CODE[part]['normal'])

                    cam = d[part]['cam']
                    res[cam] = merge_lists(res[cam], d[part]['res'])
                else:  # Separated as top and bot
                    merged_res = merge_lists(d[part]['top']['res'], d[part]['bot']['res'])
                    if self.is_normal(merged_res) is True:
                        d[part]['top']['res'][6].append(DEFECT_CODE[part]['normal'])

                    for tb in ['top', 'bot']:
                        cam = d[part][tb]['cam']
                        res[cam] = merge_lists(res[cam], d[part][tb]['res'])


            for cam in res.keys():
                total = [item for sublist in res[cam][:6] for item in sublist]
                res[cam][6].extend(total)
                res[cam] = [','.join(list(set(sublist))) if sublist else '' for sublist in res[cam]]

            return res
        
        def get_defect_by_parts(self):
            d = self.defect
            data = {
                'head': [],
                'body': [],
                'powder': [],
                'wing' : [],
                'anchor': []
            }

            for part in d.keys():
                if 'res' in d[part].keys():
                    data[part] = d[part]['res']
                else:  # Separated as top and bot
                    data[part] = merge_lists(d[part]['top']['res'], d[part]['bot']['res'])

            res = {}

            for key, values in data.items():
                res[key] = ','.join(list(set([item for sublist in values for item in sublist if item])))

            return res


    def __init__(self, data_paths):
        self.lot = None
        self.num = 0

        self.img_path = data_paths['camera']  # {'CAM1': [] ~ 'CAM4': []}
        self.sensor_data_path = data_paths['sensor']
        self.sensor_res_path = {'html': '', 'image': ''}

        self.defect = self.Defect()

        infer_stat_conf = {
            'head': [
                ['match', 1],
                ['damage', 1],
                ['light_corr', 1],
                ['heavy_corr', 1],
                ['comp', 1],
                ['pin_attach', 1],
                ['pin_exist', 1],
                ['pin_bent', 1],
                ['pin_corr', 1]
            ],
            'body': [
                ['paint_1', 1],
                ['paint_2', 1],
                ['paint_3', 1],
                ['type', 1],
                ['warn', 2],
                ['crack_shift', 1],
                ['light_corr', 1],
                ['heavy_corr', 2],
                ['ring_exist', 1],
                ['ring_damage', 1],
                ['gauge', 1],
                ['filling_leak', 1],
                ['fo', 1]
            ],
            'wing': [
                ['bent', 1],
                ['damage', 1],
                ['corr', 1]
            ],
            'powder': [
                ['exist', 1],
                ['position', 1],
                ['condition', 1],
                ['discolor', 1],
                ['fo', 1]
            ],
            'anchor': [
                ['exist', 1],
                ['heavy_corr', 1]
            ]
        }
        self.infer_stat = {'head': {}, 'body': {}, 'wing': {}, 'powder': {}, 'anchor': {}}
        for key in self.infer_stat.keys():
            for stat_conf in infer_stat_conf[key]:
                self.infer_stat[key][stat_conf[0]] = {'total': stat_conf[1], 'done': 0, 'res': None}

    def __str__(self):
        l_margin = ' ' * COL1_LEFT_MARGIN

        res = ''
        contents = [
            [['탄체', 'body'],                  ['신관', 'head'],                  ['추진장약', 'powder']],
            [['도색 표기 흐림', 'paint_1'],       ['비인가 신관 결합', 'match'],       ['부분 망실', 'exist']],
            [['도색 표기 불량', 'paint_2'],       ['파손', 'damage'],                ['부적절한 위치', 'position']],
            [['도색 표기 착오', 'paint_3'],       ['소부식', 'light_corr'],           ['장력상실/파손', 'condition']],
            [['탄종 혼합', 'type'],              ['중부식', 'heavy_corr'],           ['변색', 'discolor']],
            [['적색 경고표지', 'warn'],           ['구성품 손상/망실', 'comp'],        ['이물질', 'fo']],
            [['균열 또는 뒤틀림', 'crack_shift'],  ['안전핀 결합', 'pin_attach'],      [None]],
            [['소부식', 'light_corr'],           ['안전핀 망실/절단', 'pin_exist'],   ['날개', 'wing']],
            [['중부식', 'heavy_corr'],           ['안전핀 구부러짐', 'pin_bent'],     ['굴곡 및 파손', 'bent']],
            [['폐쇄링 망실', 'ring_exist'],       ['안전핀 소부식', 'pin_corr'],       ['파손 및 절단', 'damage']],
            [['폐쇄링 손상', 'ring_damage'],      [None],                           ['부식', 'corr']],
            [['링게이지', 'gauge'],               ['추진약멈치', 'anchor'],           [None]],
            [['충전물 누출/흔적', 'filling_leak'], ['망실', 'exist'],                 [None]],
            [['이물질', 'fo'],                   ['중부식', 'heavy_corr'],           [None]]
        ]

        max_cat2_len = [0] * len(contents[0])
        for i in range(len(contents)):
            for j in range(len(contents[i])):
                if i == 0 or contents[i][j][0] is None:
                    continue
                max_cat2_len[j] = max(max_cat2_len[j], wcswidth(contents[i][j][0]))

        cat1 = [''] * len(contents[0])
        for i, row_contents in enumerate(contents):
            buf = l_margin
            for j, item in enumerate(row_contents):
                if cat1[j] == '' and item[0] is not None:  # New cat1
                    cat1[j] = item[1]
                    buf += ' ' + item[0]
                    buf += ' ' * (wcswidth(' └─ ') + max_cat2_len[j] + 8 - wcswidth(item[0]))
                    buf += ' ' * COL1_INNER_COL_SPACE
                    continue

                if item[0] is None:  # Curr cat1 done
                    cat1[j] = ''
                    buf += ' ' * (wcswidth('  └─ ') + max_cat2_len[j] + 8)
                    buf += ' ' * COL1_INNER_COL_SPACE
                    continue

                if i == len(contents) - 1:
                    buf += '  └─ '
                elif contents[i + 1][j][0] is None:
                    buf += '  └─ '
                else:
                    buf += '  ├─ '

                buf += ' ' * (max_cat2_len[j] - wcswidth(item[0])) + item[0] + ' : '
                stat = self.infer_stat[cat1[j]][item[1]]
                if stat['res'] is not None:
                    buf += '\033[92m정상\033[97m ' if stat['res'] is True else '\033[91m결함\033[97m '
                else:
                    buf += '     '  # '{:2d}/{:2d}'.format(stat['done'], stat['total'])
                buf += ' ' * COL1_INNER_COL_SPACE
            res += buf + '\n'

        return res

    def update_infer_stat(self, cat1, cat2, res):
        self.infer_stat[cat1][cat2]['done'] += 1

        if self.infer_stat[cat1][cat2]['res'] is False:
            return

        if res is False:
            self.infer_stat[cat1][cat2]['res'] = False
            return

        if self.infer_stat[cat1][cat2]['done'] == self.infer_stat[cat1][cat2]['total']:
            self.infer_stat[cat1][cat2]['res'] = True

    def done(self):
        self.__move_data_files()

    def __move_data_files(self):
        """
        Move data files of processed bomb to dir_path['done'] directories
        """
        pre_img_path = get_abs_path(DIR_PATH['pre']['camera'])
        pre_sensor_data_path = get_abs_path(DIR_PATH['pre']['sensor'])

        done_img_path = get_abs_path(DIR_PATH['done']['camera'])
        done_sensor_data_path = get_abs_path(DIR_PATH['done']['sensor']['data'])

        # Move image files
        for key in self.img_path.keys():
            for img_path in self.img_path[key]:
                new_img_path = img_path.replace(pre_img_path, done_img_path)
                try:
                    shutil.move(img_path, new_img_path)
                except IOError:
                    os.makedirs(os.path.dirname(new_img_path))
                    shutil.move(img_path, new_img_path)

        # Move sensor data file
        data_path = self.sensor_data_path
        new_data_path = data_path.replace(pre_sensor_data_path, done_sensor_data_path)
        try:
            shutil.move(data_path, new_data_path)
        except IOError:
            os.makedirs(os.path.dirname(new_data_path))
            shutil.move(data_path, new_data_path)
