from configparser import ConfigParser

CONFIG = ConfigParser()
CONFIG.read('data/conf.ini')

PRINT_ERR_TRACEBACK = True

COL1_LEFT_MARGIN = 1
COL1_H_BAR_LEN = 88
COL1_INNER_COL_SPACE = 2

COL2_LEFT_MARGIN = 2
COL2_H_BAR_LEN = 31

PREV_RES_QUEUE_LEN = 13

NUM_BOMB_AVG_INFER_TIME = 5

PATH_IN_EXCEL = 'absolute'  # ['relative', 'absolute']

ROOT_PATH = CONFIG['PATH']['ROOT']

DST_PATH = CONFIG['PATH']['DST']

DIR_PATH = {
    'pre': {
        'camera': f'{ROOT_PATH}/ACDM/image',
        'sensor': f'{ROOT_PATH}/ACDM/data'
    },
    'done': {
        'camera': f'{ROOT_PATH}/{DST_PATH}/image',
        'sensor': {
            'data': f'{ROOT_PATH}/{DST_PATH}/data',
            'html': f'{ROOT_PATH}/{DST_PATH}/laser_result/html',
            'image': f'{ROOT_PATH}/{DST_PATH}/laser_result/image'
        },
        'excel': f'{ROOT_PATH}/{DST_PATH}/excel'
    }
}

DEFECT_CODE = {
    'head': {
        'normal': 'KC256_G09_A_00',
        'match': 'KC256_G09_C_09',  # 비인가된 신관 결합
        'damage': 'KC256_G09_B_04',  # 파손
        'light_corr': 'KC256_G09_D_11',  # 소부식
        'heavy_corr': 'KC256_G09_C_08',  # 중부식
        'comp': 'KC256_G09_B_02',  # 구성품 손상/망실
        'pin_attach': 'KC256_G09_B_01',  # 안전핀 결합
        'pin_exist': 'KC256_G09_B_05',  # 안전핀 망실/절단
        'pin_bent': 'KC256_G09_D_12',  # 안전핀 구부러짐
        'pin_corr': 'KC256_G09_D_13',  # 안전핀 소부식
    },
    'body': {
        'normal': 'KC256_G23_A_00',
        'top': {
            'warn': 'KC256_G23_B_03',  # 적색 경고표지 식별 불가
            'filling_leak': 'KC256_G23_C_10',  # 충전물 누출/흔적
        },
        'bot': {
            'paint_1': 'KC256_G23_D_12',  # 도색 표기 흐림
            'paint_2': 'KC256_G23_C_09',  # 도색 표기 불량
            'paint_3': 'KC256_G23_B_01',  # 도색 표기 착오
            'type': 'KC256_G23_B_02',  # 탄종 혼합
            'crack_shift': 'KC256_G23_B_04',  # 균열 축 뒤틀림
            'light_corr': 'KC256_G23_D_14',  # 소부식
            'heavy_corr': 'KC256_G23_C_06',  # 중부식
            'ring_exist': 'KC256_G23_B_05',  # 폐쇄링 망실
            'ring_damage': 'KC256_G23_C_11',  # 폐쇄링 손상
            'gauge': 'KC256_G23_C_07',  # 링게이지
            'fo': 'KC256_G23_D_13'  # 이물질
        }
    },
    'wing': {
        'normal': 'KC256_G30_A_00',
        'bent': 'KC256_G30_B_01',  # 굴곡 및 파손
        'damage': 'KC256_G30_C_02',  # 파손 및 절단
        'corr': 'KC256_G30_C_03'  # 부식
    },
    'powder': {
        'normal': 'KC256_G38_A_00',
        'top': {
            'position': 'KC256_G38_B_02',  # 부적절한 위치
        },
        'bot': {
            'exist': 'KC256_G38_B_01',  # 부분 망실
            'condition': 'KC256_G38_C_03',  # 장력상실 또는 파손
            'discolor': 'KC256_G38_D_04',  # 약포 변색
            'fo': 'KC256_G38_D_05'  # 이물질
        }
    },
    'anchor': {
        'normal': 'KC256_G40_A_00',
        'exist': 'KC256_G40_C_01',  # 부분 망실
        'heavy_corr': 'KC256_G40_C_02',  # 중부식
    }
}
