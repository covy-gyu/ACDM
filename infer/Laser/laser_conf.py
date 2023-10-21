from conf import *

infer_15_17 = {
    'threshold': {
        'crack': 0.1e-3,
        'fo': float(CONFIG['LASER_CONF']['THRESHOLD_FO']) / 1000.0,
        'shaft': 0.05e-3
    },
    'phys': {
        'dist_to_axis': 0.05265,
        'sec_per_rev': 18.0,
        'bias': 0.0
    },
    'plot': {
        'unit': 'mm'
    },
    'data': CONFIG['LASER_CONF']['DATA']
}
