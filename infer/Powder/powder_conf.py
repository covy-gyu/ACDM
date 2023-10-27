infer_30 = {
    'crop_rect': {
        'top': 0,
        'bottom': 200,
        'left': 520,
        'right': 760
    },
    'brightness_threshold': {
        'lower': 50,
        'upper': 200
    },
    'density_threshold': 0.1,
    'pixel_cnt': 7500
}

infer_31 = {
    'template_img': 'data/resource/31_mask.png',
    'mask_rect': {
        'top_left': (380, 0),
        'bottom_right': (900, 480)
    },
    'infer_threshold': 0.004
}


infer_32 = {
    'mask_rect': {
        'top_left': (420, 200),
        'bottom_right': (860, 960)
    }
}


infer_33 = {
    'mask_rect': {
        'top_left': (420, 0),
        'bottom_right': (860, 200)
    },
    'rel_avg': 3051.3,
    'threshold': 100000
}

infer_34 = {
    'mask_rect': {
        'top_left': (420, 0),
        'bottom_right': (860, 300)
    },
    'fo_detect_area': {
        'min': 23,
        'max': 26
    }
}
