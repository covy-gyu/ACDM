import math
import numpy as np
import cv2


def crop_img(src_img, rect):
    return src_img[rect['top']:rect['bottom'] + 1, rect['left']:rect['right'] + 1]


def __get_curve_len_info(vertices):
    res = {
        'total': 0,
        'interval': []
    }

    prev_pos = None
    for v in vertices:
        if prev_pos is None:
            prev_pos = v
            continue

        interval_len = math.sqrt((v[0] - prev_pos[0]) ** 2 + (v[1] - prev_pos[1]) ** 2)
        res['total'] += interval_len
        res['interval'].append(interval_len)

        prev_pos = v

    return res


def __get_warp_key_pts(key_points, len_info, pos_in_s):
    res = [key_points[-1], key_points[0]]

    len_sum = 0
    interval_idx = 0
    for len_in_s in pos_in_s[1:-1]:
        while len_sum + len_info['interval'][interval_idx] < len_in_s:
            len_sum += len_info['interval'][interval_idx]
            interval_idx += 1

        rel_pos_in_interval = (len_in_s - len_sum) / len_info['interval'][interval_idx]
        p1 = key_points[interval_idx]
        p2 = key_points[interval_idx + 1]
        pos = (int((p2[0] - p1[0]) * rel_pos_in_interval + p1[0]),
               int((p2[1] - p1[1]) * rel_pos_in_interval + p1[1]))

        res.insert(1, pos)

    return np.array(res)


def __get_warp_rect(n_rect_per_side, conf):
    len_info = {
        'top': __get_curve_len_info(conf['warp_key_pts']['top']),
        'bottom': __get_curve_len_info(conf['warp_key_pts']['bottom'])
    }

    max_rad = np.pi / 2 * 0.5
    pts_rel_pos = np.sin(np.linspace(0, max_rad, n_rect_per_side + 1)) / np.sin(max_rad)
    pts_pos_in_s = {
        'top': pts_rel_pos * len_info['top']['total'],
        'bottom': pts_rel_pos * len_info['bottom']['total']
    }

    warp_key_pts = {}
    for key in ['top', 'bottom']:
        buf1 = __get_warp_key_pts(conf['warp_key_pts'][key], len_info[key], pts_pos_in_s[key])
        buf2 = np.copy(buf1)
        buf2[:, 0] = 2 * conf['warp_key_pts'][key][0][0] - buf2[:, 0]  # Reflect x position
        warp_key_pts[key] = np.append(buf1, buf2[-2::-1], axis=0)

    res = []
    for i in range(n_rect_per_side * 2):
        res.append({
            'l_top': warp_key_pts['top'][i],
            'r_top': warp_key_pts['top'][i + 1],
            'l_bot': warp_key_pts['bottom'][i],
            'r_bot': warp_key_pts['bottom'][i + 1]
        })

    return res


def __do_warp(image, warp_rect):
    base_idx = len(warp_rect) // 2
    base_rect = warp_rect[base_idx]
    width = base_rect['r_bot'][0] - base_rect['l_bot'][0]
    height = base_rect['l_bot'][1] - base_rect['l_top'][1]

    res_width = len(warp_rect) * width
    res_height = height
    res = np.zeros((res_height, res_width, 3), dtype=np.uint8)

    for i, rect in enumerate(warp_rect):
        src = np.array((rect['l_top'], rect['r_top'], rect['r_bot'], rect['l_bot']), dtype=np.float32)
        dst = np.array(((width * i, 0), (width * (i + 1), 0),
                        (width * (i + 1), height - 1), (width * i, height - 1)), dtype=np.float32)
        mat = cv2.getPerspectiveTransform(src, dst)
        warped_img = cv2.warpPerspective(image, mat, (res_width, res_height))
        res[:, width * i:width * (i + 1)] = warped_img[:, width * i:width * (i + 1)]

    return res


def warp_img(image, conf):
    res = __do_warp(image, __get_warp_rect(5, conf))
    return res


def get_start_idx(img_list, conf):
    def char_chk(image):
        rng = conf['char_chk_range']
        hist = cv2.calcHist([image[rng[0]:rng[1] + 1]],
                            [0], None, [256], [100, 256])
        return np.sum(hist)

    chk_res = list(map(char_chk, img_list))
    start_idx = chk_res.index(max(chk_res))

    return start_idx


def concat_images(img_list, conf):
    overlap = conf['concat_overlap']

    num_img = len(img_list)
    start_idx = get_start_idx(img_list, conf)

    img_shape = img_list[0].shape
    res = np.zeros((img_shape[0], (img_shape[1] - overlap * 2) * 6, 3), dtype=np.uint8)
    for i in range(num_img):
        img_idx = (start_idx + i) % num_img

        x_offset = (img_shape[1] - overlap * 2) * i
        res[:, x_offset:x_offset + (img_shape[1] - overlap * 2)] = img_list[img_idx][:, overlap:-overlap]

    return res


def rotate_img(src, angle):
    (h, w) = src.shape[:2]
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(src, mat, (w, h))
