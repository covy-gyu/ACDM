import copy
from typing import Union, Callable, Tuple

import plotly.graph_objects as go
from plotly.offline import plot

from conf import *
from util import *

import infer.Laser.laser_conf as laser_conf
import logs.logmg as logmg

import matplotlib.pyplot as plt


class LaserData:
    def __init__(self, file_path):
        file = open(file_path, encoding='utf-8', errors='ignore')
        self.raw_str_data = [line.split(',') for line in file.readlines()]
        file.close()

        self.orig_data = None
        self.ztr_data = None

    def preprocess(self, save_path, dist_to_axis):
        self.orig_data = None
        self.ztr_data = None

        data = np.array(self.__get_defect_removed_rows())
        self.__interpolate_rows(data)
        data = self.__get_defect_removed_cols(data)
        self.__save_defect_removed_data(data, save_path=save_path)

        if laser_conf.infer_15_17['data'] == 'z':
            self.orig_data = {'z': np.linspace(-0.01, 0.01, len(data[0])), 'r': 0.001 * data}
        else:
            self.orig_data = {'z': 0.001 * data[0], 'r': 0.001 * data[2::3]}  # Change length unit [mm] to [m]

        self.ztr_data = self.__get_ztr_data(self.orig_data, dist_to_axis)

    def __get_defect_removed_rows(self):
        res = []

        defected_idx = []
        n_col = None
        for idx, line in enumerate(self.raw_str_data):
            try:
                buf = []

                for val in line:
                    if '\n' in val:
                        break
                    buf.append(float(val))

                if n_col is None:
                    n_col = len(buf)

                if len(buf) != n_col:  # Check number of columns
                    raise ValueError

                res.append(buf)
            except ValueError:  # Row is defected
                for i in range(3):
                    idx_buf = idx - idx % 3 + i
                    if idx_buf not in defected_idx[-3:] and idx_buf < len(self.raw_str_data):
                        defected_idx.append(idx_buf)

                res.append([])

        # Remove defected rows
        defected_idx.reverse()
        for idx in defected_idx:
            curr_idx = idx
            del res[idx]

        return res

    @staticmethod
    def __get_defect_removed_cols(data):
        val1 = data[0][0]
        val2 = data[0][-1]

        st_idx = 0
        en_idx = 0
        for i in range(len(data[0])):
            if data[0][i] != val1:
                st_idx = i - 1
                break

        for i in range(len(data[0]) - 1, -1, -1):
            if data[0][i] != val2:
                en_idx = i + 1
                break

        return data[:, st_idx: en_idx + 1]

    @staticmethod
    def rm_paused_intervals(data, color_map):
        r = data['r']

        mse_buf = []
        for i in range(1, len(r) - 1):
            mse1 = np.mean((r[i] - r[i-1]) ** 2)
            mse2 = np.mean((r[i+1] - r[i]) ** 2)
            mse_buf.append(min(mse1, mse2))

        sz_wnd = int(len(r) / 360)

        buf = []
        for i in range(sz_wnd, len(mse_buf) - sz_wnd):
            buf.append(min(mse_buf[i - sz_wnd:i + sz_wnd + 1]))

        threshold = (max(buf) - min(buf)) * 0.2

        r = r[sz_wnd + 1:-sz_wnd - 1]
        r = r[buf > threshold]

        color_map_res = color_map[sz_wnd + 1:-sz_wnd - 1]
        color_map_res = color_map_res[buf > threshold]

        return {
            'z': data['z'],
            'theta': np.linspace(0, 2 * np.pi, len(r)),
            'r': r
        }, color_map_res

    @staticmethod
    def __interpolate_row(row):
        _0_st_idx = None
        for i in range(len(row)):
            if row[i] == 0 and _0_st_idx is None:  # Beginning of 0s
                _0_st_idx = i - 1
                continue

            if _0_st_idx is None:  # Don't need interpolate
                continue

            if row[i] == 0:
                if i != len(row) - 1:
                    continue

            # Do linear interpolation
            st_val = row[_0_st_idx] if _0_st_idx != -1 else row[i]
            en_val = row[i] if i != len(row) - 1 else row[_0_st_idx]

            slope = (en_val - st_val) / (i - _0_st_idx)
            n_zeros = i - _0_st_idx - 1
            if i == len(row) - 1:
                n_zeros += 1

            for j in range(1, 1 + n_zeros):
                row[_0_st_idx + j] = st_val + slope * j

            _0_st_idx = None

    @staticmethod
    def __interpolate_rows(rows):
        for i in range(len(rows)):
            LaserData.__interpolate_row(rows[i])

    @staticmethod
    def __save_defect_removed_data(data, save_path):
        if save_path is None:
            return

        if data is None:
            return

        f = open(save_path, 'w')
        for row in data:
            buf = list(map(lambda val: str(round(val, 6)), row))
            f.write(','.join(buf) + '\n')
        f.close()

    @staticmethod
    def __get_ztr_data(orig_data, dist_to_axis):
        radius = dist_to_axis - orig_data['r']

        res = {
            'z': np.linspace(orig_data['z'][0], orig_data['z'][-1], orig_data['r'].shape[1]),
            'theta': np.linspace(0, 2 * np.pi * 360 / 360, orig_data['r'].shape[0]),
            'r': radius,
            'r_avg': np.average(radius)
        }

        return res


def get_diff_map(data: dict,
                 sz_wnd: Tuple[float, float],
                 q_perc: Tuple[float, float] = (25, 75),
                 func: Callable = np.min) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    var_reduced_radius = np.zeros(data['r'].shape)
    n_theta, n_z = data['r'].shape

    th_idx_rng = (sz_wnd[0] / 2) // (data['r_avg'] * (data['theta'][1] - data['theta'][0]))
    th_idx_rng = int(th_idx_rng)
    z_idx_rng = (sz_wnd[1] / 2) // (data['z'][1] - data['z'][0])
    z_idx_rng = int(z_idx_rng)

    for i in range(n_theta):
        for j in range(n_z):
            theta_st = i - th_idx_rng if i > th_idx_rng else 0
            theta_en = i + th_idx_rng if i + th_idx_rng < n_theta else n_theta
            z_st = j - z_idx_rng if j > z_idx_rng else 0
            z_en = j + z_idx_rng if j + z_idx_rng < n_z else n_z

            test_rect = data['r'][theta_st:theta_en + 1, z_st:z_en + 1]
            var_reduced_radius[i][j] = func(test_rect)

    q1, q3 = np.percentile(var_reduced_radius, q_perc, axis=0)
    base_value = np.mean([q1, q3], axis=0)

    mean_diff_map = var_reduced_radius - base_value
    value_diff_map = var_reduced_radius - data['r']

    return mean_diff_map, value_diff_map, var_reduced_radius


def cylindrical_to_map(data: dict) -> dict:
    """
    Transform (theta, z, r) -> (x, y, z)

    :param data: cylindrical data
    :return: transformed data
    """

    res = {
        'x': np.repeat(data['theta'], len(data['z'])),
        'y': np.tile(data['z'], len(data['theta'])),
        'z': data['r'].reshape(-1),
    }

    return res


def plot_data(plot_conf: dict, data: dict, diff_maps):
    dir_names = {
        'html': os.path.dirname(plot_conf['html']),
        'image': os.path.dirname(plot_conf['image'])
    }

    if not os.path.exists(dir_names['html']):
        os.makedirs(dir_names['html'])

    if not os.path.exists(dir_names['image']):
        os.makedirs(dir_names['image'])

    mean_diff_map = diff_maps[0]
    val_diff_map = diff_maps[1]
    var_reduced_radius = diff_maps[2]

    data, val_diff_map = LaserData.rm_paused_intervals(data, val_diff_map)

    n_max_surface_data = 400000
    n_data = data['r'].shape[0] * data['r'].shape[1]
    div = int(np.ceil(np.sqrt(n_data / n_max_surface_data)))

    surface_data = {
        'z': data['z'][::div] * 100,
        'theta': data['theta'][::div],
        'r': data['r'][::div, ::div] * 100
    }
    surface_ca_data = cylindrical_to_cartesian(surface_data)
    surface_color_base = np.transpose(val_diff_map[::div, ::div])

    plot_diff_3d(plot_conf, surface_ca_data, surface_color_base)
    # plot_bool_3d(out_path, ca_data, diff_map, color_base)


def plot_diff_3d(plot_conf: dict, ca_data: dict, color_base: np.ndarray):
    diff_color = copy.deepcopy(color_base)
    diff_color[diff_color > 0] = 0
    fig = go.Figure(
        go.Surface(x=ca_data['x'], y=ca_data['y'], z=ca_data['z'], surfacecolor=diff_color)
    )
    fig.update_layout(
        scene={
            'xaxis': {
                'nticks': 7,
                'range': [-5, 5]
            },
            'yaxis': {
                'nticks': 7,
                'range': [-5, 5]
            },
            'zaxis': {
                'nticks': 7,
                'range': [-5, 5]
            },
            'aspectmode': 'cube'
        }
    )
    plot(fig, filename=plot_conf['html'], auto_open=False)
    fig.write_image(plot_conf['image'])


def plot_bool_3d(html_path, img_path, cy_data, is_crack, is_corr, res_bool, unit):
    # Get cartesian coordinate data
    ca_data = cylindrical_to_cartesian(cy_data)
    mult = {
        'm': 1,
        'cm': 1e+2,
        'mm': 1e+3
    }
    ca_data['x'] *= mult[unit]
    ca_data['y'] *= mult[unit]
    ca_data['z'] *= mult[unit]

    # Plot data
    if is_crack and is_corr:
        colorscale = [[0, 'red'], [0.5, 'green'], [1, 'yellow']]
    elif not is_crack and not is_corr:
        colorscale = [[0, 'green'], [1, 'green']]
    elif is_crack and not is_corr:
        colorscale = [[0, 'red'], [1, 'green']]
    else:  # not is_crack and is_fo
        colorscale = [[0, 'green'], [1, 'yellow']]

    html_dir = DIR_PATH['done']['sensor']['html']
    img_dir = DIR_PATH['done']['sensor']['image']
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    fig = go.Figure(
        go.Surface(x=ca_data['x'], y=ca_data['y'], z=ca_data['z'], surfacecolor=res_bool, colorscale=colorscale)
    )
    fig.write_image(img_path)
    plot(fig, filename=html_path, auto_open=False)


def plot_scatter_3d(out_path, data, diff_map):
    scatter_data = cylindrical_to_map({'r': diff_map, 'theta': data['theta'], 'z': data['z']})
    marker = dict(size=1, opacity=0.8, color=scatter_data['z'])
    fig = go.Figure(
        go.Scatter3d(x=scatter_data['x'], y=scatter_data['y'], z=scatter_data['z'],
                     mode='markers', marker=marker)
    )
    plot(fig, filename=out_path + 'scatter_3D.html', auto_open=False)


def get_cylindrical_deltas(data: dict) -> dict:
    """
    Get delta values from cylindrical coordinate data

    :param data: cylindrical coordinate data (e.g. return dict of get_data_from_file())
    :return: dictionary of delta values
             idx 0 : minimum value, idx 1 : maximum value
    """

    res = dict()
    d_theta = data['theta'][1:] - data['theta'][:-1]
    res['d_theta'] = [np.min(d_theta), np.max(d_theta)]

    d_z = data['z'][1:] - data['z'][:-1]
    res['d_z'] = [np.min(d_z), np.max(d_z)]

    r_rng = [np.min(data['r']), np.max(data['r'])]

    res['d_l'] = [r_rng[0] * np.min(d_theta), r_rng[1] * np.max(d_theta)]

    return res


def add_test_patch(data, width, height, thickness=-1e-3):
    deltas = get_cylindrical_deltas(data)
    d_l = (deltas['d_l'][0] + deltas['d_l'][1]) / 2
    d_z = (deltas['d_z'][0] + deltas['d_z'][1]) / 2

    th_idx_rng = int((width / 2) / d_l)
    z_idx_rng = int((height / 2) / d_z)

    ctr_th_idx = int(len(data['theta']) / 2)
    ctr_z_idx = int(len(data['z']) / 2)
    th_st_idx = ctr_th_idx - th_idx_rng
    th_en_idx = ctr_th_idx + th_idx_rng
    z_st_idx = ctr_z_idx - z_idx_rng
    z_en_idx = ctr_z_idx + z_idx_rng

    for i in range(th_st_idx, th_en_idx + 1):
        for j in range(z_st_idx, z_en_idx + 1):
            data['r'][i][j] += thickness


def cylindrical_to_cartesian(data: dict) -> dict:
    """
    Transform cylindrical coordinate vertices data to cartesian coordinate vertices

    :param data: cylindrical coordinate vertices data
    :return: cartesian coordinate vertices data
             format : location of j-th vertex of i-th z-value
                      = (x[i][j], y[i][j], z[i][j])
    """

    x = np.transpose((data['r'] * np.cos(data['theta']).reshape(-1, 1)))
    y = np.transpose((data['r'] * np.sin(data['theta']).reshape(-1, 1)))
    z = np.repeat(data['z'].reshape(-1, 1), len(data['theta']), axis=1)

    return {'x': x, 'y': y, 'z': z}


def is_defected(diff_data: np.ndarray, threshold: float, over_th: True) -> bool:
    if over_th is True:
        return not np.array(diff_data < threshold).all()
    else:
        return not np.array(diff_data > threshold).all()


def run(data_path,
        threshold: dict,
        phys_conf: dict,
        plot_conf: dict,
        sz_diff_wnd: Union[tuple, list] = (0.5e-3, 0.5e-3)):
    # Read and preprocess data
    laser_data = LaserData(data_path)
    laser_data.preprocess(None, phys_conf['dist_to_axis'])

    # Find cracks
    crack_diff_map = get_diff_map(laser_data.ztr_data, sz_wnd=sz_diff_wnd, func=np.min)
    crack_bool = crack_diff_map[0] < -threshold['crack']

    # Find foreign objects
    fo_diff_map = get_diff_map(laser_data.ztr_data, sz_wnd=sz_diff_wnd, func=np.max)
    fo_bool = fo_diff_map[0] > threshold['fo']

    # Detect defects
    is_crack = is_defected(crack_diff_map[0], threshold=-threshold['crack'], over_th=False)
    is_corr = is_defected(fo_diff_map[0], threshold=threshold['fo'], over_th=True)

    # Create result bool map
    diff_shape = crack_diff_map[0].shape
    _res_bool = np.zeros(diff_shape)
    _res_bool[crack_bool] = -1  # Crack = -1
    _res_bool[fo_bool] = 1  # FO = 1
    res_bool = np.transpose(_res_bool)

    # Detect shaft shift
    radius_buf = copy.deepcopy(laser_data.ztr_data['r'])
    radius_buf[_res_bool != 0] = np.NAN
    r_min = np.nanmin(radius_buf, axis=0)
    radius_buf[_res_bool != 0] = np.NAN
    r_max = np.nanmax(radius_buf, axis=0)

    r_diff_max = np.max(r_max - r_min)
    is_shaft_shifted = r_diff_max > threshold['shaft']

    # Plot data
    plot_data(plot_conf, laser_data.ztr_data, crack_diff_map)

    return is_crack, is_corr, is_shaft_shifted


def infer17(bomb):
    """
    탄체 균열 또는 뒤틀림, 중부식
    """
    logmg.i.log("# 탄체 균열 또는 뒤틀림, 중부식")
    conf = laser_conf.infer_15_17

    data_file = bomb.sensor_data_path
    plot_config = {
        'html': rep_sl_join(DIR_PATH['done']['sensor']['html'],
                            '{}_{}.html'.format(bomb.lot.name, bomb.num)),
        'image': rep_sl_join(DIR_PATH['done']['sensor']['image'],
                             '{}_{}.png'.format(bomb.lot.name, bomb.num)),
        'unit': conf['plot']['unit']
    }

    # try:
    is_crack, is_corr, is_shaft_shifted = run(data_file, threshold=conf['threshold'],
                                              plot_conf=plot_config, phys_conf=conf['phys'])
    # except Exception as e:
    #     is_crack = True
    #     is_corr = True
    #     is_shaft_shifted = True

    #     plot_config['html_file'] = ''
    #     plot_config['img_file'] = ''
    #     print(e)
    #     time.sleep(1000)

    bomb.sensor_res_path = {
        'html': get_abs_path(plot_config['html']),
        'image': get_abs_path(plot_config['image'])
    }

    result1 = "정상"
    result2 = "정상"
    is_ok = True
    is_crack_shift = False
    # if is_crack or is_shaft_shifted:
    #     bomb.defect['body']['bot']['res'][6].append(DEFECT_CODE['body']['bot']['crack_shift'])
    #     is_ok = False
    #     is_crack_shift = True
    #     result1 = "결함"
    if is_corr:
        bomb.defect['body']['bot']['res'][6].append(DEFECT_CODE['body']['bot']['heavy_corr'])
        is_ok = False
        result2 = "결함"

    # bomb.update_infer_stat('body', 'crack_shift', not is_crack_shift)
    bomb.update_infer_stat('body', 'heavy_corr', is_ok)

    # logmg.i.log("탄체 균열 또는 뒤틀림 결과 : %s", result1)
    logmg.i.log("탄체 중부식 결과 : %s", result2)
    return is_ok
