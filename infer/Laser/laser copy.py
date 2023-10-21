import copy
from typing import Union, Callable, Tuple

import plotly.graph_objects as go
from plotly.offline import plot

from conf import *
from util import *

import infer.Laser.laser_conf as laser_conf
import logs.logmg as logmg


def get_data_from_file(file_path: str,
                       dist_to_axis: float,
                       sec_per_rev: float = 18.0,
                       bias: float = 0) -> Union[dict, None]:
    """
    Get data of a revolution from file.

    :param file_path: path of data file(*.csv)
    :param dist_to_axis: distance from sensor to rotational axis
    :param sec_per_rev: seconds per a revolution [sec]
    :param bias: bias time [sec]
    :return: dictionary form cylindrical coordinate data of vertices
            rev_ztr['z'] : z values of vertices [m]
            rev_ztr['theta'] : theta values of vertices [rad]
            rev_ztr['r'] : radial values of vertex [m]
                           in r = r(theta, z) form,
                           r(rev_ztr['theta'][i], rev_ztr['z'][j]) = rev_ztr['r'][i][j]
    """

    # Get data of a revolution
    ztr = get_ztr_from_file(file_path, dist_to_axis)
    tm_a = time.time_ns()
    rev_ztr = get_rev_data(ztr, sec_per_rev, bias)
    tm_b = time.time_ns()

    # Convert time to theta[rad]
    rev_ztr['theta'] = 2 * np.pi * rev_ztr['t'] / rev_ztr['rev_t']
    del rev_ztr['rev_t']

    return rev_ztr


def get_ztr_from_file(file_path: str, dist_to_axis: float) -> dict:
    """
    Read data from data file and return dictionary that have z-pos, elapsed time and surface radius data arrays

    :param file_path: path of file (both of absolute and relative path can be used)
    :param dist_to_axis: distance from sensor to rotation axis
    :return: dictionary of datas
    """

    # Drop last 2 cols and last 3 rows (To remove outliers)

    # Convert csv row to float list
    def csv_line_to_list(csv_line, to_ndarray=True):
        res = [float(val) for val in csv_line.split(',')[:-1]][:-2]  # Drop last 2 cols
        if to_ndarray:
            return np.array(res, dtype=np.float32)
        return res

    # Read lines from file
    file = open(file_path)
    f_lines = file.readlines()[:-3]  # Drop last 3 rows
    file.close()

    # Read z values
    z = 0.001 * csv_line_to_list(f_lines[0])

    # Read time values
    t = []
    for line in f_lines[1::3]:
        t.append(csv_line_to_list(line, to_ndarray=False)[0])
    t = np.array(t)

    # Read radius values
    dist_to_surface = []
    for line in f_lines[2::3]:
        dist_to_surface.append(csv_line_to_list(line, to_ndarray=False))
    dist_to_surface = 0.001 * np.array(dist_to_surface)  # Convert [mm] to [m]
    r = dist_to_axis - dist_to_surface

    if np.min(r) <= 0:
        raise Exception('Invalid axis distance (too small)')

    return {'z': z, 't': t, 'r': r}


def fill_zeros(ztr: dict):
    """
    Do linear interpolate to remove(fill) zeros

    :param ztr:
    :return:
    """

    n_row = len(ztr['r'])
    n_col = len(ztr['r'][0])
    for i in range(n_row):
        row_data = ztr['r'][i]

        st_idx = None
        idx = 0
        while idx != n_col:
            if row_data[idx] == 0 and st_idx is None:
                st_idx = idx - 1
            elif row_data[idx] != 0 and st_idx is not None:
                chg = 0 if st_idx == -1 else (row_data[idx] - row_data[st_idx]) / (idx - st_idx)
                for _idx in range(st_idx + 1, idx):
                    ztr['r'][i][_idx] = row_data[idx] - chg * (idx - _idx)
                st_idx = None

            idx += 1

        if st_idx is not None:
            for _idx in range(st_idx + 1, n_col):
                ztr['r'][i][_idx] = ztr['r'][i][st_idx]


def get_rev_data(ztr: dict, sec_per_rev: float = 18.0, bias: float = 0) -> Union[dict, None]:
    """
    Extract ztr data of a revolution

    :param ztr: ztr data returned from get_ztr_from_file()
    :param sec_per_rev: second per revolution [sec]
    :param bias: bias time [sec]
    :return: ztr(+rev_t) data of a revolution
    """

    # Check is argument valid
    if bias > ztr['t'][-1]:
        raise Exception('Invalid bias')

    n_data = len(ztr['t'])
    res = dict()

    st_idx = None
    for i in range(n_data):
        if ztr['t'][i] >= bias:
            st_idx = i
            break
    bias = ztr['t'][st_idx]

    en_idx = None
    for i in range(st_idx + 1, n_data):
        if ztr['t'][i] > bias + sec_per_rev:
            en_idx = i - 1
            break

        if i == n_data - 1:
            en_idx = i

    # Redef second per revolution
    if en_idx != n_data - 1:
        res['rev_t'] = ztr['t'][en_idx]
    else:
        res['rev_t'] = sec_per_rev

    res['z'] = copy.deepcopy(ztr['z'])
    res['t'] = ztr['t'][st_idx:en_idx + 1] - bias
    res['r'] = ztr['r'][st_idx:en_idx + 1]
    fill_zeros(res)

    return res


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


def linear_interpolate_theta(data: dict, max_d_l: float) -> dict:
    """
    Create theta datas using linear interpolation

    :param data: cylindrical coordinate data
    :param max_d_l: maximum theta-direction length between two datas [m]
    :return: linear interpolated cylindrical data
    """

    res = dict()
    res['z'] = copy.deepcopy(data['z'])

    deltas = get_cylindrical_deltas(data)
    n_div = int(np.ceil(deltas['d_l'][1] / max_d_l))

    theta = [data['theta'][0]]
    for i in range(1, len(data['theta'])):
        prev_theta = data['theta'][i - 1]
        next_theta = data['theta'][i]
        for j in range(1, n_div + 1):
            theta.append(prev_theta + (next_theta - prev_theta) * j / n_div)
    res['theta'] = np.array(theta)

    r = [data['r'][0]]
    for i in range(1, len(data['theta'])):
        prev_r = data['r'][i - 1]
        next_r = data['r'][i]
        for j in range(1, n_div + 1):
            r.append(prev_r + (next_r - prev_r) * j / n_div)
    res['r'] = np.array(r)

    return res


def reduce_z(data: dict, max_d_z: float, func: Callable) -> dict:
    res = dict()
    res['theta'] = copy.deepcopy(data['theta'])

    idx_group = []
    idx_buf = 0
    for i in range(len(data['z'])):
        if data['z'][i] - data['z'][idx_buf] > max_d_z:
            if i - 2 >= idx_buf:
                idx_group.append([idx_buf, i - 2])
                idx_buf = i - 1
            else:
                idx_group.append([idx_buf, i - 1])
                idx_buf = i

    z = []
    for st, en in idx_group:
        z.append(data['z'][st])
    res['z'] = np.array(z)

    r = []
    for st, en in idx_group:
        r.append(func(data['r'][:, st:en + 1], axis=1))
    res['r'] = np.transpose(np.array(r))

    return res


def get_diff_map(data: dict,
                 sz_wnd: Tuple[float, float],
                 q_perc: Tuple[float, float] = (25, 75),
                 func: Callable = np.min) -> np.ndarray:
    buf = np.zeros(data['r'].shape)
    n_theta, n_z = data['r'].shape

    deltas = get_cylindrical_deltas(data)
    th_idx_rng = int((sz_wnd[0] / 2) / deltas['d_l'][1])
    z_idx_rng = int((sz_wnd[1] / 2) / deltas['d_z'][1])

    for i in range(n_theta):
        for j in range(n_z):
            theta_st = i - th_idx_rng if i > th_idx_rng else 0
            theta_en = i + th_idx_rng if i + th_idx_rng < n_theta else n_theta
            z_st = j - z_idx_rng if j > z_idx_rng else 0
            z_en = j + z_idx_rng if j + z_idx_rng < n_z else n_z

            test_rect = data['r'][theta_st:theta_en, z_st:z_en]
            buf[i][j] = func(test_rect)

    q1, q3 = np.percentile(buf, q_perc, axis=0)
    base_value = np.mean([q1, q3], axis=0)

    res = buf - base_value

    return res


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


def plot_data(out_path: str, data: dict, diff_map: np.ndarray, threshold: float):
    ca_data = cylindrical_to_cartesian(data)
    color_base = np.transpose(diff_map)

    plot_diff_3d(out_path, ca_data, color_base)
    # plot_bool_3d(out_path, ca_data, diff_map, color_base)
    plot_scatter_3d(out_path, data, diff_map)


def plot_diff_3d(out_path: str, ca_data: dict, color_base: np.ndarray):
    diff_color = copy.deepcopy(color_base)
    diff_color[diff_color > 0] = 0
    fig = go.Figure(
        go.Surface(x=ca_data['x'], y=ca_data['y'], z=ca_data['z'], surfacecolor=diff_color)
    )
    plot(fig, filename=out_path + 'diff_3D.html', auto_open=False)


'''
def plot_bool_3d(out_path, ca_data, diff_map, color_base):
    fig = go.Figure(
        go.Surface(x=ca_data['x'], y=ca_data['y'], z=ca_data['z'], surfacecolor=defect_bool,
                   colorscale=[[0, 'red' if is_defected(diff_map, threshold) else 'green'], [1, 'green']])
    )
    plot(fig, filename=out_path + 'bool_3D.html', auto_open=False)
'''


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
    else:  # not is_crack and is_corr
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
        max_d_l: float = 0.2e-3, max_d_z: float = 0.2e-3,
        sz_diff_wnd: Union[tuple, list] = (1e-3, 1e-3)):
    # Get data from file
    orig_cy_data = get_data_from_file(data_path,
                                      dist_to_axis=phys_conf['dist_to_axis'],
                                      sec_per_rev=phys_conf['sec_per_rev'], bias=phys_conf['bias'])
    # add_test_patch(orig_cy_data, 1e-2, 0.5e-2, thickness=-0.1e-3)

    # Interpolate to fill theta values
    cy_data = linear_interpolate_theta(orig_cy_data, max_d_l=max_d_l)

    # Find cracks
    crack_cy_data = reduce_z(cy_data, max_d_z=max_d_z, func=np.min)
    crack_diff_map = get_diff_map(crack_cy_data, sz_wnd=sz_diff_wnd, func=np.min)
    crack_bool = crack_diff_map < -threshold['crack']

    # Find foreign objects
    fo_cy_data = reduce_z(cy_data, max_d_z=max_d_z, func=np.max)
    fo_diff_map = get_diff_map(fo_cy_data, sz_wnd=sz_diff_wnd, func=np.max)
    fo_bool = fo_diff_map > threshold['corr']

    # Detect defects
    is_crack = is_defected(crack_diff_map, threshold=-threshold['crack'], over_th=False)
    is_corr = is_defected(fo_diff_map, threshold=threshold['corr'], over_th=True)

    # Create result bool map
    diff_shape = crack_diff_map.shape
    _res_bool = np.zeros(diff_shape)
    _res_bool[crack_bool] = -1  # Crack = -1
    _res_bool[fo_bool] = 1  # FO = 1
    res_bool = np.transpose(_res_bool)

    # Detect shaft shift
    crack_cy_data['r'][_res_bool != 0] = np.NAN
    r_min = np.nanmin(crack_cy_data['r'], axis=0)
    fo_cy_data['r'][_res_bool != 0] = np.NAN
    r_max = np.nanmax(fo_cy_data['r'], axis=0)

    r_diff_max = np.max(r_max - r_min)
    is_shaft_shifted = r_diff_max > threshold['shaft']

    # Plot bool 3d
    plot_bool_3d(plot_conf['html_file'], plot_conf['img_file'], crack_cy_data, is_crack, is_corr, res_bool,
                 plot_conf['unit'])

    return is_crack, is_corr, is_shaft_shifted


def infer_15_17(bomb):
    """
    탄체 균열 또는 뒤틀림, 중부식
    """
    logmg.i.log("# 탄체 균열 또는 뒤틀림, 중부식")
    conf = laser_conf.infer_15_17

    data_file = bomb.sensor_data_path
    plot_config = {
        'html_file': rep_sl_join(DIR_PATH['done']['sensor']['html'],
                                 '{}_{}.html'.format(bomb.lot.name, bomb.num)),
        'img_file': rep_sl_join(DIR_PATH['done']['sensor']['image'],
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
        'html': get_abs_path(plot_config['html_file']),
        'image': get_abs_path(plot_config['img_file'])
    }
    logmg.i.log("-------------------------------------------",bomb.sensor_res_path)
    result1 = "정상"
    result2 = "정상"
    is_ok = True
    is_crack_shift = False
    if is_crack or is_shaft_shifted:
        bomb.defect['body']['bot']['res'][6].append(DEFECT_CODE['body']['bot']['crack_shift'])
        is_ok = False
        is_crack_shift = True
        result1 = "결함"
    if is_corr:
        bomb.defect['body']['bot']['res'][6].append(DEFECT_CODE['body']['bot']['heavy_corr'])
        is_ok = False
        result2 = "결함"

    bomb.update_infer_stat('body', 'crack_shift', not is_crack_shift)
    bomb.update_infer_stat('body', 'heavy_corr', not is_corr)
    
    logmg.i.log("탄체 균열 또는 뒤틀림 결과 : %s", result1)
    logmg.i.log("탄체 중부식 결과 : %s", result2)
    return is_ok
