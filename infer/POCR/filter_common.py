import math
import numpy as np
from numba import cuda


@cuda.jit('void(uint8[:, :, :], uint8[:, :, :], int64, float64)')
def __border_filter_cuda_kernel(src_img, res_img, sz_wnd, mult):
    idx = cuda.grid(2)
    stride = cuda.gridsize(2)

    def set_res_pixel(location, value):
        res_img[location[0]][location[1]][0] = value
        res_img[location[0]][location[1]][1] = value
        res_img[location[0]][location[1]][2] = value

    for idx_x in range(idx[1], src_img.shape[1] - sz_wnd, stride[1]):
        for idx_y in range(idx[0], src_img.shape[0] - sz_wnd, stride[0]):
            set_res_pixel((idx_y, idx_x), 255)
            if idx_x < sz_wnd or idx_y < sz_wnd:
                continue

            threshold = math.floor(src_img[idx_y][idx_x][0] * mult)
            wnd_max = 0
            for wnd_idx_x in range(idx_x - sz_wnd, idx_x + sz_wnd):
                for wnd_idx_y in range(idx_y - sz_wnd, idx_y + sz_wnd):
                    val = src_img[wnd_idx_y][wnd_idx_x][0]
                    if wnd_max < val:
                        wnd_max = val

            if wnd_max > threshold:
                set_res_pixel((idx_y, idx_x), 0)


def __apply_border_filter(src_img, sz_wnd=2, mult=1.7, on_gpu=False):
    res_img = None

    if on_gpu is True:
        n_blocks = tuple(np.array((3, 4)) * 5)  # 12 * 5 blocks = 60 blocks
        n_ths_per_block = tuple(np.array((4, 8)) * 3)  # 32 * 3 threads = 96 threads = 3 warps

        s = cuda.stream()

        d_src_img = cuda.to_device(np.ascontiguousarray(src_img), stream=s)
        d_res_img = cuda.device_array_like(src_img, stream=s)

        __border_filter_cuda_kernel[n_blocks, n_ths_per_block, s, 0](d_src_img, d_res_img, sz_wnd, mult)
        s.synchronize()

        res_img = d_res_img.copy_to_host()
    else:  # Run on CPU
        res_img = np.full(src_img.shape, 255, dtype=np.uint8)

        for i in range(sz_wnd, src_img.shape[0] - sz_wnd):
            for j in range(sz_wnd, src_img.shape[1] - sz_wnd):
                threshold = np.floor(src_img[i][j][0] * mult)
                if np.max(src_img[i-sz_wnd:i+sz_wnd+1, j-sz_wnd:j+sz_wnd+1, 0]) > threshold:
                    res_img[i][j].fill(0)

    return res_img


def apply_filters(image, mult):
    res = __apply_border_filter(image, mult=mult, on_gpu=cuda.is_available())
    return res
