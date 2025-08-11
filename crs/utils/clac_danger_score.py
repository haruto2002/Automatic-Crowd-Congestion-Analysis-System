import numpy as np
from scipy.spatial.distance import cdist
import cv2

# 流量変化
def clac_div_origin(around_data, center_pos, func_para, distance_decay, vec_decay_method=None):
    decay, max_x, clip_value = func_para
    vx_list = []
    vy_list = []
    # print("vec_decay_method: ", vec_decay_method)
    for pos, vec in around_data:
        if vec_decay_method =="exp_func":
            vx,vy=exp_decay(vec[0]),exp_decay(vec[1])
        elif vec_decay_method == "clip":
            vx,vy=clip(vec[0],vec[1],clip_value)
        elif vec_decay_method is None:
            vx,vy=vec[0],vec[1]
        else:
            raise ValueError(f"Invalid vec_decay_method: {vec_decay_method}")
        if pos[0] < center_pos[0]:
            vx = relu_like_minus(vx, decay)
        else:
            vx = relu_like_plus(vx, decay)
        if pos[1] < center_pos[1]:
            vy = relu_like_minus(vy, decay)
        else:
            vy = relu_like_plus(vy, decay)
        vx_list.append(vx)
        vy_list.append(vy)
    vx_array = np.array(vx_list)
    vy_array = np.array(vy_list)
    if distance_decay is None:
        div = np.sum(vx_array) + np.sum(vy_array)
    else:
        decay_vx_array = vx_array * distance_decay
        decay_vy_array = vy_array * distance_decay
        div = np.sum(decay_vx_array) + np.sum(decay_vy_array)
    div_danger_score=-div
    return div_danger_score

def exp_decay(v, a=12.0):
    if v > 0:
        return 0.3 * (1 - np.exp(-a * v))
    else:
        return -0.3 * (1 - np.exp(a * v))

# 回転
def clac_curl_origin(around_data, center_pos, func_para, distance_decay, vec_decay_method=None):
    decay, max_x, clip_value = func_para
    vx_list = []
    vy_list = []
    for pos, vec in around_data:
        if vec_decay_method =="exp_func":
            vx,vy=exp_decay(vec[0]),exp_decay(vec[1])
        elif vec_decay_method == "clip":
            vx,vy=clip(vec[0],vec[1],clip_value)
        elif vec_decay_method is None:
            vx,vy=vec[0],vec[1]
        else:
            raise ValueError(f"Invalid vec_decay_method: {vec_decay_method}")
        # get x方向でy成分
        if pos[0] < center_pos[0]:
            vy = relu_like_minus(vy, decay)
        else:
            vy = relu_like_plus(vy, decay)
        # get y方向でx成分
        if pos[1] < center_pos[1]:
            vx = relu_like_minus(vx, decay)
        else:
            vx = relu_like_plus(vx, decay)

        vx_list.append(vx)
        vy_list.append(vy)

    vx_array = np.array(vx_list)
    vy_array = np.array(vy_list)
    if distance_decay is None:
        curl = np.sum(vx_array) - np.sum(vy_array)
    else:
        decay_vx_array = vx_array * distance_decay
        decay_vy_array = vy_array * distance_decay
        curl = np.sum(decay_vx_array) - np.sum(decay_vy_array)
    
    curl_danger_score=abs(curl)
    return curl_danger_score

def get_vec_var(around_data, func_para, distance_decay):
    decay, max_x, clip_value = func_para
    vx_list = []
    vy_list = []
    for pos, vec in around_data:
        vx, vy = clip(vec[0], vec[1], clip_value)
        vx_list.append(vx)
        vy_list.append(vy)

    vx_array = np.array(vx_list)
    vy_array = np.array(vy_list)
    original_vec_array = np.column_stack([vx_array, vy_array])
    decay_vx_array = vx_array * distance_decay
    decay_vy_array = vy_array * distance_decay
    vec_array = np.column_stack([decay_vx_array, decay_vy_array])
    avg_vec = np.sum(vec_array,axis=0)/np.sum(distance_decay)
    diff_norms = np.sum((original_vec_array-avg_vec)**2, axis=1)*distance_decay
    vec_var = np.sum(diff_norms)/np.sum(distance_decay)
    return vec_var

# 回転×密度
def calc_div(around_data, center_pos, func_para, distance_decay_method=None, max_x=100,vec_decay_method=None):
    if distance_decay_method is None:
        density=len(around_data)
        distance_decay = None
    elif distance_decay_method == "gaussian":
        density, distance_decay = get_gaussian_kernel_density(center_pos, around_data[:,0],R=max_x)
    elif distance_decay_method == "liner":
        density, distance_decay = get_liner_decay_density(center_pos, around_data[:,0],max_x=max_x)
    div = clac_div_origin(around_data, center_pos, func_para, distance_decay, vec_decay_method=vec_decay_method)
    return div*density,density
    
# 回転×密度
def calc_curl(around_data, center_pos, func_para, distance_decay_method=None, max_x=100,vec_decay_method=None):
    if distance_decay_method is None:
        density=len(around_data)
        distance_decay = None
    elif distance_decay_method == "gaussian":
        density, distance_decay = get_gaussian_kernel_density(center_pos, around_data[:,0],R=max_x)
    elif distance_decay_method == "liner":
        density, distance_decay = get_liner_decay_density(center_pos, around_data[:,0],max_x=max_x)
    curl = clac_curl_origin(around_data, center_pos, func_para, distance_decay, vec_decay_method=vec_decay_method)
    return curl*density,density
    
def calc_crowd_pressure(around_data, center_pos, func_para, distance_decay_method="gaussian",max_x=100):
    if distance_decay_method is None:
        density=len(around_data)
        distance_decay = None
    elif distance_decay_method == "gaussian":
        density, distance_decay = get_gaussian_kernel_density(center_pos, around_data[:,0],R=max_x)
    elif distance_decay_method == "liner":
        density, distance_decay = get_liner_decay_density(center_pos, around_data[:,0],max_x=max_x)
    vec_var = get_vec_var(around_data, func_para, distance_decay)
    return vec_var*density,density



# 中心点に向かってくる力はそのまま、離れていく力は減衰させる関数
# 発散：流出量ー流入量
# 流出量
def relu_like_plus(
    data, decay=10
):  # 中心より右(xがプラス側)もしくは下(yがプラス側)の場合
    if data > 0:
        return data * decay
        # return 0
    else:
        return data


# 流入量
def relu_like_minus(
    data, decay=10
):  # 中心より左(xがマイナス側)もしくは上(yがマイナス側)の場合
    if data < 0:
        return -data * decay
        # return 0
    else:
        return -data

# 速度の大きさに上限を設定
def clip(vx, vy, clip_value=0.5):
    if clip_value is None:
        return vx, vy
    return min(vx, clip_value), min(vy, clip_value)

def get_gaussian_kernel_density(eval_point,positions, R=0.3):
    if len(positions)==0:
        return 0,None
    if positions.ndim == 1:
        positions = np.expand_dims(positions, axis=0)
    # 距離行列: 各グリッド点から歩行者までの距離 (H*W, N)
    distances = cdist(np.expand_dims(eval_point, axis=0), positions)  # shape: (1, N)

    # Gaussian Kernel の適用
    kernel_vals = np.exp(-0.5 * (distances / R)**2) / (2 * np.pi * R**2)  # (1, N)

    # 各グリッドにおける密度：各人からの影響を合計
    density = np.sum(kernel_vals[0])

    return density, kernel_vals[0]

def get_liner_decay_density(eval_point,positions, max_x=100):
    if len(positions)==0:
        return 0,None
    if positions.ndim == 1:
        positions = np.expand_dims(positions, axis=0)
    # 距離行列: 各グリッド点から歩行者までの距離 (H*W, N)
    distances = cdist(np.expand_dims(eval_point, axis=0), positions)  # shape: (1, N)

    decay_vals=[distance_liner_decay(d, max_x) for d in distances[0]]

    # 各グリッドにおける密度：各人からの影響を合計
    density = np.sum(decay_vals)

    return density, decay_vals

# 中心からの距離で減衰
def distance_liner_decay(d, max_x=100):
    max_y = 1  # 切片
    # max_x = 200  # x軸との交点
    if d < max_x:
        return (-max_y / max_x) * d + max_y
    else:
        return 0



def get_map_data(size, vec_data, grid_size, func_para, dan_ver=None, distance_decay_method=None,vec_decay_method=None):
    # vec_data.shape >> (num_people, pos(2), vec(2))
    refine = False
    if refine:
        refine_size = 5
        assert grid_size % refine_size == 0
        grid_size = grid_size // refine_size
    x_size, y_size = size
    assert x_size % grid_size == 0 and y_size % grid_size == 0
    x_bins = np.arange(0, x_size + 1, grid_size)
    y_bins = np.arange(0, y_size + 1, grid_size)
    danger_map = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
    density_map = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
    decay, max_x, clip_value = func_para
    if distance_decay_method == "gaussian":
        around_range = max_x*3
    elif distance_decay_method == "liner" or distance_decay_method is None:
        around_range = max_x
    else:
        raise ValueError(f"Invalid distance_decay_method: {distance_decay_method}")
    for i in range(len(y_bins) - 1):
        for j in range(len(x_bins) - 1):
            center_pos = np.array(
                [
                    grid_size * (j + 1 / 2),
                    grid_size * (i + 1 / 2),
                ]
            )
            around_data = vec_data[
                (vec_data[:, 0, 0] < center_pos[0] + around_range)
                & (vec_data[:, 0, 0] > center_pos[0] - around_range)
                & (vec_data[:, 0, 1] < center_pos[1] + around_range)
                & (vec_data[:, 0, 1] > center_pos[1] - around_range)
            ]

            if len(around_data) == 0:
                danger_map[i, j] = 0
                density_map[i, j] = 0
                continue

            if dan_ver == "div":
                div, density = calc_div(around_data, center_pos, func_para,distance_decay_method=distance_decay_method, max_x=max_x,vec_decay_method=vec_decay_method)
            elif dan_ver == "curl":
                div, density = calc_curl(around_data, center_pos, func_para,distance_decay_method=distance_decay_method, max_x=max_x,vec_decay_method=vec_decay_method)
            elif dan_ver == "crowd_pressure":
                div, density = calc_crowd_pressure(around_data, center_pos, func_para,distance_decay_method=distance_decay_method, max_x=max_x)
            else:
                raise ValueError(f"Invalid dan_ver: {dan_ver}")
            danger_map[i, j] = div
            density_map[i, j] = density

    if refine:
        danger_map = cv2.resize(
            danger_map,
            (danger_map.shape[0] // refine_size, danger_map.shape[1] // refine_size),
            interpolation=cv2.INTER_LINEAR,
        ) 
        density_map = cv2.resize(
            density_map,
            (density_map.shape[0] // refine_size, density_map.shape[1] // refine_size),
            interpolation=cv2.INTER_LINEAR,
        ) 
    return danger_map, density_map


def get_grid_vec_data(size, vec_data_stack_list, grid_size=3, distance_decay_method=None, R=0.3):
    # vec_data_stack_list.shape >> (stack_frame_num, num_people, pos(2), vec(2))
    x_size, y_size = size
    # assert x_size % grid_size == 0 and y_size % grid_size == 0
    x_bins = np.arange(0, x_size + 1, grid_size)
    y_bins = np.arange(0, y_size + 1, grid_size)
    grid_vec_data = np.zeros((len(y_bins) - 1, len(x_bins) - 1, 2, 2))
    decay_density_map = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
    # grid_vec_data >> (y_size, x_size, pos(2), vec(2))
    around_range = R*3
    for i in range(len(y_bins) - 1):
        for j in range(len(x_bins) - 1):
            # グリッドの中心座標を計算
            center_pos = np.array(
                [
                    x_bins[j] + grid_size * 0.5,
                    y_bins[i] + grid_size * 0.5,
                ]
            )
            for k, vec_data in enumerate(vec_data_stack_list):
                grid_data = vec_data[
                    (vec_data[:, 0, 0] < center_pos[0] + grid_size * 0.5)
                    & (vec_data[:, 0, 0] >= center_pos[0] - grid_size * 0.5)
                    & (vec_data[:, 0, 1] < center_pos[1] + grid_size * 0.5)
                    & (vec_data[:, 0, 1] >= center_pos[1] - grid_size * 0.5)
                ]
                if k == 0:
                    grid_data_stack = grid_data
                else:
                    grid_data_stack = np.concatenate([grid_data_stack, grid_data], axis=0)

            if len(grid_data_stack) == 0:
                avg_vec = np.array([np.nan, np.nan])
            else:
                avg_vec = np.mean(grid_data_stack[:, 1, :], axis=0)
            grid_vec_data[i, j, 0] = center_pos
            grid_vec_data[i, j, 1] = avg_vec

            density = 0
            for vec_data in vec_data_stack_list:
                around_data = vec_data[
                    (vec_data[:, 0, 0] < center_pos[0] + around_range)
                    & (vec_data[:, 0, 0] > center_pos[0] - around_range)
                    & (vec_data[:, 0, 1] < center_pos[1] + around_range)
                    & (vec_data[:, 0, 1] > center_pos[1] - around_range)
                ]
                if distance_decay_method is None:
                    density += len(around_data)
                elif distance_decay_method == "gaussian":
                    decay_density, kernel_vals = get_gaussian_kernel_density(center_pos, around_data[:,0], R=R)
                    density += decay_density
                elif distance_decay_method == "liner":
                    decay_density, kernel_vals = get_liner_decay_density(center_pos, around_data[:, 0], max_x=R)
                    density += decay_density
                   
            decay_density_map[i, j] = density/len(vec_data_stack_list)
    return grid_vec_data, decay_density_map


def get_curl_map(grid_vec_data, grid_size):
    map_data = np.zeros((grid_vec_data.shape[0], grid_vec_data.shape[1]))
    for i in range(grid_vec_data.shape[0]):
        for j in range(grid_vec_data.shape[1]):
            if (
                i == 0
                or j == 0
                or i == grid_vec_data.shape[0] - 1
                or j == grid_vec_data.shape[1] - 1
            ):
                curl = np.nan
            else:
                # x方向のy成分比較
                right = grid_vec_data[i, j + 1, 1]
                left = grid_vec_data[i, j - 1, 1]
                if np.any(right == np.nan) or np.any(left == np.nan):
                    x_curl = 0
                else:
                    x_curl = (right[0] - left[0]) / (2 * grid_size)
                # y方向のx成分比較
                up = grid_vec_data[i - 1, j, 1]
                down = grid_vec_data[i + 1, j, 1]
                if np.any(up == np.nan) or np.any(down == np.nan):
                    y_curl = 0
                else:
                    y_curl = (down[1] - up[1]) / (2 * grid_size)

                if (np.any(right == np.nan) or np.any(left == np.nan)) and (
                    np.any(up == np.nan) or np.any(down == np.nan)
                ):
                    curl = np.nan
                else:
                    curl = x_curl + y_curl
            map_data[i, j] = curl

    return map_data


def get_CL_map(curl_map, grid_vec_data, roi_size=3):
    CL_map = np.zeros_like(curl_map)
    for i in range(curl_map.shape[0]):
        for j in range(curl_map.shape[1]):
            if np.isnan(
                curl_map[
                    max(0, i - roi_size) : min(curl_map.shape[0], i + roi_size + 1),
                    max(0, j - roi_size) : min(curl_map.shape[1], j + roi_size + 1),
                ]
            ).all():
                CL_map[i, j] = 0
            else:
                # 周辺7x7のマスの中の最大と最小を取得（nanを除外）
                max_curl = np.nanmax(
                    curl_map[
                        max(0, i - roi_size) : min(curl_map.shape[0], i + roi_size + 1),
                        max(0, j - roi_size) : min(curl_map.shape[1], j + roi_size + 1),
                    ]
                )
                min_curl = np.nanmin(
                    curl_map[
                        max(0, i - roi_size) : min(curl_map.shape[0], i + roi_size + 1),
                        max(0, j - roi_size) : min(curl_map.shape[1], j + roi_size + 1),
                    ]
                )
                # 周辺7x7のマスの中の速度の平均のノルム
                vec_in_roi = grid_vec_data[
                    max(0, i - roi_size) : min(
                        grid_vec_data.shape[0], i + roi_size + 1
                    ),
                    max(0, j - roi_size) : min(
                        grid_vec_data.shape[1], j + roi_size + 1
                    ),
                    1,
                ]
                # 3次元配列を2次元配列に変換
                vec_in_roi_flat = vec_in_roi.reshape(-1, 2)
                # nanを含まない有効なベクトルのみを抽出
                valid_vecs = vec_in_roi_flat[~np.isnan(vec_in_roi_flat).any(axis=1)]

                if len(valid_vecs) > 0:
                    # 有効なベクトルのノルムの平均を計算
                    avg_vec_norm = np.mean(np.linalg.norm(valid_vecs, axis=1))
                else:
                    avg_vec_norm = 0

                # 周辺7x7のマスの中の速度の平均のノルムが0の場合は0を返す
                if avg_vec_norm == 0:
                    CL_map[i, j] = 0
                else:
                    CL_map[i, j] = (max_curl - min_curl) / avg_vec_norm

    return CL_map


def calc_Cd_map(size, vec_data_stack_list, grid_size, roi_size=3, distance_decay_method=None, max_x=100):
    grid_vec_data, decay_density_map = get_grid_vec_data(size, vec_data_stack_list, grid_size=grid_size, distance_decay_method=distance_decay_method, R=max_x)
    # display_vec_data(grid_vec_data,grid_size)
    curl_map = get_curl_map(grid_vec_data, grid_size)
    CL_map = get_CL_map(curl_map, grid_vec_data, roi_size)
    assert decay_density_map.shape == CL_map.shape
    Cd_map = CL_map * decay_density_map
    return Cd_map,decay_density_map,grid_vec_data

def display_vec_data(grid_vec_data,grid_size):
    img=np.zeros((1100,1500,3),dtype=np.uint8)+255
    for i in range(0, img.shape[0], grid_size):
        cv2.line(img, (0, i), (img.shape[1], i), (200, 200, 200), 1)
    for j in range(0, img.shape[1], grid_size):
        cv2.line(img, (j, 0), (j, img.shape[0]), (200, 200, 200), 1)
    vec_data=grid_vec_data.reshape(-1,2,2)
    for pos,vec in vec_data:
        if np.isnan(vec[1]).all():
            continue
        x,y=pos
        vec_x,vec_y=vec*100
        cv2.arrowedLine(img,(int(x),int(y)),(int(x+vec_x),int(y+vec_y)),(0,255,0),2)
    cv2.imwrite("vec_data.png",img)
    # return img