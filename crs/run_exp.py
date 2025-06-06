import os 
import yaml
import subprocess
from tqdm import tqdm

def run_exp(default_cfg, overwrite_data, cfg_path):
    results_base_dir_name,dir_name,grid_size,func_para,dan_ver,distance_decay_method,div_avg,vec_decay_method=overwrite_data
    over_write_cfg={"results_base_dir_name":results_base_dir_name,
                    "dir_name":dir_name, 
                    "grid_size":grid_size, 
                    "func_para":func_para, 
                    "dan_ver":dan_ver, 
                    "distance_decay_method":distance_decay_method, 
                    "div_avg":div_avg,
                    "vec_decay_method":vec_decay_method}
    cfg=default_cfg.copy()
    cfg.update(over_write_cfg)
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # run_command=f"bash danger/scripts/run_main_exp.sh {cfg_path}"
    run_command=f"sbatch danger/scripts/run_main_exp.sh {cfg_path}"
    subprocess.run(run_command, shell=True, cwd="/homes/hnakayama/P2P/P2PNet")

def set_para_div_arround_range(date,results_base_dir_name):
    grid_size=5
    dan_ver="div"
    distance_decay_method="gaussian"
    max_x_list = [round(m * 13.5, 2) for m in [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]
    div_avg=False
    clip_value=None
    vec_decay_method="exp_func"
    input_params_list=[]
    for max_x in max_x_list:
        dir_name=f"{date}_div_arround_range_{max_x}"
        func_para=[1, max_x, clip_value]
        params=[results_base_dir_name, dir_name, grid_size, func_para, dan_ver, distance_decay_method, div_avg, vec_decay_method]
        input_params_list.append(params)
    return input_params_list

def set_para_div_decay_method(date,results_base_dir_name):
    input_params_list=[]
    div_avg=False
    clip_value=None
    vec_decay_method="exp_func"
    grid_size=5
    dan_ver="div"
    max_x=20.25
    distance_decay_method_list=["gaussian", None]
    vec_decay_method_list=["exp_func", None]
    # distance_decay_method_list=["liner", "gaussian", None]
    for vec_decay_method in vec_decay_method_list:
        if vec_decay_method == "exp_func":
            vec_decay_method_name="exp_func"
        elif vec_decay_method is None:
            vec_decay_method_name="wo_vec_decay"
        for distance_decay_method in distance_decay_method_list:
            if distance_decay_method is None:
                # max_x=50
                distance_decay_method_name="wo_distance_decay"
            # elif distance_decay_method=="liner":
            #     # max_x=50
            #     distance_decay_method_name="liner_decay"
            elif distance_decay_method=="gaussian":
                # max_x=50/3
                distance_decay_method_name="gaussian_decay"
            dir_name=f"{date}_div_{vec_decay_method_name}_{distance_decay_method_name}"
            func_para=[1, max_x, clip_value]
            params=[results_base_dir_name, dir_name, grid_size, func_para, dan_ver, distance_decay_method, div_avg, vec_decay_method]
            input_params_list.append(params)

    return input_params_list

def set_para_div_default(date,results_base_dir_name):
    input_params_list=[]
    div_avg=False
    clip_value=None
    vec_decay_method="exp_func"
    grid_size=5
    dan_ver="div"
    distance_decay_method="gaussian"
    max_x=20.25
    dir_name=f"{date}_proposed_method_{max_x}"
    func_para=[1, max_x, clip_value]
    params=[results_base_dir_name, dir_name, grid_size, func_para, dan_ver, distance_decay_method, div_avg, vec_decay_method]
    input_params_list.append(params)

    max_x=13.50
    dir_name=f"{date}_proposed_method_{max_x}"
    func_para=[1, max_x, clip_value]
    params=[results_base_dir_name, dir_name, grid_size, func_para, dan_ver, distance_decay_method, div_avg, vec_decay_method]
    input_params_list.append(params)

    return input_params_list

def set_para_crowd_danger(date,results_base_dir_name):
    input_params_list=[]
    vec_decay_method=None
    clip_value=None
    div_avg=False

    # follow my method
    dir_name=f"{date}_crowd_danger_follow_div"
    grid_size=5
    dan_ver="Cd"
    distance_decay_method="gaussian"
    max_x=20.25
    func_para=[1, max_x, clip_value]
    params=[results_base_dir_name, dir_name, grid_size, func_para, dan_ver, distance_decay_method, div_avg, vec_decay_method]
    input_params_list.append(params)

    # follow original paper
    dir_name=f"{date}_crowd_danger"
    grid_size=3
    dan_ver="Cd"
    distance_decay_method="gaussian"
    max_x=20.25
    func_para=[1, max_x, clip_value]
    params=[results_base_dir_name, dir_name, grid_size, func_para, dan_ver, distance_decay_method, div_avg, vec_decay_method]
    input_params_list.append(params)

    # wider range
    dir_name=f"{date}_crowd_danger_wider_range"
    grid_size=10
    dan_ver="Cd"
    distance_decay_method="gaussian"
    max_x=20.25
    func_para=[1, max_x, clip_value]
    params=[results_base_dir_name, dir_name, grid_size, func_para, dan_ver, distance_decay_method, div_avg, vec_decay_method]
    input_params_list.append(params)

    return input_params_list

def set_para_crowd_pressure(date,results_base_dir_name):
    input_params_list=[]
    vec_decay_method=None
    clip_value=None
    div_avg=False

    # follow my method
    dir_name=f"{date}_crowd_pressure_follow_div"
    grid_size=5
    dan_ver="crowd_pressure"
    distance_decay_method="gaussian"
    max_x=20.25
    func_para=[1, max_x, clip_value]
    params=[results_base_dir_name, dir_name, grid_size, func_para, dan_ver, distance_decay_method, div_avg, vec_decay_method]
    input_params_list.append(params)

    # follow original paper
    dir_name=f"{date}_crowd_pressure"
    grid_size=5
    dan_ver="crowd_pressure"
    distance_decay_method="gaussian"
    max_x=13.5
    func_para=[1, max_x, clip_value]
    params=[results_base_dir_name, dir_name, grid_size, func_para, dan_ver, distance_decay_method, div_avg, vec_decay_method]
    input_params_list.append(params)

    return input_params_list

def main_div_decay_method(start_num,date,results_base_dir_name):
    default_cfg=yaml.safe_load(open("danger/config_exp/config_default.yaml", "r"))
    params=set_para_div_decay_method(date,results_base_dir_name)
    for i, param in enumerate(params):
        exp_num=start_num+i+1
        run_exp(default_cfg, param, f"danger/config_exp/config_{date}_{exp_num}.yaml")
    return exp_num

def main_div_arround_range(start_num,date,results_base_dir_name):
    default_cfg=yaml.safe_load(open("danger/config_exp/config_default.yaml", "r"))
    params=set_para_div_arround_range(date,results_base_dir_name)
    for i, param in enumerate(params):
        exp_num=start_num+i+1
        run_exp(default_cfg, param, f"danger/config_exp/config_{date}_{exp_num}.yaml")
    return exp_num

def main_crowd_danger(start_nums,date,results_base_dir_name):
    default_cfg=yaml.safe_load(open("danger/config_exp/config_default.yaml", "r"))
    params=set_para_crowd_danger(date,results_base_dir_name)
    for i, param in enumerate(params):
        exp_num=start_nums+i+1
        run_exp(default_cfg, param, f"danger/config_exp/config_{date}_{exp_num}.yaml")
    return exp_num

def main_crowd_pressure(start_nums,date,results_base_dir_name):
    default_cfg=yaml.safe_load(open("danger/config_exp/config_default.yaml", "r"))
    params=set_para_crowd_pressure(date,results_base_dir_name)
    for i, param in enumerate(params):
        exp_num=start_nums+i+1
        run_exp(default_cfg, param, f"danger/config_exp/config_{date}_{exp_num}.yaml")
    return exp_num

def main_div_default(start_nums,date,results_base_dir_name):
    default_cfg=yaml.safe_load(open("danger/config_exp/config_default.yaml", "r"))
    params=set_para_div_default(date,results_base_dir_name)
    for i, param in enumerate(params):
        exp_num=start_nums+i+1
        run_exp(default_cfg, param, f"danger/config_exp/config_{date}_{exp_num}.yaml")


def main_exp_v1():
    start_num=0
    date="0419"
    results_base_dir_name="results_arround_range"
    exp_num=main_div_arround_range(start_num,date,results_base_dir_name)

def main_exp_v2():
    date="0419"
    exp_num=0
    results_base_dir_name="results_comparison_method"
    exp_num=main_crowd_danger(exp_num,date,results_base_dir_name)
    exp_num=main_crowd_pressure(exp_num,date,results_base_dir_name)
    exp_num=main_div_default(exp_num,date,results_base_dir_name)

def main_exp_v3():
    date="0419"
    exp_num=0
    results_base_dir_name="results_ablation_study_v2"
    exp_num=main_div_decay_method(exp_num,date,results_base_dir_name)

def main_exp_v4():
    date="0502"
    exp_num=0
    results_base_dir_name="results_comparison_method_fin"
    exp_num=main_crowd_danger(exp_num,date,results_base_dir_name)
    exp_num=main_div_default(exp_num,date,results_base_dir_name)

if __name__=="__main__":
    # main_exp_v1()
    # main_exp_v2()
    # main_exp_v3()
    main_exp_v4()
