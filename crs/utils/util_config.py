import os
import sys

from omegaconf import OmegaConf


def setup_config():
    args = sys.argv

    config_file_name = args[1]
    config_file_path = f"crs/config/{config_file_name}.yaml"
    if os.path.exists(config_file_path):
        cfg = OmegaConf.load(config_file_path)
    else:
        raise FileNotFoundError("No YAML file !!!" + config_file_path)

    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args_list=args[2:]))

    config_name_comp = {"override_cmd": args[2:]}
    cfg = OmegaConf.merge(cfg, config_name_comp)

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    return cfg


def update_cfg(cfg):
    with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
