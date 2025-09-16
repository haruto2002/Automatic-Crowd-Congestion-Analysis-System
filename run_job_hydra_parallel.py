import hydra
from omegaconf import DictConfig, OmegaConf
import subprocess
import sys
import os
import logging
import shlex
import json
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config_parallel")
def main(cfg: DictConfig) -> None:
    """Main function - Load configuration with Hydra and execute pipeline"""
    # Set log level
    log_level = getattr(cfg.execution, "log_level", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level))

    # Validate configuration
    if not validate_config(cfg):
        logger.error("Configuration validation failed")
        sys.exit(1)

    # Execute pipeline
    all_script_path = create_parallel_script_with_config(cfg)

    logger.info(f"Main Script Path: {all_script_path}")

    run_cmd = f"bash {all_script_path}"
    # subprocess.run(run_cmd, shell=True)

    logger.info(f"'{run_cmd}' has been executed")


def create_parallel_script_with_config(cfg: DictConfig) -> str:
    io_info_cfg = cfg.io_info
    execution_cfg = cfg.execution
    pipeline_cfg = cfg.pipeline

    time_log_file = os.path.join(cfg.result_save_dir, "time_log.txt")

    os.makedirs(cfg.result_save_dir, exist_ok=True)

    logger.info("=== Creating IO info file ===")

    create_IO_info_file(io_info_cfg)

    all_IO_info_file_path = os.path.join(
        io_info_cfg.setting_execution_args.SAVE_DIR, "IO_info.json"
    )

    job_list = OmegaConf.to_container(execution_cfg, resolve=True).keys()
    for job in job_list:
        job_cfg = execution_cfg[job]
        logger.info(
            f"=== {job.upper()} (steps: {', '.join(map(str, job_cfg.steps))}, "
            + f"node_type: {job_cfg.node_type}, "
            + f"parallel: {'Parallel' if job_cfg.parallel else 'Single'}) ==="
        )
        if job_cfg.parallel:
            cmd_list = []
            node_id = 1
            IO_info_file_path = os.path.join(all_IO_info_file_path)
            for step in job_cfg.steps:
                step_config = pipeline_cfg[step]
                script = step_config.script
                step_inputs = step_config.inputs

                if not os.path.exists(script):
                    logger.error(f"Script {script} not found")
                    raise FileNotFoundError(script)

                cmd = (
                    ["bash", script]
                    + [
                        (
                            str(IO_info_file_path)
                            if p == "IO_info_file"
                            else str(cfg.settings[p])
                        )
                        for p in step_inputs
                    ]
                    + [str(job_cfg.node_type)]
                    + [str(job_cfg.log_level)]
                )
                cmd_list.append(cmd)

            create_run_script(cfg, job, cmd_list, node_id, time_log_file)

        else:
            num_node = job_cfg.num_use_node
            IO_info_file_list = separate_IO_info_file(
                cfg, all_IO_info_file_path, job, num_node
            )
            for i, IO_info_file_path in enumerate(IO_info_file_list):
                cmd_list = []
                node_id = i + 1
                for step in job_cfg.steps:
                    step_config = pipeline_cfg[step]
                    script = step_config.script
                    step_inputs = step_config.inputs

                    if not os.path.exists(script):
                        logger.error(f"Script {script} not found")
                        raise FileNotFoundError(script)

                    cmd = (
                        ["bash", script]
                        + [
                            (
                                str(IO_info_file_path)
                                if p == "IO_info_file"
                                else str(cfg.settings[p])
                            )
                            for p in step_inputs
                        ]
                        + [str(job_cfg.node_type)]
                        + [str(job_cfg.log_level)]
                    )
                    cmd_list.append(cmd)
                create_run_script(cfg, job, cmd_list, node_id, time_log_file)

    logger.info("=== Creating all run script ===")
    all_script_path = create_all_run_script(cfg, job_list, time_log_file)

    return all_script_path


def validate_config(cfg: DictConfig) -> bool:
    """Validate configuration"""
    try:
        # Check required sections
        required_sections = ["settings", "execution", "pipeline"]
        for section in required_sections:
            if not hasattr(cfg, section):
                logger.error(
                    f"Required section '{section}' not found in configuration file"
                )
                return False

        # Check input file existence
        settings = cfg.settings
        if not os.path.exists(settings.detection_weight_path):
            logger.error(f"Weight file not found: {settings.detection_weight_path}")
            return False

        logger.info("Configuration validation completed")
        return True

    except Exception as e:
        logger.error(f"Error occurred during configuration validation: {e}")
        return False


def create_IO_info_file(io_info_cfg):
    try:
        cmd = ["python", io_info_cfg.setting_execution_file] + [
            "--" + arg + "=" + value
            for arg, value in OmegaConf.to_container(
                io_info_cfg.setting_execution_args, resolve=True
            ).items()
        ]
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            text=True,
            capture_output=True,
            check=True,
            timeout=None,
        )
        if result.stdout:
            logger.info(result.stdout.rstrip())
        if result.stderr:
            for line in result.stderr.splitlines():
                if line.startswith("INFO:"):
                    logger.info(line)
                elif line.startswith("WARNING:"):
                    logger.warning(line)
                else:
                    logger.error(line)

    except subprocess.CalledProcessError as e:
        if e.stdout:
            logger.info(e.stdout.rstrip())
        if e.stderr:
            logger.error(e.stderr.rstrip())
        raise


def separate_IO_info_file(cfg, all_IO_info_file_path, job_name, num_node):
    IO_info_dict = json.load(open(all_IO_info_file_path, "r"))

    items = list(IO_info_dict.items())
    n = len(items) // num_node
    if n == 0:
        n = 1
        num_node = len(items)
        logger.warning(f"num_node is too large, set num_node to {num_node}")
    each_node_IO_info_dict_list = [
        dict(items[i * n : (i + 1) * n]) for i in range(num_node)
    ]

    save_dir = os.path.join(cfg.settings.IO_info_dir, job_name)
    os.makedirs(save_dir, exist_ok=True)
    IO_info_file_path_list = []
    for i in range(num_node):
        node_id = i + 1
        IO_info_file_path = os.path.join(save_dir, f"IO_info_{node_id:02d}.json")
        IO_info_file_path_list.append(IO_info_file_path)
        with open(IO_info_file_path, "w") as f:
            json.dump(each_node_IO_info_dict_list[i], f, indent=4)
    return IO_info_file_path_list


def create_run_script(cfg, job, cmd_list, node_id, time_log_file):
    save_dir = os.path.join(cfg.qsub_script.save_dir, job)
    os.makedirs(save_dir, exist_ok=True)
    job_magic = cfg.qsub_script.qsub_magic
    os.makedirs(job_magic.log_dir, exist_ok=True)
    job_cfg = cfg.execution[job]
    magic_lines = [
        "#!/bin/bash",
        f"#PBS -l select={job_magic.num_node}",
        f"#PBS -q {job_cfg.node_type}",
        f"#PBS -l walltime={job_cfg.walltime}",
        f"#PBS -P {job_magic.group}",
        f"#PBS -e {job_magic.log_dir}",
        f"#PBS -o {job_magic.log_dir}",
        f"cd {job_magic.cwd}",
        f"source {job_magic.env_path}",
    ]
    with open(os.path.join(save_dir, f"{job}_{node_id:02d}.sh"), "w") as f:
        f.write("\n".join(magic_lines))
        f.write("\n\n")
        f.write("set -e")
        f.write("\n\n")
        for cmd in cmd_list:
            f.write(_format_cmd_multiline(cmd))
            f.write("\n\n")
        f.write(
            "echo "
            + '"'
            + f"{job}_{node_id:02d}"
            + " DONE"
            + "(${PBS_JOBID}):"
            + " ${"
            + "SECONDS"
            + "} seconds"
            + '"'
            + " >> "
            + f"{time_log_file}"
        )


def _format_cmd_multiline(tokens: list[str]) -> str:
    return (
        tokens[0]
        + " "
        + tokens[1]
        + " \\\n  "
        + " \\\n  ".join(shlex.quote(t) for t in tokens[2:])
    )


def create_all_run_script(cfg, job_list, time_log_file):
    experiment_name = cfg.experiment_name
    save_path = os.path.join(cfg.qsub_script.save_dir, "all_process.sh")
    previous_job_list = []
    all_cmd_dict = {}
    for i, job in enumerate(job_list):
        run_script_list = sorted(
            glob.glob(os.path.join(cfg.qsub_script.save_dir, job) + "/*.sh")
        )
        cmd_list = []
        submitted_job_list = []
        for run_script in run_script_list:
            job_name = os.path.basename(run_script).split(".")[0]
            if i == 0:
                cmd = f"{job_name}=$(qsub -N {experiment_name}_{job_name} {run_script})"
            else:
                wait_cmd = ":".join(
                    ["${" + str(job) + "}" for job in previous_job_list]
                )
                cmd = f'{job_name}=$(qsub -N {experiment_name}_{job_name} -W "depend=afterok:{wait_cmd}" {run_script})'
            submitted_job_list.append(job_name)
            cmd_list.append(cmd)
        previous_job_list = submitted_job_list
        all_cmd_dict[job] = cmd_list

    with open(save_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {cfg.qsub_script.qsub_magic.cwd}\n")
        f.write(f"source {cfg.qsub_script.qsub_magic.env_path}\n")
        f.write("\n\n")
        for job, cmd_list in all_cmd_dict.items():
            f.write(f"# {job}")
            for cmd in cmd_list:
                f.write("\n")
                f.write(cmd)
            f.write("\n\n")

    return save_path


if __name__ == "__main__":
    main()
