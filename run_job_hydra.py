import hydra
from omegaconf import DictConfig
import subprocess
import sys
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
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
    run_pipeline_with_config(cfg)


def run_pipeline_with_config(cfg: DictConfig) -> None:
    settings = cfg.settings
    execution = cfg.execution
    pipeline_config = cfg.pipeline

    os.makedirs(settings.save_dir, exist_ok=True)

    logger.info(f"Output dir: {settings.save_dir}")

    logger.info(f"Pipeline: {execution.pipeline}\n")

    # Initialize time log
    with open(settings.time_log_file, "w") as f:
        f.write("# step time_sec status\n")

    total_time = 0.0

    for step in execution.pipeline:
        start_time = time.time()
        step_config = pipeline_config[step]
        script = step_config.script
        step_inputs = step_config.inputs

        logger.info(f"=== Executing step {step} ({script}) ===")

        if not os.path.exists(script):
            logger.error(f"Script {script} not found")
            raise FileNotFoundError(script)

        cmd = (
            ["bash", script]
            + [str(settings[p]) for p in step_inputs]
            + [str(execution.node_type)]
            + [str(execution.step_log_level)]
        )

        try:
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
                logger.warning(result.stderr.rstrip())

            status = "SUCCESS"

        except subprocess.CalledProcessError as e:
            if e.stdout:
                logger.info(e.stdout.rstrip())
            if e.stderr:
                logger.error(e.stderr.rstrip())
            status = f"FAIL({e.returncode})"
            end_time = time.time()
            with open(settings.time_log_file, "a") as f:
                f.write(f"{step} {end_time - start_time:.2f} ({status})\n")
            raise

        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed
        with open(settings.time_log_file, "a") as f:
            f.write(f"{step} {elapsed:.2f} ({status})\n")

    with open(settings.time_log_file, "a") as f:
        f.write(f"total {total_time:.2f}\n")

    logger.info("Pipeline completed successfully")


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
        if not os.path.exists(settings.video_path):
            logger.error(f"Video file not found: {settings.video_path}")
            return False

        if not os.path.exists(settings.weight_path):
            logger.error(f"Weight file not found: {settings.weight_path}")
            return False

        if not os.path.exists(settings.bev_file):
            logger.error(f"BEV file not found: {settings.bev_file}")
            return False

        if not os.path.exists(settings.size_file):
            logger.error(f"Size file not found: {settings.size_file}")
            return False

        logger.info("Configuration validation completed")
        return True

    except Exception as e:
        logger.error(f"Error occurred during configuration validation: {e}")
        return False


if __name__ == "__main__":
    main()
