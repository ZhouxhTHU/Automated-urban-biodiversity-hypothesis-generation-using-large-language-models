import os
import logging
import datetime
import sys

# A global variable to store the logger for easy access across modules
_logger = None


def setup_logger(file_name, model_name: str, parent_dir: str = None):
    """
    Configure the logger to output to the console and to a file specific to this run.

    Args:
        model_name (str): The model name used to name the log folder.
        parent_dir (str, optional): Parent directory for logs. If None, uses "runs_log"

    Returns:
        tuple[logging.Logger, str]: The configured logger instance and the log directory path for this run.
    """
    global _logger
    if _logger and hasattr(_logger, 'log_dir'):
        return _logger, _logger.log_dir

    # 1. Create unique identifiers to avoid file duplication
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Create a log folder for this run
    run_dir_name = f"{file_name}_{model_name}_{timestamp}"

    # Use the custom parent directory or the default "runs_log" directory.
    if parent_dir is None:
        log_dir = os.path.join("runs_log", run_dir_name)
    else:
        # Create a file_name subdirectory under the specified parent, followed by a timestamped directory.
        log_dir = os.path.join(parent_dir, file_name, f"{model_name}_{timestamp}")

    os.makedirs(log_dir, exist_ok=True)

    # 3. Configure the logger
    logger = logging.getLogger(run_dir_name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate addition of processors
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. Create a FileHandler - write to log.txt
    log_file_path = os.path.join(log_dir, "run_log.txt")
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 5. Create a console handler (StreamHandler) - print to the screen
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # Attach the log directory to the logger object for easy access
    logger.log_dir = log_dir
    _logger = logger

    return logger, log_dir


import os
import logging  # Assuming a logger is defined elsewhere, as used in the original function

_logger = logging.getLogger(__name__)


def save_config_summary(log_dir: str, config_info: dict, mode: str = 'w'):
    """
    Save the configuration information to the specified log directory.

    Args:
        log_dir (str): The log directory path for this run.
        config_info (dict): A dictionary containing configuration information.
        mode (str): File opening mode. Use 'w' for overwrite (default)
                    or 'a' for append.

    Raises:
        ValueError: If the provided mode is not 'w' or 'a'.
    """
    if mode not in ('w', 'a'):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'w' (write/overwrite) or 'a' (append).")

    config_file_path = os.path.join(log_dir, "config_summary.txt")

    # Conditionally write the header only if in 'w' mode (overwrite)
    # or if the file doesn't exist and we're in 'a' mode (append will create it)
    write_header = (mode == 'w') or (mode == 'a' and not os.path.exists(config_file_path))

    with open(config_file_path, mode, encoding='utf-8') as f:
        if write_header:
            f.write("--- Configuration Summary ---\n\n")

        # Add a separator if appending to distinguish blocks, unless it's a new file
        # if mode == 'a' and not write_header:
        #     f.write("\n--- Appended Configuration ---\n\n")

        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")

    _logger.info(f"Configuration has been saved (mode: {mode}) to: {config_file_path}")


def save_to_txt(txt_name:str, log_dir: str, file_path: str, logger: logging.Logger):
    """Saves the entire content of a given file to a file in the log directory."""
    output_file_path = os.path.join(log_dir, txt_name)

    try:
        with open(file_path, 'r', encoding='utf-8') as f_in:
            content = f_in.read()

        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"--- From {os.path.basename(file_path)} (Saved on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n\n")
            f_out.write(content)

        logger.info(f"Current saved to: {output_file_path}")
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}. Cannot save.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving: {e}")
