"""Funcs for logging"""
import logging
import os


def build_logger(log_level, logger_name, out_file=None):
    """Build logger instance.

    Parameters
    ----------
    log_level : type
        Set logging level.
    logger_name : type
        Name of the logger instance.
    out_file : type
        Path for logger output.

    Returns
    -------
    out: logging.Logger
        Logger instance.

    """
    logger = logging.Logger(logger_name)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if out_file:
        out_dir = '/'.join(out_file.split('/')[:-1])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        file_handler = logging.FileHandler(out_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(log_level)
    return logger
