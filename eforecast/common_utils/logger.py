import logging
import os


def create_logger(logger_name, abs_path, logger_path, write_type='w'):
    assert write_type in ('w', 'a')
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(os.path.join(abs_path, logger_path), mode=write_type, encoding="utf-8")
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
