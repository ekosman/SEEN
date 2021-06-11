import logging
import os
import sys
from os import path


def register_logger(log_file=None, stdout=True):
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    handlers = []

    if stdout:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(format="%(asctime)s %(message)s",
                        handlers=handlers,
                        level=logging.INFO,
                        )
    logging.root.setLevel(logging.INFO)


def register_exps_dir(dir_path):
    if not path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def log_args_description(args):
    logging.info("Running command:")
    logging.info(' '.join(sys.argv))
    s = """
====== All settings used ======:\n
"""
    for k, v in sorted(vars(args).items()):
        s += f"      {k}: {v}\n"

    logging.info(s)
