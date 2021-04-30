import logging
import sys
import os
from os import path
import numpy as np


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


def distance_from_line(p1, p2, p3):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    try:
        d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    except:
        d = 0
    return d


def get_threshold_by_distance(measures_sorted_flat):
    """
    :param measures_sorted_flat: array of values sorted in descending order
    :return:
    """
    logging.info(f"find threshold for: {str(measures_sorted_flat)}")

    p1 = [0, max(measures_sorted_flat)]
    p2 = [len(measures_sorted_flat) - 1, min(measures_sorted_flat)]
    orig_ds = np.array([distance_from_line(p1, p2, [k, measures_sorted_flat[k]]) for k in range(len(measures_sorted_flat))])

    intersections, = np.where(orig_ds < max(orig_ds)/100)
    offset = intersections[-2]
    print(f"offset = {offset}")
    ds = orig_ds[offset:]

    most_far = np.argmax(ds) + 1
    close_indices, = np.where(ds <= 1e-6)
    closest = close_indices[1] if len(close_indices) >= 2 else np.inf
    chosen = np.clip(min(most_far, closest), 0, len(measures_sorted_flat) - 1)
    chosen += offset
    try:
        threshold = measures_sorted_flat[chosen]
        delta = (max(measures_sorted_flat) - threshold) / chosen

        logging.info(f"chose threshold {threshold} @ {chosen}, delta: {delta}")

        return chosen, orig_ds, threshold
        # return threshold, chosen, ds, delta
    except:
        print(f"most far: {most_far}")
        print(f"closest: {closest}")
        exit()


