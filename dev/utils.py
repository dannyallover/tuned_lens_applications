from collections.abc import Iterable
from typing import Any
import torch
import numpy as np


def dumpclean(obj: Any, level: int) -> None:
    """
    Developed to traverse and print objects at each |level| (since json cannot
    serialize lambdas).

    Parameters
    ----------
    obj: Any, required
        The object to be traversed.
    level: required, int
        Indicates how many tabs out to print the value.

    Returns
    ------
    None
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, "__iter__"):
                print("\t" * level, k, ":")
                dumpclean(v, level + 1)
            else:
                print("\t" * level, "%s : %s" % (k, v))
    elif isinstance(obj, list):
        for v in obj:
            if hasattr(v, "__iter__"):
                dumpclean(v, level)
            else:
                print("\t" * level, v)
    else:
        print("\t" * level, obj)

def to_numpy(t: torch.Tensor) -> np.ndarray:
    """
    Detach tensor from computational graph, place it on the cpu, and convert it
    to a numpy array.

    Parameters
    ----------
    t: tensor, required
        Any pytorch tensor.

    Returns
    ------
    np.ndarray
        Converted numpy array.
    """
    return t.detach().cpu().numpy()

def mean_up_low(t: torch.Tensor) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Compute the mean, upper bound on confidence, and lower bound on confidence
    of the values.

    Parameters
    ----------
    t : torch.Tensor, required
        Tensor with numerical values.

    Returns
    ------
    t_mean: np.ndarray
        Mean values.
    t_up: np.ndarray
        Upper confidence values.
    t_low: np.ndarray
        Lower confidence values.
    """
    n_q = t.shape[1]
    t_mean = t.mean(dim=1)
    t_std = t.std(dim=1)
    t_up = t_mean + ((1.96 * t_std) / np.sqrt(n_q))
    t_low = t_mean - ((1.96 * t_std) / np.sqrt(n_q))

    return to_numpy(t_mean), to_numpy(t_up), to_numpy(t_low)

def combine_dicts(dict1: dict, dict2: dict) -> dict:
    for key in dict1:
        dict1[key] += dict2[key]
    return dict1