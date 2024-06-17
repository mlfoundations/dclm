from collections import Counter
from typing import Dict

import numpy as np


def percentiles(values):
    """
    Calculate percentiles of a list of values. In case the percentile is not in the list, the nearest value is returned.

    Arguments:
    values -- A list of values.

    Returns:
    A dictionary with the percentiles.
    """
    if not isinstance(values, list):
        raise TypeError("Values must be a list.")
    vals = dict(zip(["min", "max", "median", "p25", "p75", "p90", "p95", "p99"],
                    np.percentile(values, [0., 100, 50, 25, 75, 90, 95, 99], method="nearest")))
    vals['mean'] = np.mean(values)
    return vals


def histogram(values):
    """
    Calculate a histogram of a list of values.

    Arguments:
    values -- A list of values.

    Returns:
    A dictionary with the histogram.
    """
    if isinstance(values[0], str):
        return Counter(values)
    else:
        hist, bin_edges = np.histogram(values, bins=10)
        return dict(zip(bin_edges, hist))


def threshold_transform(weighted_values: Dict[str, float], threshold, default='unknown') -> str:
    """
    Given a dict where the keys are weighted by the values, return the key with the highest value,
    if it is above the given threshold (otherwise return the default value).

    Arguments:
    weighted_values -- A dict where the keys are weighted by the values.
    threshold -- The threshold to use.
    default -- The default value to return if the threshold is not met.

    Returns:
    The key with the highest value, if it is above the given threshold (otherwise return the default value).
    """
    weighted_values = {k: v for k, v in weighted_values.items() if v >= threshold}
    if len(weighted_values) == 0:
        return default
    return max(weighted_values, key=weighted_values.get)