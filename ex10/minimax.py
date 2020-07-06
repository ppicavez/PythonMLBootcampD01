import numpy as np


def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray
    using the min-max,→ standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """

    x_sort = sorted(x)
    x_norm = (x - x_sort[0]) / (x_sort[- 1] - x_sort[0])
    return x_norm


X = np.array([0, 15, -9, 7, 12, 3, -21])
print(minmax(X))
