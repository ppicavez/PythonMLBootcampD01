import numpy as np

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without any for loop. The three arrays must have compatible dimensions.
    Args:
    x: has to be a numpy.ndarray, a matrix of dimension m * 1.
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    theta: has to be a numpy.ndarray, a 2 * 1 vector.
    Returns:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta is an empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if not len(x) or not len(y) or not len(theta):
        return None
    if len(x.shape) != 1 or len(y.shape) != 1 or theta.shape != (2, ):
        return None
    if len(x) != len(y):
        return None

    X_prime = np.c_[np.ones((len(x), 1)), x]
    result = (X_prime.T @ (X_prime @ theta - y)) / len(x)
    return result


if __name__ == "__main__":
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
    # Example 0:
    theta1 = np.array([2, 0.7])
    print(gradient(x, y, theta1))

    # Example 1:
    theta2 = np.array([1, -0.4])
    print(gradient(x, y, theta2))
