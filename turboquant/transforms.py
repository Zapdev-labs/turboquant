import numpy as np
from typing import Tuple


def walsh_hadamard_transform(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform (FWHT) in O(n log n).

    The Walsh-Hadamard transform is an orthogonal transform used for
    random rotation in TurboQuant. It has no multiplication operations,
    only additions and subtractions, making it very fast.

    Args:
        x: Input array of shape (..., n) where n is a power of 2

    Returns:
        Transformed array with same shape
    """
    x = x.copy()
    n = x.shape[-1]

    # Ensure n is power of 2
    assert (n & (n - 1)) == 0, "Input dimension must be power of 2"

    # Iterative FWHT
    h = 1
    while h < n:
        # Process pairs of h elements
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                # Butterfly operation
                a = x[..., j]
                b = x[..., j + h]
                x[..., j] = a + b
                x[..., j + h] = a - b
        h *= 2

    # Normalize
    return x / np.sqrt(n)


def inverse_walsh_hadamard_transform(x: np.ndarray) -> np.ndarray:
    """Inverse Walsh-Hadamard Transform.

    The inverse is the same as forward (self-inverse property).
    """
    return walsh_hadamard_transform(x)


def polar_transform(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform Cartesian coordinates to polar coordinates.

    For a d-dimensional vector, returns:
    - radius: The magnitude (single value per vector)
    - angles: d-1 angles representing direction

    Args:
        x: Input array of shape (..., d)

    Returns:
        Tuple of (radius, angles)
    """
    # Compute radius (magnitude)
    radius = np.linalg.norm(x, axis=-1, keepdims=True)

    # Compute angles recursively
    # For 2D: θ = atan2(y, x)
    # For higher dimensions, use recursive polar transformation
    angles = _cartesian_to_angles(x)

    return radius, angles


def inverse_polar_transform(radius: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Transform polar coordinates back to Cartesian.

    Args:
        radius: Magnitude array of shape (..., 1)
        angles: Angles array of shape (..., d-1)

    Returns:
        Cartesian coordinates of shape (..., d)
    """
    return _angles_to_cartesian(radius, angles)


def _cartesian_to_angles(x: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to angles using recursive transformation.

    For d-dimensional input, produces d-1 angles.
    """
    shape = x.shape
    d = shape[-1]

    if d == 1:
        return np.zeros(shape[:-1] + (0,))

    if d == 2:
        # Simple 2D case
        return np.arctan2(x[..., 1:2], x[..., 0:1])

    # Recursive case for higher dimensions
    # First angle: atan2(x_d, sqrt(x_1^2 + ... + x_{d-1}^2))
    x_without_last = x[..., :-1]
    x_last = x[..., -1:]

    radius_2d = np.linalg.norm(x_without_last, axis=-1, keepdims=True)
    angle_1 = np.arctan2(x_last, radius_2d)

    # Remaining angles from recursive call
    if d > 2:
        remaining_angles = _cartesian_to_angles(x_without_last)
        return np.concatenate([angle_1, remaining_angles], axis=-1)

    return angle_1


def _angles_to_cartesian(radius: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Convert angles back to Cartesian coordinates."""
    n_angles = angles.shape[-1]
    d = n_angles + 1

    if d == 1:
        return radius

    if d == 2:
        # 2D case
        x0 = radius * np.cos(angles[..., 0:1])
        x1 = radius * np.sin(angles[..., 0:1])
        return np.concatenate([x0, x1], axis=-1)

    # Recursive case
    first_angle = angles[..., 0:1]
    remaining_angles = angles[..., 1:]

    # First coordinate
    x0 = radius * np.cos(first_angle)

    # Remaining coordinates from recursive call
    remaining_radius = radius * np.sin(first_angle)
    remaining_coords = _angles_to_cartesian(remaining_radius, remaining_angles)

    return np.concatenate([x0, remaining_coords], axis=-1)


def random_rotation_matrix(d: int, seed: int = 42) -> np.ndarray:
    """Generate a random orthogonal rotation matrix.

    Uses QR decomposition on a Gaussian matrix to generate
    a random orthogonal matrix with determinant +1.

    Args:
        d: Dimension of the matrix
        seed: Random seed for reproducibility

    Returns:
        Orthogonal matrix of shape (d, d)
    """
    np.random.seed(seed)

    # Generate random Gaussian matrix
    A = np.random.randn(d, d)

    # QR decomposition
    Q, R = np.linalg.qr(A)

    # Ensure proper rotation (det = +1, not -1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    return Q.astype(np.float32)
