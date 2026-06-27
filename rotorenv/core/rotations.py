"""Shared rotation utilities (Euler <-> quaternion <-> matrix).

A single source of truth for the frame convention used across physics backends
and the renderer. We use the **ZYX** Euler convention: a body-to-world rotation
is ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``, with Euler angles ordered
``(roll, pitch, yaw)`` to match :class:`~rotorenv.core.state.DroneState`.

Quaternions are stored as ``[w, x, y, z]`` (scalar-first), unit-norm.
"""

from __future__ import annotations

import numpy as np


def euler_to_rotmat(orientation: np.ndarray) -> np.ndarray:
    """Return the ZYX body-to-world rotation matrix for Euler angles.

    Args:
        orientation: ``(roll, pitch, yaw)`` in radians.

    Returns:
        A ``(3, 3)`` rotation matrix mapping body-frame vectors to world frame.
    """
    roll, pitch, yaw = (float(orientation[0]), float(orientation[1]), float(orientation[2]))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def euler_to_quat(orientation: np.ndarray) -> np.ndarray:
    """Convert ZYX Euler angles to a scalar-first unit quaternion ``[w, x, y, z]``.

    Args:
        orientation: ``(roll, pitch, yaw)`` in radians.

    Returns:
        Unit quaternion as a length-4 ``float64`` array.
    """
    roll, pitch, yaw = (float(orientation[0]), float(orientation[1]), float(orientation[2]))
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,  # w
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy,  # z
        ],
        dtype=np.float64,
    )


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert a scalar-first quaternion to ZYX Euler angles ``(roll, pitch, yaw)``.

    Pitch is clamped to ``[-pi/2, pi/2]``; the asin argument is clipped to guard
    against gimbal-lock numerical overshoot.

    Args:
        q: Quaternion ``[w, x, y, z]`` (need not be normalised).

    Returns:
        ``(roll, pitch, yaw)`` in radians as a length-3 ``float64`` array.
    """
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Return the body-to-world rotation matrix for a scalar-first quaternion.

    Args:
        q: Quaternion ``[w, x, y, z]`` (need not be normalised; it is normalised
            internally).

    Returns:
        A ``(3, 3)`` rotation matrix.
    """
    q = normalize_quat(q)
    w, x, y, z = (q[0], q[1], q[2], q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions ``a ⊗ b``.

    Args:
        a: Left quaternion ``[w, x, y, z]``.
        b: Right quaternion ``[w, x, y, z]``.

    Returns:
        The product quaternion as a length-4 ``float64`` array.
    """
    aw, ax, ay, az = (a[0], a[1], a[2], a[3])
    bw, bx, by, bz = (b[0], b[1], b[2], b[3])
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Return ``q`` rescaled to unit norm (identity quaternion if degenerate)."""
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm
