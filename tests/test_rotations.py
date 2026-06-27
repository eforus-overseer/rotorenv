"""Tests for the shared rotation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from rotorenv.core.rotations import (
    euler_to_quat,
    euler_to_rotmat,
    normalize_quat,
    quat_mul,
    quat_to_euler,
    quat_to_rotmat,
)


def test_identity_orientation_is_identity_matrix() -> None:
    """Zero Euler angles produce the identity rotation."""
    np.testing.assert_allclose(euler_to_rotmat(np.zeros(3)), np.eye(3), atol=1e-12)


def test_rotmat_is_orthonormal() -> None:
    """A rotation matrix satisfies R^T R = I and det(R) = 1."""
    rot = euler_to_rotmat(np.array([0.3, -0.4, 1.1]))
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0)


def test_euler_quat_roundtrip() -> None:
    """Euler -> quat -> Euler recovers the original angles (away from gimbal lock)."""
    angles = np.array([0.2, -0.5, 0.9])
    recovered = quat_to_euler(euler_to_quat(angles))
    np.testing.assert_allclose(recovered, angles, atol=1e-9)


def test_euler_and_quat_rotmats_agree() -> None:
    """The matrix from Euler equals the matrix from the equivalent quaternion."""
    angles = np.array([0.3, 0.2, -0.7])
    np.testing.assert_allclose(
        euler_to_rotmat(angles), quat_to_rotmat(euler_to_quat(angles)), atol=1e-12
    )


def test_quat_mul_identity() -> None:
    """Multiplying by the identity quaternion is a no-op."""
    q = normalize_quat(np.array([0.5, 0.5, 0.5, 0.5]))
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(quat_mul(q, identity), q, atol=1e-12)


def test_normalize_degenerate_quat() -> None:
    """A near-zero quaternion normalises to the identity rather than NaN."""
    np.testing.assert_array_equal(
        normalize_quat(np.zeros(4)), np.array([1.0, 0.0, 0.0, 0.0])
    )


def test_yaw_rotation_maps_x_axis() -> None:
    """A 90-degree yaw maps body +x onto world +y."""
    rot = euler_to_rotmat(np.array([0.0, 0.0, np.pi / 2.0]))
    np.testing.assert_allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
