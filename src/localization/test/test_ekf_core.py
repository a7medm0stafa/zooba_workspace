"""
Unit Tests for EKF Core (4-state filter)
==========================================
Tests the mathematical correctness of the EKF2D class.
Run: python3 -m pytest src/localization/test/test_ekf_core.py -v
"""

import math
import numpy as np
import pytest
import sys
import os

# Add the localization package to the path for standalone testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from localization.ekf_core import EKF2D


class TestStationary:
    """Tests with a stationary vehicle — covariance must stay bounded."""

    def test_position_holds_at_origin(self):
        """Stationary vehicle: position must remain (0, 0)."""
        ekf = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf.set_initial_state(0.0, 0.0, 0.0)

        for _ in range(500):  # 10 seconds at 50 Hz
            ekf.predict(omega_z=0.0, dt=0.02)
            ekf.zupt(R=1e-6)

        assert abs(ekf.position_x) < 1e-6, f"x drifted to {ekf.position_x}"
        assert abs(ekf.position_y) < 1e-6, f"y drifted to {ekf.position_y}"
        assert abs(ekf.velocity) < 0.01, f"v drifted to {ekf.velocity}"

    def test_covariance_bounded_stationary(self):
        """With ZUPT active, position σ must not grow unboundedly."""
        ekf = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf.set_initial_state(0.0, 0.0, 0.0)

        # Run for 100 seconds at 50 Hz — the old filter would reach σ > 3.0
        for _ in range(5000):
            ekf.predict(omega_z=0.0, dt=0.02)
            ekf.zupt(R=1e-6)

        cov = ekf.covariance_diagonal
        sigma_x = math.sqrt(cov[0])
        sigma_y = math.sqrt(cov[1])

        # Position σ must stay small (< 0.15 m), not explode to 3+ like old filter
        assert sigma_x < 0.15, f"σ_x exploded to {sigma_x} (old filter bug)"
        assert sigma_y < 0.15, f"σ_y exploded to {sigma_y} (old filter bug)"

    def test_heading_stable_when_gyro_zero(self):
        """With zero gyro input, heading should not drift."""
        ekf = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf.set_initial_state(0.0, 0.0, math.radians(45.0))

        for _ in range(500):
            ekf.predict(omega_z=0.0, dt=0.02)
            ekf.zupt(R=1e-6)

        # Heading should stay at 45°
        assert abs(math.degrees(ekf.heading) - 45.0) < 0.1


class TestStraightLine:
    """Tests driving in a straight line along the X axis."""

    def test_straight_line_x_grows(self):
        """Driving forward at 0.2 m/s for 10s → x ≈ 2.0 m."""
        ekf = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf.set_initial_state(0.0, 0.0, 0.0)  # heading = 0 (along +X)

        for _ in range(500):  # 10 seconds at 50 Hz
            ekf.predict(omega_z=0.0, dt=0.02)
            ekf.update_velocity(0.2, R=0.05**2)

        # x should be approximately 2.0 m
        assert abs(ekf.position_x - 2.0) < 0.1, f"x = {ekf.position_x}"
        # y should stay near 0
        assert abs(ekf.position_y) < 0.05, f"y drifted to {ekf.position_y}"

    def test_straight_line_y_axis(self):
        """Driving along +Y (heading = 90°) at 0.1 m/s for 5s → y ≈ 0.5 m."""
        ekf = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf.set_initial_state(0.0, 0.0, math.pi / 2)  # heading = +Y

        for _ in range(250):  # 5 seconds at 50 Hz
            ekf.predict(omega_z=0.0, dt=0.02)
            ekf.update_velocity(0.1, R=0.05**2)

        assert abs(ekf.position_y - 0.5) < 0.1, f"y = {ekf.position_y}"
        assert abs(ekf.position_x) < 0.05, f"x drifted to {ekf.position_x}"


class TestRotation:
    """Tests pure rotation (turning in place)."""

    def test_pure_rotation_90_degrees(self):
        """Rotating at 0.5 rad/s for π/2 / 0.5 = π s ≈ 3.14s → heading ≈ 90°."""
        ekf = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf.set_initial_state(0.0, 0.0, 0.0)

        omega = 0.5  # rad/s
        duration = math.pi / (2.0 * omega)  # seconds to rotate 90°
        steps = int(duration / 0.02)

        for _ in range(steps):
            ekf.predict(omega_z=omega, dt=0.02)
            ekf.zupt(R=1e-6)  # stationary (turning in place)

        # Heading should be approximately 90°
        assert abs(math.degrees(ekf.heading) - 90.0) < 2.0, \
            f"heading = {math.degrees(ekf.heading):.1f}°"
        # Position should not change (turning in place)
        assert abs(ekf.position_x) < 0.01
        assert abs(ekf.position_y) < 0.01


class TestZUPT:
    """Tests the Zero-Velocity Update."""

    def test_zupt_clamps_velocity(self):
        """ZUPT should drive velocity to ~0."""
        ekf = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf.set_initial_state(0.0, 0.0, 0.0, v=0.5)  # start at 0.5 m/s

        # Apply ZUPT many times — velocity should converge to 0
        for _ in range(100):
            ekf.predict(omega_z=0.0, dt=0.02)
            ekf.zupt(R=1e-6)

        assert abs(ekf.velocity) < 0.001, f"v = {ekf.velocity} after ZUPT"

    def test_zupt_bounds_velocity_covariance(self):
        """ZUPT should keep P_vv small."""
        ekf = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf.set_initial_state(0.0, 0.0, 0.0)

        for _ in range(1000):
            ekf.predict(omega_z=0.0, dt=0.02)
            ekf.zupt(R=1e-6)

        P_vv = ekf.P[ekf.IV, ekf.IV]
        assert P_vv < 0.001, f"P_vv = {P_vv} — should be clamped by ZUPT"


class TestAngleWrapping:
    """Tests angle normalization at ±π boundaries."""

    def test_normalize_positive_overflow(self):
        """Angle > π should wrap to negative."""
        result = EKF2D._normalize_angle(math.pi + 0.1)
        assert -math.pi <= result <= math.pi
        assert abs(result - (-math.pi + 0.1)) < 1e-10

    def test_normalize_negative_overflow(self):
        """Angle < -π should wrap to positive."""
        result = EKF2D._normalize_angle(-math.pi - 0.1)
        assert -math.pi <= result <= math.pi
        assert abs(result - (math.pi - 0.1)) < 1e-10

    def test_heading_wraps_during_rotation(self):
        """Continuous rotation should wrap heading correctly."""
        ekf = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf.set_initial_state(0.0, 0.0, math.pi - 0.1)  # just before +π

        # Rotate past the +π boundary
        for _ in range(50):
            ekf.predict(omega_z=1.0, dt=0.02)

        # Heading should have wrapped to negative side
        assert -math.pi <= ekf.heading <= math.pi

    def test_heading_update_wraps_innovation(self):
        """Heading update with large angle difference should wrap correctly."""
        ekf = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf.set_initial_state(0.0, 0.0, math.pi - 0.05)  # near +π

        # Measure heading just past -π (should be a small innovation, not 2π)
        ekf.update_heading(-math.pi + 0.05, R=0.01)

        # After update, heading should be near ±π, not pulled to 0
        assert abs(abs(ekf.heading) - math.pi) < 0.2


class TestQFormulation:
    """Tests that the noise-input Q formulation behaves correctly."""

    def test_q_position_noise_couples_through_jacobian(self):
        """Position covariance growth should couple through F Jacobian.

        The noise-input G matrix maps σ_v into position noise via cos(θ)·dt,
        independent of current velocity. However, the Jacobian F propagates
        existing P_vv uncertainty into P_xx proportionally to v via the
        F[x,v] = cos(θ)·dt term. So with velocity updates that maintain
        different velocity states, covariance growth should differ.
        """
        # Predict-only (no velocity updates): P_vv grows, feeds into P_xx via F
        ekf_no_update = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf_no_update.set_initial_state(0.0, 0.0, 0.0, v=0.2)
        for _ in range(100):
            ekf_no_update.predict(omega_z=0.0, dt=0.02)
        cov_no_update = ekf_no_update.covariance_diagonal[0]

        # With velocity updates: P_vv stays bounded, less feeds into P_xx
        ekf_with_update = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf_with_update.set_initial_state(0.0, 0.0, 0.0, v=0.2)
        for _ in range(100):
            ekf_with_update.predict(omega_z=0.0, dt=0.02)
            ekf_with_update.update_velocity(0.2, R=0.05**2)
        cov_with_update = ekf_with_update.covariance_diagonal[0]

        # Without updates, covariance should grow much faster
        assert cov_no_update > cov_with_update, \
            f"P_xx(no_update)={cov_no_update} should > P_xx(with_update)={cov_with_update}"

    def test_q_with_zupt_bounds_position_noise(self):
        """At v=0 WITH ZUPT, position noise should stay small.

        Without ZUPT, P_vv grows from G's [1,0] row, which propagates into
        P_xx through F's F[x,v]=cos(θ)·dt term. ZUPT clamps P_vv, which
        indirectly bounds position covariance.
        """
        ekf = EKF2D(sigma_v=0.1, sigma_omega=0.02)
        ekf.set_initial_state(0.0, 0.0, 0.0, v=0.0)

        for _ in range(500):  # 10 seconds
            ekf.predict(omega_z=0.0, dt=0.02)
            ekf.zupt(R=1e-6)

        sigma_x = math.sqrt(ekf.covariance_diagonal[0])
        # With ZUPT clamping P_vv, position noise should stay very small
        assert sigma_x < 0.15, f"σ_x = {sigma_x} — ZUPT should bound this"
