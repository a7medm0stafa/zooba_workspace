"""
EKF Core — 5-State Extended Kalman Filter for 2D Vehicle Localization
======================================================================
FILE:    localization/localization/ekf_core.py
STATUS:  v3.0 — 5-state EKF with online gyro bias estimation
CREATED: 2026-05-17

PURPOSE:
    Pure-Python/Numpy EKF engine for 2D vehicle localization.
    No ROS dependencies — called by ekf_localization_node.py.

STATE VECTOR (5 × 1):
    x[0] = x        position X in odom frame  [m]
    x[1] = y        position Y in odom frame  [m]
    x[2] = θ        heading / yaw             [rad]
    x[3] = v        longitudinal velocity     [m/s]
    x[4] = b_ω      gyroscope yaw-rate bias   [rad/s]

    The gyro bias state allows the filter to estimate and remove
    dynamic bias caused by motor vibrations (~0.1 rad/s on hardware).

PREDICTION MODEL (Ackermann kinematics):
    x'   = x + v·cos(θ)·dt
    y'   = y + v·sin(θ)·dt
    θ'   = θ + (ω_gyro - b_ω)·dt    (bias-corrected gyroscope)
    v'   = v
    b_ω' = b_ω                       (random walk bias model)

MEASUREMENT MODELS:
    1. Encoder velocity:  z = v_enc,  h(x) = v
    2. Ackermann heading: z = θ_pre + (v·tan(δ)/L)·dt,  h(x) = θ
       Cross-checks gyro against steering geometry → makes bias observable.
    3. ZUPT (v ≈ 0):     z = 0,  h(x) = v
    4. IMU heading:       z = θ_imu,  h(x) = θ  (optional, disabled by default)

ALL UNITS: meters, radians, seconds.
"""

import math
import numpy as np


class EKF2D:
    """5-state EKF: [x, y, θ, v, b_ω] with online gyro bias estimation."""

    # State indices
    IX = 0       # x position
    IY = 1       # y position
    ITHETA = 2   # heading / yaw
    IV = 3       # longitudinal velocity
    IB = 4       # gyro yaw-rate bias

    N_STATES = 5

    def __init__(self, sigma_v: float, sigma_omega: float,
                 sigma_bias: float = 0.005):
        """
        Parameters
        ----------
        sigma_v : float       Process noise σ for velocity [m/s].
        sigma_omega : float   Process noise σ for yaw rate [rad/s].
        sigma_bias : float    Process noise σ for bias drift [rad/s²].
                              Controls how fast bias can change.
                              Small = slow adaptation, stable bias.
                              Large = fast adaptation, noisy bias.
        """
        self.x = np.zeros(self.N_STATES)
        self.P = np.diag([0.01, 0.01, 0.01, 0.1, 0.01])
        self._sigma_v = sigma_v
        self._sigma_omega = sigma_omega
        self._sigma_bias = sigma_bias
        self._theta_pre_predict = 0.0

    # ==================== Initialization ====================

    def set_initial_state(self, x: float, y: float, theta: float,
                          v: float = 0.0, bias: float = 0.0):
        """Hard-set the state vector. Resets covariance."""
        self.x = np.array([x, y, theta, v, bias], dtype=float)
        self.P = np.diag([0.01, 0.01, 0.01, 0.1, 0.01])

    # ==================== Prediction ====================

    def predict(self, omega_z: float, dt: float):
        """EKF prediction step.

        Parameters
        ----------
        omega_z : float   Raw gyroscope yaw rate [rad/s] (before EKF bias removal).
        dt : float        Time step [s].
        """
        if dt <= 0.0:
            return

        x, y, theta, v, b_omega = self.x
        self._theta_pre_predict = theta

        cos_th = math.cos(theta)
        sin_th = math.sin(theta)

        # Bias-corrected yaw rate
        omega_corrected = omega_z - b_omega

        # ---- State prediction ----
        x_new = x + v * cos_th * dt
        y_new = y + v * sin_th * dt
        theta_new = theta + omega_corrected * dt
        v_new = v
        b_new = b_omega  # random walk

        # ---- Jacobian F (5×5) ----
        F = np.eye(self.N_STATES)
        F[self.IX, self.ITHETA] = -v * sin_th * dt
        F[self.IX, self.IV] = cos_th * dt
        F[self.IY, self.ITHETA] = v * cos_th * dt
        F[self.IY, self.IV] = sin_th * dt
        F[self.ITHETA, self.IB] = -dt  # ∂θ'/∂b_ω = -dt

        # ---- Process noise Q (noise-input) ----
        # G maps [w_v, w_ω, w_b] into state space
        G = np.array([
            [cos_th * dt, 0.0, 0.0],
            [sin_th * dt, 0.0, 0.0],
            [0.0,         dt,  0.0],
            [1.0,         0.0, 0.0],
            [0.0,         0.0, dt],
        ])
        Q_c = np.diag([self._sigma_v**2, self._sigma_omega**2,
                        self._sigma_bias**2])
        Q = G @ Q_c @ G.T

        self.P = F @ self.P @ F.T + Q
        self.x = np.array([x_new, y_new, self._normalize_angle(theta_new),
                           v_new, b_new])
        self.P = 0.5 * (self.P + self.P.T)

    # ==================== Measurement Updates ====================

    def update_velocity(self, v_measured: float, R: float):
        """Encoder velocity measurement update."""
        H = np.zeros((1, self.N_STATES))
        H[0, self.IV] = 1.0
        self._joseph_update(np.array([v_measured]),
                            np.array([self.x[self.IV]]),
                            H, np.array([[R]]))

    def update_heading(self, theta_measured: float, R: float):
        """Absolute heading measurement update (angle-wrapped)."""
        H = np.zeros((1, self.N_STATES))
        H[0, self.ITHETA] = 1.0

        innovation = np.array([
            self._normalize_angle(theta_measured - self.x[self.ITHETA])
        ])

        S = H @ self.P @ H.T + np.array([[R]])
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.x = self.x + (K @ innovation).flatten()
        self.x[self.ITHETA] = self._normalize_angle(self.x[self.ITHETA])

        I_KH = np.eye(self.N_STATES) - K @ H
        R_mat = np.array([[R]])
        self.P = I_KH @ self.P @ I_KH.T + K @ R_mat @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_heading_from_ackermann(self, steering_angle: float,
                                       wheelbase: float,
                                       dt: float, R: float):
        """Ackermann heading measurement — makes gyro bias observable.

        Computes expected heading from steering geometry and uses it
        as a measurement. The innovation (Ackermann vs gyro-predicted)
        drives BOTH heading correction AND bias estimation.

        When the gyro has vibration bias:
          - Innovation = (ω_ack - ω_gyro + b_ω) · dt
          - Kalman gain corrects θ AND learns b_ω
          - Over ~2-5 seconds, b_ω converges to true bias

        Parameters
        ----------
        steering_angle : float   Commanded steering angle δ [rad].
        wheelbase : float        Axle distance L [m].
        dt : float               Time step [s].
        R : float                Measurement noise variance [rad²].
        """
        if dt <= 0.0:
            return

        v = self.x[self.IV]
        delta_clamped = max(-1.2, min(1.2, steering_angle))
        omega_ack = v * math.tan(delta_clamped) / wheelbase

        theta_ack = self._normalize_angle(
            self._theta_pre_predict + omega_ack * dt
        )
        self.update_heading(theta_ack, R)

    def zupt(self, R: float):
        """Zero-Velocity Update."""
        H = np.zeros((1, self.N_STATES))
        H[0, self.IV] = 1.0
        self._joseph_update(np.array([0.0]),
                            np.array([self.x[self.IV]]),
                            H, np.array([[R]]))

    # ==================== Core EKF Math ====================

    def _joseph_update(self, z, z_pred, H, R):
        """Generic EKF measurement update (Joseph form)."""
        innovation = z - z_pred
        S = H @ self.P @ H.T + R

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.x = self.x + (K @ innovation).flatten()
        self.x[self.ITHETA] = self._normalize_angle(self.x[self.ITHETA])

        I_KH = np.eye(self.N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ==================== Utilities ====================

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    # ==================== Accessors ====================

    @property
    def position_x(self) -> float:
        return float(self.x[self.IX])

    @property
    def position_y(self) -> float:
        return float(self.x[self.IY])

    @property
    def heading(self) -> float:
        return float(self.x[self.ITHETA])

    @property
    def velocity(self) -> float:
        return float(self.x[self.IV])

    @property
    def gyro_bias(self) -> float:
        """Estimated gyro yaw-rate bias [rad/s]."""
        return float(self.x[self.IB])

    @property
    def covariance_diagonal(self) -> np.ndarray:
        return np.diag(self.P)
