"""
EKF Core — 4-State Extended Kalman Filter for 2D Vehicle Localization
======================================================================
FILE:    localization/localization/ekf_core.py
STATUS:  COMPLETE REWRITE — replaces the broken 5-state EKF
CREATED: 2026-05-17

PURPOSE:
    Pure-Python/Numpy EKF engine for 2D vehicle localization.
    No ROS dependencies — called by ekf_localization_node.py.
    Can be unit-tested standalone.

STATE VECTOR (4 × 1):
    x[0] = x        position X in odom frame  [m]
    x[1] = y        position Y in odom frame  [m]
    x[2] = θ        heading / yaw             [rad]
    x[3] = v        longitudinal velocity     [m/s]

    NOTE: Gyro bias is NOT a filter state. The Arduino firmware performs
    static gyro-Z offset calibration at boot (200 samples). The ROS node
    subtracts a yaw-angle offset during its own settling period.

PREDICTION MODEL (unicycle kinematics):
    x'  = x + v·cos(θ)·dt
    y'  = y + v·sin(θ)·dt
    θ'  = θ + ω_z·dt          (ω_z = bias-corrected gyroscope)
    v'  = v                    (constant between encoder updates)

PROCESS NOISE (noise-input formulation):
    Q_d = G · diag(σ_v², σ_ω²) · Gᵀ
    where G maps velocity/yaw-rate noise into state space.
    This ensures position noise grows ONLY through velocity uncertainty,
    fixing the unbounded covariance explosion of the old filter.

MEASUREMENT MODELS:
    1. Encoder velocity:  z = v_enc,  h(x) = v,  H = [0,0,0,1]
    2. IMU heading:       z = θ_imu,  h(x) = θ,  H = [0,0,1,0]
       (DISABLED by default — Arduino yaw is pure gyro integration,
        so using it double-counts gyro data. Kept for future use with
        magnetometer or GPS heading.)
    3. ZUPT (v ≈ 0):      z = 0,      h(x) = v,  H = [0,0,0,1]
    4. Ackermann heading rate:  z = θ_prev + (v·tan(δ)/L)·dt,
       h(x) = θ,  H = [0,0,1,0]
       Cross-checks gyro-integrated heading against Ackermann kinematics.
       Fights dynamic gyro bias from motor vibrations.

ALL UNITS: meters, radians, seconds. No exceptions.
"""

import math
import numpy as np


class EKF2D:
    """4-state EKF for 2D vehicle localization: [x, y, θ, v].

    Parameters
    ----------
    sigma_v : float
        Process noise standard deviation for velocity [m/s].
        Controls how much position uncertainty grows per prediction step.
    sigma_omega : float
        Process noise standard deviation for yaw rate [rad/s].
        Controls how much heading uncertainty grows per prediction step.
    """

    # State indices — named constants for clarity
    IX = 0       # x position
    IY = 1       # y position
    ITHETA = 2   # heading / yaw
    IV = 3       # longitudinal velocity

    N_STATES = 4

    def __init__(self, sigma_v: float, sigma_omega: float):
        # ---- State vector ----
        self.x = np.zeros(self.N_STATES)

        # ---- Covariance matrix ----
        self.P = np.diag([0.01, 0.01, 0.01, 0.1])

        # ---- Process noise parameters (continuous-time σ) ----
        self._sigma_v = sigma_v
        self._sigma_omega = sigma_omega

        # ---- Pre-prediction heading (for Ackermann update) ----
        self._theta_pre_predict = 0.0

    # ==================== Initialization ====================

    def set_initial_state(self, x: float, y: float, theta: float,
                          v: float = 0.0):
        """Hard-set the state vector. Resets covariance to initial values.

        Parameters
        ----------
        x, y : float    Position [m].
        theta : float   Heading [rad].
        v : float       Velocity [m/s].
        """
        self.x = np.array([x, y, theta, v], dtype=float)
        self.P = np.diag([0.01, 0.01, 0.01, 0.1])

    # ==================== Prediction ====================

    def predict(self, omega_z: float, dt: float,
                omega_ackermann: float = None):
        """EKF prediction step using unicycle kinematic model.

        Parameters
        ----------
        omega_z : float
            Bias-corrected gyroscope yaw rate [rad/s].
            Used for heading prediction ONLY if omega_ackermann is None.
        dt : float
            Time step [s]. Must be > 0.
        omega_ackermann : float, optional
            Heading rate from Ackermann kinematics: v·tan(δ)/L [rad/s].
            When provided, this REPLACES the gyro for heading prediction.
            This is essential on hardware where motor vibrations cause
            dynamic gyro bias (~0.1 rad/s) that corrupts heading.
        """
        if dt <= 0.0:
            return

        x, y, theta, v = self.x

        # Store pre-prediction heading for Ackermann update
        self._theta_pre_predict = theta

        cos_th = math.cos(theta)
        sin_th = math.sin(theta)

        # ---- Choose heading rate source ----
        # Ackermann kinematics is preferred when available because
        # the gyro has significant dynamic bias from motor vibrations.
        omega_for_heading = omega_ackermann if omega_ackermann is not None else omega_z

        # ---- State prediction f(x, u, dt) ----
        x_new = x + v * cos_th * dt
        y_new = y + v * sin_th * dt
        theta_new = theta + omega_for_heading * dt
        v_new = v  # constant velocity model

        # ---- Jacobian F = ∂f/∂x ----
        F = np.eye(self.N_STATES)
        F[self.IX, self.ITHETA] = -v * sin_th * dt
        F[self.IX, self.IV] = cos_th * dt
        F[self.IY, self.ITHETA] = v * cos_th * dt
        F[self.IY, self.IV] = sin_th * dt

        # ---- Process noise Q (noise-input formulation) ----
        # G maps [w_v, w_ω] noise into state space
        # G = [[cos(θ)·dt,  0   ],
        #      [sin(θ)·dt,  0   ],
        #      [0,          dt  ],
        #      [1,          0   ]]
        G = np.array([
            [cos_th * dt, 0.0],
            [sin_th * dt, 0.0],
            [0.0,         dt],
            [1.0,         0.0],
        ])
        Q_c = np.diag([self._sigma_v ** 2, self._sigma_omega ** 2])
        Q = G @ Q_c @ G.T

        # ---- Covariance prediction ----
        self.P = F @ self.P @ F.T + Q

        # ---- Update state ----
        self.x = np.array([x_new, y_new, self._normalize_angle(theta_new),
                           v_new])

        # Enforce symmetry (numerical stability)
        self.P = 0.5 * (self.P + self.P.T)

    # ==================== Measurement Updates ====================

    def update_velocity(self, v_measured: float, R: float):
        """Encoder velocity measurement update.

        Parameters
        ----------
        v_measured : float
            Measured longitudinal velocity from encoder [m/s].
        R : float
            Measurement noise variance (σ²) [m²/s²].
        """
        H = np.zeros((1, self.N_STATES))
        H[0, self.IV] = 1.0

        z = np.array([v_measured])
        z_pred = np.array([self.x[self.IV]])

        self._joseph_update(z, z_pred, H, np.array([[R]]))

    def update_heading(self, theta_measured: float, R: float):
        """Absolute heading measurement update (angle-wrapped).

        NOTE: DISABLED by default in the node. The Arduino yaw is pure
        gyro integration — using it double-counts gyro data. Enable only
        if an absolute heading source (magnetometer, GPS) is available.

        Parameters
        ----------
        theta_measured : float
            Heading measurement [rad].
        R : float
            Measurement noise variance (σ²) [rad²].
        """
        H = np.zeros((1, self.N_STATES))
        H[0, self.ITHETA] = 1.0

        z = np.array([theta_measured])
        z_pred = np.array([self.x[self.ITHETA]])

        # Angle-wrapped innovation
        innovation = np.array([
            self._normalize_angle(z[0] - z_pred[0])
        ])

        S = H @ self.P @ H.T + np.array([[R]])
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return  # singular — skip

        self.x = self.x + (K @ innovation).flatten()
        self.x[self.ITHETA] = self._normalize_angle(self.x[self.ITHETA])

        # Joseph form
        I_KH = np.eye(self.N_STATES) - K @ H
        R_mat = np.array([[R]])
        self.P = I_KH @ self.P @ I_KH.T + K @ R_mat @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def zupt(self, R: float):
        """Zero-Velocity Update (ZUPT).

        Injects a virtual measurement that v = 0.
        Drives P_vv → R, which via the noise-input Q formulation
        indirectly bounds position covariance growth.

        Parameters
        ----------
        R : float
            ZUPT measurement noise variance (σ²) [m²/s²].
            Use a very small value (e.g. 1e-6) for strong clamping.
        """
        H = np.zeros((1, self.N_STATES))
        H[0, self.IV] = 1.0

        z = np.array([0.0])
        z_pred = np.array([self.x[self.IV]])

        self._joseph_update(z, z_pred, H, np.array([[R]]))

    def update_heading_from_ackermann(self, steering_angle: float,
                                       wheelbase: float,
                                       dt: float, R: float):
        """Ackermann heading rate measurement update.

        Uses the Ackermann kinematic model to compute the expected
        heading based on current velocity and steering angle, then
        treats it as a heading measurement to correct gyro drift.

        How it works:
            θ_expected = θ_pre_predict + (v · tan(δ) / L) · dt
        where θ_pre_predict was saved before the gyro prediction.
        The EKF then compares θ_expected vs θ_current (gyro-predicted)
        and corrects toward a blend of the two.

        This fights dynamic gyro bias from motor vibrations by providing
        an independent heading rate estimate from steering geometry.

        Parameters
        ----------
        steering_angle : float
            Commanded steering angle δ [rad].
        wheelbase : float
            Distance between front and rear axles L [m].
        dt : float
            Time step [s].
        R : float
            Measurement noise variance (σ²) [rad²].
            Higher = trust gyro more, Lower = trust Ackermann more.
            Recommended: 0.05 to 0.5 (rad²).
        """
        if dt <= 0.0:
            return

        v = self.x[self.IV]

        # Ackermann yaw rate: ω = v · tan(δ) / L
        # Clamp steering to avoid tan() blowup near ±90°
        delta_clamped = max(-1.2, min(1.2, steering_angle))
        omega_ackermann = v * math.tan(delta_clamped) / wheelbase

        # Ackermann-predicted heading:
        # "Starting from where we were before prediction, heading should
        #  have changed by ω_ackermann · dt"
        theta_ackermann = self._normalize_angle(
            self._theta_pre_predict + omega_ackermann * dt
        )

        # Use the existing heading update with angle wrapping
        self.update_heading(theta_ackermann, R)

    # ==================== Core EKF Math ====================

    def _joseph_update(self, z: np.ndarray, z_pred: np.ndarray,
                       H: np.ndarray, R: np.ndarray):
        """Generic EKF measurement update using Joseph form.

        Joseph form: P = (I - KH) P (I - KH)^T + K R K^T
        More numerically stable than the standard form P = (I - KH) P.

        Parameters
        ----------
        z : array       Measurement vector.
        z_pred : array  Predicted measurement h(x).
        H : array       Measurement Jacobian.
        R : array       Measurement noise covariance.
        """
        innovation = z - z_pred
        S = H @ self.P @ H.T + R

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return  # singular S — skip this update

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
        """X position [m]."""
        return float(self.x[self.IX])

    @property
    def position_y(self) -> float:
        """Y position [m]."""
        return float(self.x[self.IY])

    @property
    def heading(self) -> float:
        """Heading [rad]."""
        return float(self.x[self.ITHETA])

    @property
    def velocity(self) -> float:
        """Longitudinal velocity [m/s]."""
        return float(self.x[self.IV])

    @property
    def covariance_diagonal(self) -> np.ndarray:
        """Diagonal of P (uncertainties²)."""
        return np.diag(self.P)
