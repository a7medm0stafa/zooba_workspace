"""
EKF Core — Extended Kalman Filter with Bicycle Kinematic Model
================================================================
FILE: localization/localization/ekf_core.py
STATUS: NEW FILE — added as part of EKF localization upgrade
CREATED: 2026-04-24

WHAT THIS FILE DOES:
    Pure-Python/Numpy EKF implementation for 2D vehicle localization.
    This is the math engine — no ROS dependencies, no topics.
    Called by ekf_localization_node.py to run predict/update steps.

WHAT CHANGED (vs. old system):
    - OLD: odometry_node.py used direct yaw = imu_yaw, then x += Δd·cos(yaw)
           → open-loop dead-reckoning, no drift correction
    - NEW: This EKF tracks 5 states including gyro bias, fusing encoder + IMU
           optimally with Kalman gain, plus ZUPT to prevent stationary drift

STATE VECTOR (5 × 1):
    x[0] = x           position X in odom frame [m]
    x[1] = y           position Y in odom frame [m]
    x[2] = θ (theta)   heading / yaw [rad]
    x[3] = v           longitudinal velocity [m/s]
    x[4] = ω_bias      gyroscope Z-axis bias [rad/s]  ← KEY ADDITION

PREDICTION MODEL (bicycle kinematics):
    x'       = x + v·cos(θ)·dt
    y'       = y + v·sin(θ)·dt
    θ'       = θ + (ω_gyro − ω_bias)·dt    ← bias is subtracted!
    v'       = v                             (constant between encoder updates)
    ω_bias'  = ω_bias                        (slow random walk)

MEASUREMENT MODELS:
    1. Encoder velocity:   z = [v_encoder],   h(x) = [v]
    2. Gyro rate:          z = [ω_gyro],      h(x) = [ω_bias]  (bias estimation)
    3. IMU heading:        z = [θ_imu],       h(x) = [θ]       (low trust ~8.5°)
    4. ZUPT (v≈0):         z = [0, 0],        h(x) = [v, 0]    (stationary lock)

COMPATIBILITY:
    - Works with both hardware and simulation
    - No ROS dependency — can be tested standalone with numpy
    - Used by ekf_localization_node.py (the ROS2 wrapper)
"""

import math
import numpy as np


class BicycleEKF:
    """Extended Kalman Filter for 2D bicycle-model vehicle localization.

    Tracks [x, y, θ, v, ω_bias] and fuses encoder + IMU measurements
    to produce a drift-resistant pose estimate.

    Parameters
    ----------
    process_noise : dict
        Keys: 'x', 'y', 'yaw', 'vel', 'gyro_bias' — diagonal Q values.
    encoder_vel_noise : float
        Encoder velocity measurement noise (σ²).
    gyro_rate_noise : float
        Gyroscope angular rate measurement noise (σ²).
    imu_yaw_noise : float
        IMU complementary-filter heading measurement noise (σ²).
    zupt_noise : float
        Zero-velocity virtual measurement noise (σ²).
    """

    # State indices
    IX = 0   # x position
    IY = 1   # y position
    ITHETA = 2  # heading
    IV = 3   # velocity
    IBIAS = 4   # gyro bias

    N_STATES = 5

    def __init__(self, process_noise: dict,
                 encoder_vel_noise: float = 0.05,
                 gyro_rate_noise: float = 0.01,
                 imu_yaw_noise: float = 0.15,
                 zupt_noise: float = 0.001):

        # ---- State & covariance ----
        self.x = np.zeros(self.N_STATES)
        self.P = np.eye(self.N_STATES) * 0.1

        # Initial uncertainty: position known, bias unknown
        self.P[self.IX, self.IX] = 0.01
        self.P[self.IY, self.IY] = 0.01
        self.P[self.ITHETA, self.ITHETA] = 0.01
        self.P[self.IV, self.IV] = 0.1
        self.P[self.IBIAS, self.IBIAS] = 0.01

        # ---- Process noise Q (continuous-time, scaled by dt in predict) ----
        self.q_diag = np.array([
            process_noise.get('x', 0.01),
            process_noise.get('y', 0.01),
            process_noise.get('yaw', 0.005),
            process_noise.get('vel', 0.1),
            process_noise.get('gyro_bias', 0.0001),
        ])

        # ---- Measurement noise variances ----
        self.R_encoder = np.array([[encoder_vel_noise ** 2]])
        self.R_gyro = np.array([[gyro_rate_noise ** 2]])
        self.R_yaw = np.array([[imu_yaw_noise ** 2]])
        self.R_zupt = np.eye(2) * (zupt_noise ** 2)

    def set_state(self, x: float, y: float, theta: float,
                  v: float = 0.0, gyro_bias: float = 0.0):
        """Set the state vector directly (e.g. for initialization)."""
        self.x[self.IX] = x
        self.x[self.IY] = y
        self.x[self.ITHETA] = theta
        self.x[self.IV] = v
        self.x[self.IBIAS] = gyro_bias

    # ==================== Prediction ====================

    def predict(self, omega_gyro: float, dt: float):
        """EKF prediction step using bicycle kinematic model.

        Parameters
        ----------
        omega_gyro : float
            Raw gyroscope reading around Z axis [rad/s].
            The estimated bias is subtracted internally.
        dt : float
            Time step [s]. Must be > 0.
        """
        if dt <= 0.0:
            return

        x, y, theta, v, w_bias = self.x

        # Corrected yaw rate (subtract estimated bias)
        omega_corrected = omega_gyro - w_bias

        # Kinematic prediction
        cos_th = math.cos(theta)
        sin_th = math.sin(theta)

        x_new = x + v * cos_th * dt
        y_new = y + v * sin_th * dt
        theta_new = self._normalize_angle(theta + omega_corrected * dt)
        v_new = v  # velocity model: constant between encoder updates
        bias_new = w_bias  # bias model: slow random walk

        # Jacobian F = ∂f/∂x
        F = np.eye(self.N_STATES)
        F[self.IX, self.ITHETA] = -v * sin_th * dt
        F[self.IX, self.IV] = cos_th * dt
        F[self.IY, self.ITHETA] = v * cos_th * dt
        F[self.IY, self.IV] = sin_th * dt
        F[self.ITHETA, self.IBIAS] = -dt  # ∂θ'/∂ω_bias = -dt

        # Process noise Q (scaled by dt for discrete-time)
        Q = np.diag(self.q_diag * dt)

        # Update state and covariance
        self.x = np.array([x_new, y_new, theta_new, v_new, bias_new])
        self.P = F @ self.P @ F.T + Q

        # Enforce covariance symmetry (numerical stability)
        self.P = 0.5 * (self.P + self.P.T)

    # ==================== Measurement Updates ====================

    def update_velocity(self, v_measured: float):
        """Encoder velocity measurement update.

        Parameters
        ----------
        v_measured : float
            Measured longitudinal velocity from encoder [m/s].
        """
        # Measurement model: z = H·x, where H selects v
        H = np.zeros((1, self.N_STATES))
        H[0, self.IV] = 1.0

        z = np.array([v_measured])
        z_pred = np.array([self.x[self.IV]])

        self._ekf_update(z, z_pred, H, self.R_encoder)

    def update_gyro(self, omega_measured: float):
        """Gyroscope angular rate measurement update.

        This update helps the EKF estimate the gyro bias by constraining
        the bias state based on the measured gyro reading.

        Parameters
        ----------
        omega_measured : float
            Raw gyroscope Z-axis reading [rad/s].
        """
        H = np.zeros((1, self.N_STATES))
        H[0, self.IBIAS] = 1.0

        z = np.array([omega_measured])
        z_pred = np.array([self.x[self.IBIAS]])

        self._ekf_update(z, z_pred, H, self.R_gyro)

    def update_heading(self, theta_measured: float):
        """Absolute heading measurement from IMU complementary filter.

        Used with HIGH measurement noise (low trust) as a soft anchor
        to prevent long-term heading divergence.

        Parameters
        ----------
        theta_measured : float
            Heading from IMU complementary filter [rad].
        """
        H = np.zeros((1, self.N_STATES))
        H[0, self.ITHETA] = 1.0

        z = np.array([theta_measured])
        z_pred = np.array([self.x[self.ITHETA]])

        # Handle angle wrapping in innovation
        innovation = np.array([self._normalize_angle(z[0] - z_pred[0])])

        S = H @ self.P @ H.T + self.R_yaw
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ innovation).flatten()
        self.x[self.ITHETA] = self._normalize_angle(self.x[self.ITHETA])
        I_KH = np.eye(self.N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_yaw @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def zupt(self):
        """Zero-Velocity Update (ZUPT).

        When the vehicle is known to be stationary, inject a virtual
        measurement that v = 0 and yaw_rate = 0. This prevents position
        drift while stopped — the primary fix for the lateral drift problem.
        """
        H = np.zeros((2, self.N_STATES))
        H[0, self.IV] = 1.0      # velocity = 0
        H[1, self.ITHETA] = 1.0  # constrain yaw to prevent drift

        z = np.array([0.0, self.x[self.ITHETA]])
        z_pred = np.array([self.x[self.IV], self.x[self.ITHETA]])

        self._ekf_update(z, z_pred, H, self.R_zupt)

    # ==================== Core EKF Math ====================

    def _ekf_update(self, z: np.ndarray, z_pred: np.ndarray,
                    H: np.ndarray, R: np.ndarray):
        """Generic EKF measurement update (Joseph form for stability).

        Parameters
        ----------
        z : array
            Measurement vector.
        z_pred : array
            Predicted measurement h(x).
        H : array
            Measurement Jacobian.
        R : array
            Measurement noise covariance.
        """
        innovation = z - z_pred
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular S — skip this update
            return

        self.x = self.x + (K @ innovation).flatten()
        self.x[self.ITHETA] = self._normalize_angle(self.x[self.ITHETA])

        # Joseph form: P = (I - KH)P(I - KH)' + KRK'
        I_KH = np.eye(self.N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ==================== Utilities ====================

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

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
        return float(self.x[self.IBIAS])

    @property
    def covariance_diagonal(self) -> np.ndarray:
        """Return the diagonal of the covariance matrix (uncertainties²)."""
        return np.diag(self.P)
