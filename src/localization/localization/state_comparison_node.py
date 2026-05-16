"""
State Comparison Node — Terminal Printer for Noisy Simulation Pipeline
=======================================================================
FILE: localization/localization/state_comparison_node.py

PURPOSE:
    Subscribes to all three VehicleState topics produced by the noisy
    simulation pipeline and prints a nicely formatted comparison table
    to the terminal so you can directly compare:

        ┌───────────────────┬─────────────────────────────────────────────┐
        │                   │  Ground Truth   │  Noisy Sensor │  EKF Est. │
        ├───────────────────┼─────────────────┼───────────────┼───────────┤
        │ x    [m]          │    1.234        │    1.289      │   1.237   │
        │ y    [m]          │   -0.052        │   -0.003      │  -0.049   │
        │ yaw  [°]          │   12.34         │   12.76       │   12.38   │
        │ vel  [m/s]        │    1.500        │    1.548      │   1.503   │
        │ yaw_rate [rad/s]  │    0.012        │    0.023      │   0.014   │
        │ steer [°]         │    2.50         │    2.92       │   2.54    │
        └───────────────────┴─────────────────┴───────────────┴───────────┘
        │ err_noise  pos=0.062 m  yaw=0.41°   vel=0.048 m/s              │
        │ err_ekf    pos=0.003 m  yaw=0.04°   vel=0.003 m/s              │
        └─────────────────────────────────────────────────────────────────┘

TOPICS:
    /vehicle/state_gt    — perfect Gazebo ground truth
    /vehicle/state_noisy — after Gaussian noise injection
    /vehicle/state       — after EKF filtering

PARAMETERS:
    gt_topic     (str)   — ground-truth topic   (default /vehicle/state_gt)
    noisy_topic  (str)   — noisy topic          (default /vehicle/state_noisy)
    ekf_topic    (str)   — EKF output topic     (default /vehicle/state)
    print_rate   (float) — table refresh rate Hz (default 2.0)
    use_color    (bool)  — ANSI color codes in output (default True)

USAGE:
    Launched automatically by closed_loop_sim_track_noisy.launch.py.
    Can also be run standalone:
        ros2 run localization state_comparison_node
"""

import math

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleState


# ============================================================
# ANSI colour helpers
# ============================================================
_RESET  = '\033[0m'
_BOLD   = '\033[1m'
_CYAN   = '\033[96m'
_GREEN  = '\033[92m'
_YELLOW = '\033[93m'
_RED    = '\033[91m'
_WHITE  = '\033[97m'
_DIM    = '\033[2m'


def _col(text: str, code: str, use_color: bool) -> str:
    return f'{code}{text}{_RESET}' if use_color else text


class StateComparisonNode(Node):
    """Prints a three-column comparison: GT | Noisy | EKF."""

    def __init__(self):
        super().__init__('state_comparison_node')

        # ============================================================
        # Parameters
        # ============================================================
        self.declare_parameter('gt_topic',    '/vehicle/state_gt')
        self.declare_parameter('noisy_topic', '/vehicle/state_noisy')
        self.declare_parameter('ekf_topic',   '/vehicle/state')
        self.declare_parameter('print_rate',  2.0)
        self.declare_parameter('use_color',   True)

        gt_topic    = self.get_parameter('gt_topic').value
        noisy_topic = self.get_parameter('noisy_topic').value
        ekf_topic   = self.get_parameter('ekf_topic').value
        self.print_rate  = self.get_parameter('print_rate').value
        self.use_color   = self.get_parameter('use_color').value

        # ============================================================
        # State storage
        # ============================================================
        self.gt:    VehicleState | None = None
        self.noisy: VehicleState | None = None
        self.ekf:   VehicleState | None = None

        self._print_count = 0

        # ============================================================
        # Subscribers
        # ============================================================
        self.create_subscription(VehicleState, gt_topic,
                                 self._gt_cb,    10)
        self.create_subscription(VehicleState, noisy_topic,
                                 self._noisy_cb, 10)
        self.create_subscription(VehicleState, ekf_topic,
                                 self._ekf_cb,   10)

        # ============================================================
        # Timer
        # ============================================================
        self.create_timer(1.0 / self.print_rate, self._print_cb)

        self.get_logger().info(
            f'State Comparison Node started  '
            f'[GT={gt_topic}  Noisy={noisy_topic}  EKF={ekf_topic}]  '
            f'@ {self.print_rate:.0f} Hz')

    # ----------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------

    def _gt_cb(self,    msg: VehicleState): self.gt    = msg
    def _noisy_cb(self, msg: VehicleState): self.noisy = msg
    def _ekf_cb(self,   msg: VehicleState): self.ekf   = msg

    # ----------------------------------------------------------------
    # Printer
    # ----------------------------------------------------------------

    def _print_cb(self):
        """Print the side-by-side comparison table."""
        # Need at least ground truth to show anything meaningful
        if self.gt is None:
            self.get_logger().info(
                'Waiting for /vehicle/state_gt ...', throttle_duration_sec=3.0)
            return

        uc = self.use_color
        self._print_count += 1

        # ---- Pull values (use NaN sentinels when topic not yet received) ----
        def _v(msg, field):
            return getattr(msg, field) if msg is not None else float('nan')

        gt_x  = _v(self.gt, 'x');    n_x  = _v(self.noisy, 'x');    e_x  = _v(self.ekf, 'x')
        gt_y  = _v(self.gt, 'y');    n_y  = _v(self.noisy, 'y');    e_y  = _v(self.ekf, 'y')
        gt_yaw = _v(self.gt,'yaw');  n_yaw = _v(self.noisy,'yaw'); e_yaw = _v(self.ekf,'yaw')
        gt_v  = _v(self.gt,'velocity'); n_v  = _v(self.noisy,'velocity'); e_v  = _v(self.ekf,'velocity')
        gt_yr = _v(self.gt,'yaw_rate'); n_yr = _v(self.noisy,'yaw_rate'); e_yr = _v(self.ekf,'yaw_rate')
        gt_st = _v(self.gt,'steering_angle')
        n_st  = _v(self.noisy,'steering_angle')
        e_st  = _v(self.ekf,'steering_angle')

        # ---- Position errors ----
        def pos_err(x1, y1, x2, y2):
            if math.isnan(x1) or math.isnan(x2): return float('nan')
            return math.sqrt((x1-x2)**2 + (y1-y2)**2)

        def ang_err(a, b):
            if math.isnan(a) or math.isnan(b): return float('nan')
            d = a - b
            while d >  math.pi: d -= 2*math.pi
            while d < -math.pi: d += 2*math.pi
            return abs(math.degrees(d))

        noise_pos_err = pos_err(gt_x, gt_y, n_x, n_y)
        noise_yaw_err = ang_err(gt_yaw, n_yaw)
        noise_vel_err = abs(gt_v - n_v) if not (math.isnan(gt_v) or math.isnan(n_v)) else float('nan')

        ekf_pos_err   = pos_err(gt_x, gt_y, e_x, e_y)
        ekf_yaw_err   = ang_err(gt_yaw, e_yaw)
        ekf_vel_err   = abs(gt_v - e_v) if not (math.isnan(gt_v) or math.isnan(e_v)) else float('nan')

        # ---- Color error values ----
        def _err_color(val, good_thresh, warn_thresh):
            if math.isnan(val): return _col('  N/A  ', _DIM, uc)
            if val < good_thresh: return _col(f'{val:7.4f}', _GREEN, uc)
            if val < warn_thresh: return _col(f'{val:7.4f}', _YELLOW, uc)
            return _col(f'{val:7.4f}', _RED, uc)

        # ---- Format helpers ----
        def _f(v, fmt='.3f'):
            return f'{v:{fmt}}' if not math.isnan(v) else '  ---  '

        def _fd(v, fmt='.2f'):   # degrees
            return f'{math.degrees(v):{fmt}}' if not math.isnan(v) else '  ---  '

        # ---- Build table ----
        W = 76   # total width
        sep   = _col('│', _DIM, uc)
        hline = _col('─' * W, _DIM, uc)
        hline2= _col('═' * W, _DIM, uc)

        header_top = _col(f'{"─"*W}', _DIM, uc)

        def row(label, gt_val, n_val, e_val, unit=''):
            L = f'{label:<18s}{unit:<6s}'
            return (f'{sep} {_col(L, _WHITE, uc)} '
                    f'{sep} {_col(gt_val, _CYAN, uc):>12s} '
                    f'{sep} {_col(n_val, _YELLOW, uc):>12s} '
                    f'{sep} {_col(e_val, _GREEN, uc):>12s} '
                    f'{sep}')

        col_header = (
            f'{sep} {"Field":<24s} '
            f'{sep} {_col("Ground Truth", _CYAN,   uc):>12s} '
            f'{sep} {_col("Noisy Sensor", _YELLOW, uc):>12s} '
            f'{sep} {_col("EKF Estimate", _GREEN,  uc):>12s} '
            f'{sep}'
        )

        # EKF improvement factor
        def improvement(noise_e, ekf_e, label):
            if math.isnan(noise_e) or math.isnan(ekf_e) or noise_e < 1e-9:
                return ''
            factor = noise_e / max(ekf_e, 1e-9)
            sym = '↓' if ekf_e < noise_e else '↑'
            col = _GREEN if ekf_e < noise_e else _RED
            return f'  {_col(f"{sym}{factor:.1f}x", col, uc)}'

        print('\n' + hline)
        print(_col(f' STATE COMPARISON  #{self._print_count:04d}', _BOLD, uc))
        print(hline)
        print(col_header)
        print(hline)
        print(row('x',             _f(gt_x),  _f(n_x),  _f(e_x),  '[m]'))
        print(row('y',             _f(gt_y),  _f(n_y),  _f(e_y),  '[m]'))
        print(row('yaw',           _fd(gt_yaw), _fd(n_yaw), _fd(e_yaw), '[°]'))
        print(row('velocity',      _f(gt_v),  _f(n_v),  _f(e_v),  '[m/s]'))
        print(row('yaw_rate',      _f(gt_yr, '.4f'), _f(n_yr, '.4f'), _f(e_yr, '.4f'), '[rad/s]'))
        print(row('steering',      _fd(gt_st, '.2f'), _fd(n_st, '.2f'), _fd(e_st, '.2f'), '[°]'))
        print(hline)

        # ---- Error summary rows ----
        p_ne = _err_color(noise_pos_err, 0.03, 0.10)
        p_ee = _err_color(ekf_pos_err,   0.02, 0.05)
        y_ne = _err_color(noise_yaw_err, 0.5,  2.0)
        y_ee = _err_color(ekf_yaw_err,   0.3,  1.0)
        v_ne = _err_color(noise_vel_err, 0.03, 0.10)
        v_ee = _err_color(ekf_vel_err,   0.02, 0.05)

        imp_pos = improvement(noise_pos_err, ekf_pos_err, 'pos')
        imp_yaw = improvement(noise_yaw_err, ekf_yaw_err, 'yaw')
        imp_vel = improvement(noise_vel_err, ekf_vel_err, 'vel')

        print(f'{sep} {_col("ERROR vs GT", _BOLD, uc):<23s} '
              f'{sep} {"pos [m]":>12s} '
              f'{sep} {"yaw [°]":>12s} '
              f'{sep} {"vel [m/s]":>12s} '
              f'{sep}')
        print(hline)
        print(f'{sep} {_col("  Noise injection:", _YELLOW, uc):<23s} '
              f'{sep} {p_ne:>12s} '
              f'{sep} {y_ne:>12s} '
              f'{sep} {v_ne:>12s} '
              f'{sep}')
        print(f'{sep} {_col("  EKF filtered:   ", _GREEN,  uc):<23s} '
              f'{sep} {p_ee:>12s} '
              f'{sep} {y_ee:>12s} '
              f'{sep} {v_ee:>12s} '
              f'{sep}')
        print(f'{sep} {_col("  EKF improvement:", _BOLD,   uc):<23s} '
              f'{sep}{imp_pos:>13s} '
              f'{sep}{imp_yaw:>13s} '
              f'{sep}{imp_vel:>13s} '
              f'{sep}')
        print(hline)


def main(args=None):
    rclpy.init(args=args)
    node = StateComparisonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
