"""
State Plot Node — Live Matplotlib Comparison: Ground Truth vs EKF Estimate
==========================================================================
FILE: localization/localization/state_plot_node.py

PURPOSE:
    Subscribes to /vehicle/state_gt (ground truth) and /vehicle/state
    (EKF estimate) and renders a live matplotlib figure that updates in
    real time while the noisy simulation is running.

    On Ctrl-C (or when the window is closed) the final figure is saved
    to ~/state_plot_<timestamp>.png automatically.

PLOTS (2 × 3 grid):
    ┌──────────────────────┬─────────────────────┬──────────────────────┐
    │  XY Trajectory       │  X position vs time │  Y position vs time  │
    ├──────────────────────┼─────────────────────┼──────────────────────┤
    │  Yaw vs time [°]     │  Velocity vs time   │  Position Error vs t │
    └──────────────────────┴─────────────────────┴──────────────────────┘

    • Blue  — Ground Truth  (/vehicle/state_gt)
    • Green — EKF Estimate  (/vehicle/state)

PARAMETERS:
    gt_topic     (str)   — ground-truth topic (default /vehicle/state_gt)
    ekf_topic    (str)   — EKF output topic   (default /vehicle/state)
    update_rate  (float) — plot refresh [Hz]  (default 5.0)
    max_history  (int)   — max data points kept (default 2000)
    save_on_exit (bool)  — save PNG when done  (default True)
    save_path    (str)   — output file (default ~/state_plot_<ts>.png)

USAGE:
    # While closed_loop_sim_track_noisy.launch.py is running:
    ros2 run localization state_plot_node

    # Custom save path:
    ros2 run localization state_plot_node \\
        --ros-args -p save_path:=/tmp/my_plot.png
"""

import csv
import math
import os
import threading
from collections import deque
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')          # works on most Linux desktops; fallback below
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleState


# ── colour palette ────────────────────────────────────────────────────────────
C_GT  = '#2196F3'   # blue
C_EKF = '#4CAF50'   # green


class StatePlotNode(Node):
    """Live-plotting ROS2 node for state comparison."""

    def __init__(self):
        super().__init__('state_plot_node')

        # ── parameters ────────────────────────────────────────────────
        self.declare_parameter('gt_topic',    '/vehicle/state_gt')
        self.declare_parameter('ekf_topic',   '/vehicle/state')
        self.declare_parameter('update_rate', 5.0)
        self.declare_parameter('max_history', 2000)
        self.declare_parameter('save_on_exit', True)
        self.declare_parameter('save_path',    '')

        gt_topic    = self.get_parameter('gt_topic').value
        ekf_topic   = self.get_parameter('ekf_topic').value
        update_rate = self.get_parameter('update_rate').value
        maxlen      = int(self.get_parameter('max_history').value)

        self.save_on_exit = self.get_parameter('save_on_exit').value
        self.save_path    = self.get_parameter('save_path').value
        if not self.save_path:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_path = os.path.expanduser(f'~/state_plot_{ts}.png')

        # CSV path — same base name, different extension
        self.csv_path = os.path.splitext(self.save_path)[0] + '.csv'

        self._update_interval_ms = int(1000.0 / update_rate)

        # ── data buffers (thread-safe deques) ─────────────────────────
        self._lock = threading.Lock()

        def _buf(): return deque(maxlen=maxlen)

        self.t0 = None   # first timestamp (seconds)

        self.gt_buf  = {'t': _buf(), 'x': _buf(), 'y': _buf(),
                        'yaw': _buf(), 'vel': _buf()}
        self.ekf_buf = {'t': _buf(), 'x': _buf(), 'y': _buf(),
                        'yaw': _buf(), 'vel': _buf()}

        # ── subscriptions ─────────────────────────────────────────────
        self.create_subscription(VehicleState, gt_topic,
                                 lambda m: self._cb(m, self.gt_buf),  10)
        self.create_subscription(VehicleState, ekf_topic,
                                 lambda m: self._cb(m, self.ekf_buf), 10)

        self.get_logger().info(
            f'State Plot Node started\n'
            f'  GT  : {gt_topic}\n'
            f'  EKF : {ekf_topic}\n'
            f'  Rate: {update_rate} Hz  |  history: {maxlen} pts\n'
            f'  PNG : {self.save_path}\n'
            f'  CSV : {self.csv_path}')

    # ── ROS callback ──────────────────────────────────────────────────

    def _cb(self, msg: VehicleState, buf: dict):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp == 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9

        with self._lock:
            if self.t0 is None:
                self.t0 = stamp
            t = stamp - self.t0

            buf['t'].append(t)
            buf['x'].append(msg.x)
            buf['y'].append(msg.y)
            buf['yaw'].append(math.degrees(msg.yaw))
            buf['vel'].append(msg.velocity)

    # ── helpers ───────────────────────────────────────────────────────

    def _snapshot(self):
        """Return thread-safe copies of both buffers."""
        with self._lock:
            def _copy(buf):
                return {k: list(v) for k, v in buf.items()}
            return _copy(self.gt_buf), _copy(self.ekf_buf)

    def save_figure(self, fig):
        """Save the figure to disk as PNG."""
        try:
            fig.savefig(self.save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            self.get_logger().info(f'PNG saved  → {self.save_path}')
        except Exception as e:
            self.get_logger().error(f'Could not save PNG: {e}')

    def save_csv(self):
        """Save all collected GT and EKF data to a CSV file."""
        try:
            gt, ekf = self._snapshot()

            if not gt['t']:
                self.get_logger().warn('No data to save — CSV skipped.')
                return

            # Align lengths (GT and EKF may have different sample counts)
            n_gt  = len(gt['t'])
            n_ekf = len(ekf['t'])

            # Compute EKF position error where both are available
            n_common = min(n_gt, n_ekf)

            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # ---- Header ----
                writer.writerow([
                    'time_s',
                    'gt_x_m', 'gt_y_m', 'gt_yaw_deg', 'gt_vel_ms',
                    'ekf_x_m', 'ekf_y_m', 'ekf_yaw_deg', 'ekf_vel_ms',
                    'pos_error_m',   # Euclidean distance GT vs EKF
                    'yaw_error_deg', # |GT yaw - EKF yaw|
                    'vel_error_ms',  # |GT vel - EKF vel|
                ])

                # ---- Rows where both GT and EKF have data ----
                for i in range(n_common):
                    t       = gt['t'][i]
                    gt_x    = gt['x'][i]
                    gt_y    = gt['y'][i]
                    gt_yaw  = gt['yaw'][i]   # already in degrees
                    gt_vel  = gt['vel'][i]
                    ekf_x   = ekf['x'][i]
                    ekf_y   = ekf['y'][i]
                    ekf_yaw = ekf['yaw'][i]
                    ekf_vel = ekf['vel'][i]

                    pos_err = math.sqrt((gt_x - ekf_x)**2 + (gt_y - ekf_y)**2)

                    yaw_diff = abs(gt_yaw - ekf_yaw)
                    if yaw_diff > 180.0:
                        yaw_diff = 360.0 - yaw_diff  # wrap to [0, 180]

                    vel_err = abs(gt_vel - ekf_vel)

                    writer.writerow([
                        f'{t:.4f}',
                        f'{gt_x:.6f}',  f'{gt_y:.6f}',
                        f'{gt_yaw:.4f}', f'{gt_vel:.6f}',
                        f'{ekf_x:.6f}', f'{ekf_y:.6f}',
                        f'{ekf_yaw:.4f}', f'{ekf_vel:.6f}',
                        f'{pos_err:.6f}',
                        f'{yaw_diff:.4f}',
                        f'{vel_err:.6f}',
                    ])

                # ---- Any extra GT rows beyond EKF coverage ----
                for i in range(n_common, n_gt):
                    writer.writerow([
                        f'{gt["t"][i]:.4f}',
                        f'{gt["x"][i]:.6f}', f'{gt["y"][i]:.6f}',
                        f'{gt["yaw"][i]:.4f}', f'{gt["vel"][i]:.6f}',
                        '', '', '', '',   # no EKF data
                        '', '', '',
                    ])

            self.get_logger().info(
                f'CSV saved   → {self.csv_path}  '
                f'({n_common} rows, {n_gt} GT pts, {n_ekf} EKF pts)')

        except Exception as e:
            self.get_logger().error(f'Could not save CSV: {e}')


# ── Matplotlib figure setup ───────────────────────────────────────────────────

def build_figure():
    """Create and style the figure."""
    BG = '#1a1a2e'
    AX = '#16213e'
    GR = '#0f3460'

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.facecolor': AX,
        'figure.facecolor': BG,
        'axes.edgecolor': '#4a4a7a',
        'axes.labelcolor': '#c0c0e0',
        'xtick.color': '#8080a0',
        'ytick.color': '#8080a0',
        'grid.color': GR,
        'grid.linewidth': 0.6,
        'text.color': '#e0e0ff',
        'legend.facecolor': '#0f0f2a',
        'legend.edgecolor': '#4a4a7a',
        'legend.fontsize': 8,
    })

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.suptitle('Vehicle State: Ground Truth vs EKF Estimate',
                 color='#e0e0ff', fontsize=13, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.38, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    axes = {
        'traj': fig.add_subplot(gs[0, 0]),
        'x':    fig.add_subplot(gs[0, 1]),
        'y':    fig.add_subplot(gs[0, 2]),
        'yaw':  fig.add_subplot(gs[1, 0]),
        'vel':  fig.add_subplot(gs[1, 1]),
        'err':  fig.add_subplot(gs[1, 2]),
    }

    titles = {
        'traj': 'XY Trajectory',
        'x':    'X position [m]',
        'y':    'Y position [m]',
        'yaw':  'Yaw [°]',
        'vel':  'Velocity [m/s]',
        'err':  'Position Error vs GT [m]',
    }
    xlabels = {
        'traj': 'X [m]', 'x': 'Time [s]', 'y': 'Time [s]',
        'yaw':  'Time [s]', 'vel': 'Time [s]', 'err': 'Time [s]',
    }
    ylabels = {
        'traj': 'Y [m]', 'x': 'X [m]', 'y': 'Y [m]',
        'yaw':  'Yaw [°]', 'vel': 'V [m/s]', 'err': 'Euclidean err [m]',
    }

    for key, ax in axes.items():
        ax.set_title(titles[key], color='#c0c0ff', fontsize=10, pad=4)
        ax.set_xlabel(xlabels[key])
        ax.set_ylabel(ylabels[key])
        ax.grid(True, alpha=0.4)
        ax.tick_params(colors='#8080a0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#4a4a7a')

    axes['traj'].set_aspect('equal', adjustable='datalim')

    return fig, axes


def make_animator(node: StatePlotNode, fig, axes):
    """Return a FuncAnimation updater closure."""

    lines = {}

    def _line(ax, color, label, lw=1.5, alpha=1.0, ls='-'):
        ln, = ax.plot([], [], color=color, label=label,
                      linewidth=lw, alpha=alpha, linestyle=ls)
        return ln

    # ---- Trajectory ----
    lines['traj_gt']  = _line(axes['traj'], C_GT,  'Ground Truth', lw=2.0)
    lines['traj_ekf'] = _line(axes['traj'], C_EKF, 'EKF Estimate', lw=1.8, ls='--', alpha=0.8)

    # Current-position dots
    lines['dot_gt'],  = axes['traj'].plot([], [], 'o', color=C_GT,  ms=7, zorder=5)
    lines['dot_ekf'], = axes['traj'].plot([], [], 'o', color=C_EKF, ms=7, zorder=5)

    # ---- Time-series ----
    for key in ('x', 'y', 'yaw', 'vel'):
        lines[f'{key}_gt']  = _line(axes[key], C_GT,  'Ground Truth', lw=2.0)
        lines[f'{key}_ekf'] = _line(axes[key], C_EKF, 'EKF Estimate', lw=1.8, ls='--', alpha=0.8)

    # ---- Error ----
    lines['err_ekf'] = _line(axes['err'], C_EKF, 'EKF position error', lw=1.8, ls='--', alpha=0.8)

    # Legends
    for key in ('traj', 'x', 'y', 'yaw', 'vel', 'err'):
        axes[key].legend(loc='upper left')

    # Stats text
    stats_text = axes['traj'].text(
        0.02, 0.02, '', transform=axes['traj'].transAxes,
        color='#e0e0ff', fontsize=8,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f0f2a', alpha=0.7))

    def _update_line(ln, xdata, ydata):
        ln.set_data(xdata, ydata)

    def animate(_frame):
        gt, ekf = node._snapshot()

        if not gt['t']:
            return list(lines.values()) + [stats_text]

        t_gt  = np.asarray(gt['t'])
        t_ekf = np.asarray(ekf['t']) if ekf['t'] else np.array([])

        # ---- Trajectory ----
        _update_line(lines['traj_gt'],  gt['x'],  gt['y'])
        _update_line(lines['traj_ekf'], ekf['x'], ekf['y'])

        if gt['x']:
            lines['dot_gt'].set_data([gt['x'][-1]],  [gt['y'][-1]])
        if ekf['x']:
            lines['dot_ekf'].set_data([ekf['x'][-1]], [ekf['y'][-1]])

        # ---- Time-series ----
        for key in ('x', 'y', 'yaw', 'vel'):
            _update_line(lines[f'{key}_gt'],  t_gt,  gt[key])
            if len(t_ekf):
                _update_line(lines[f'{key}_ekf'], t_ekf, ekf[key])

        # ---- EKF position error vs GT ----
        def _pos_err(ref, est):
            min_len = min(len(ref['t']), len(est['t']))
            if min_len == 0:
                return np.array([]), np.array([])
            t  = np.asarray(ref['t'][:min_len])
            ex = np.asarray(ref['x'][:min_len]) - np.asarray(est['x'][:min_len])
            ey = np.asarray(ref['y'][:min_len]) - np.asarray(est['y'][:min_len])
            return t, np.sqrt(ex**2 + ey**2)

        te_e, err_e = _pos_err(gt, ekf)
        _update_line(lines['err_ekf'], te_e, err_e)

        # ---- Auto-scale ----
        for ax in axes.values():
            ax.relim()
            ax.autoscale_view()

        # ---- Stats box ----
        if len(err_e):
            mean_e = float(np.mean(err_e))
            max_e  = float(np.max(err_e))
            rms_e  = float(np.sqrt(np.mean(err_e**2)))
            stats_text.set_text(
                f'EKF position error vs GT\n'
                f'  mean = {mean_e:.4f} m\n'
                f'  max  = {max_e:.4f} m\n'
                f'  RMS  = {rms_e:.4f} m\n'
                f'  pts  = {len(gt["t"])}')
        else:
            stats_text.set_text(
                f'Collecting data …\n{len(gt["t"])} GT pts received')

        return list(lines.values()) + [stats_text]

    return animate


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = StatePlotNode()

    # Spin ROS in a background thread so matplotlib can own the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, axes = build_figure()

    anim_fn = make_animator(node, fig, axes)
    ani = animation.FuncAnimation(
        fig, anim_fn,
        interval=node._update_interval_ms,
        blit=False,
        cache_frame_data=False)

    def _on_close(event):
        if node.save_on_exit:
            node.save_figure(fig)
            node.save_csv()
        rclpy.shutdown()

    fig.canvas.mpl_connect('close_event', _on_close)

    node.get_logger().info('Plot window open — close it or press Ctrl-C to exit.')

    try:
        plt.show()   # blocks until window closed
    except KeyboardInterrupt:
        pass
    finally:
        if node.save_on_exit:
            node.save_figure(fig)
            node.save_csv()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
