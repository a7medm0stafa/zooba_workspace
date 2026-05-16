"""
State Plot Node — Live Matplotlib Comparison: Noisy Sensor vs EKF Estimate
==========================================================================
FILE: localization/localization/state_plot_noisy_vs_ekf_node.py

PURPOSE:
    Subscribes to /vehicle/state_noisy (noisy feedback) and /vehicle/state
    (EKF estimate) and renders a live matplotlib figure that updates in
    real time. This allows visualizing how much noise is present and how
    effectively the EKF filters it out.

    On Ctrl-C (or when the window is closed) the final figure is saved
    to ~/state_plot_noisy_<timestamp>.png automatically.

PLOTS (2 × 3 grid):
    ┌──────────────────────┬─────────────────────┬──────────────────────┐
    │  XY Trajectory       │  X position vs time │  Y position vs time  │
    ├──────────────────────┼─────────────────────┼──────────────────────┤
    │  Yaw vs time [°]     │  Velocity vs time   │  Position Error vs t │
    └──────────────────────┴─────────────────────┴──────────────────────┘

    • Orange — Noisy Sensor  (/vehicle/state_noisy)
    • Green  — EKF Estimate  (/vehicle/state)

USAGE:
    ros2 run localization state_plot_noisy_vs_ekf_node
"""

import csv
import math
import os
import threading
from collections import deque
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleState


# ── colour palette ────────────────────────────────────────────────────────────
C_NOISY = '#FF9800'   # orange
C_EKF   = '#4CAF50'   # green


class StatePlotNoisyVsEkfNode(Node):
    """Live-plotting ROS2 node for noisy vs EKF state comparison."""

    def __init__(self):
        super().__init__('state_plot_noisy_vs_ekf_node')

        # ── parameters ────────────────────────────────────────────────
        self.declare_parameter('noisy_topic', '/vehicle/state_noisy')
        self.declare_parameter('ekf_topic',   '/vehicle/state')
        self.declare_parameter('update_rate', 5.0)
        self.declare_parameter('max_history', 2000)
        self.declare_parameter('save_on_exit', True)
        self.declare_parameter('save_path',    '')

        noisy_topic = self.get_parameter('noisy_topic').value
        ekf_topic   = self.get_parameter('ekf_topic').value
        update_rate = self.get_parameter('update_rate').value
        maxlen      = int(self.get_parameter('max_history').value)

        self.save_on_exit = self.get_parameter('save_on_exit').value
        self.save_path    = self.get_parameter('save_path').value
        if not self.save_path:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_path = os.path.expanduser(f'~/state_plot_noisy_{ts}.png')

        # CSV path — same base name, different extension
        self.csv_path = os.path.splitext(self.save_path)[0] + '.csv'

        self._update_interval_ms = int(1000.0 / update_rate)

        # ── data buffers (thread-safe deques) ─────────────────────────
        self._lock = threading.Lock()

        def _buf(): return deque(maxlen=maxlen)

        self.t0 = None   # first timestamp (seconds)

        self.noisy_buf  = {'t': _buf(), 'x': _buf(), 'y': _buf(),
                           'yaw': _buf(), 'vel': _buf()}
        self.ekf_buf    = {'t': _buf(), 'x': _buf(), 'y': _buf(),
                           'yaw': _buf(), 'vel': _buf()}

        # ── subscriptions ─────────────────────────────────────────────
        self.create_subscription(VehicleState, noisy_topic,
                                 lambda m: self._cb(m, self.noisy_buf),  10)
        self.create_subscription(VehicleState, ekf_topic,
                                 lambda m: self._cb(m, self.ekf_buf), 10)

        self.get_logger().info(
            f'State Plot Node (Noisy vs EKF) started\n'
            f'  Noisy: {noisy_topic}\n'
            f'  EKF  : {ekf_topic}\n'
            f'  Rate : {update_rate} Hz  |  history: {maxlen} pts\n'
            f'  PNG  : {self.save_path}\n'
            f'  CSV  : {self.csv_path}')

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
            return _copy(self.noisy_buf), _copy(self.ekf_buf)

    def save_figure(self, fig):
        """Save the figure to disk as PNG."""
        try:
            fig.savefig(self.save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            self.get_logger().info(f'PNG saved  → {self.save_path}')
        except Exception as e:
            self.get_logger().error(f'Could not save PNG: {e}')

    def save_csv(self):
        """Save all collected Noisy and EKF data to a CSV file."""
        try:
            noisy, ekf = self._snapshot()

            if not noisy['t']:
                self.get_logger().warn('No data to save — CSV skipped.')
                return

            n_noisy = len(noisy['t'])
            n_ekf   = len(ekf['t'])
            n_common = min(n_noisy, n_ekf)

            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # ---- Header ----
                writer.writerow([
                    'time_s',
                    'noisy_x_m', 'noisy_y_m', 'noisy_yaw_deg', 'noisy_vel_ms',
                    'ekf_x_m', 'ekf_y_m', 'ekf_yaw_deg', 'ekf_vel_ms',
                    'pos_diff_m',   # Euclidean distance Noisy vs EKF
                    'yaw_diff_deg',
                    'vel_diff_ms',
                ])

                # ---- Rows where both have data ----
                for i in range(n_common):
                    t         = noisy['t'][i]
                    noisy_x   = noisy['x'][i]
                    noisy_y   = noisy['y'][i]
                    noisy_yaw = noisy['yaw'][i]
                    noisy_vel = noisy['vel'][i]
                    ekf_x     = ekf['x'][i]
                    ekf_y     = ekf['y'][i]
                    ekf_yaw   = ekf['yaw'][i]
                    ekf_vel   = ekf['vel'][i]

                    pos_err = math.sqrt((noisy_x - ekf_x)**2 + (noisy_y - ekf_y)**2)

                    yaw_diff = abs(noisy_yaw - ekf_yaw)
                    if yaw_diff > 180.0:
                        yaw_diff = 360.0 - yaw_diff

                    vel_err = abs(noisy_vel - ekf_vel)

                    writer.writerow([
                        f'{t:.4f}',
                        f'{noisy_x:.6f}',  f'{noisy_y:.6f}',
                        f'{noisy_yaw:.4f}', f'{noisy_vel:.6f}',
                        f'{ekf_x:.6f}', f'{ekf_y:.6f}',
                        f'{ekf_yaw:.4f}', f'{ekf_vel:.6f}',
                        f'{pos_err:.6f}',
                        f'{yaw_diff:.4f}',
                        f'{vel_err:.6f}',
                    ])

                # ---- Any extra Noisy rows beyond EKF coverage ----
                for i in range(n_common, n_noisy):
                    writer.writerow([
                        f'{noisy["t"][i]:.4f}',
                        f'{noisy["x"][i]:.6f}', f'{noisy["y"][i]:.6f}',
                        f'{noisy["yaw"][i]:.4f}', f'{noisy["vel"][i]:.6f}',
                        '', '', '', '',
                        '', '', '',
                    ])

            self.get_logger().info(
                f'CSV saved   → {self.csv_path}  '
                f'({n_common} rows, {n_noisy} Noisy pts, {n_ekf} EKF pts)')

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
    fig.suptitle('Vehicle State: Noisy Sensor vs EKF Estimate',
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
        'err':  'Difference (Noisy vs EKF) [m]',
    }
    xlabels = {
        'traj': 'X [m]', 'x': 'Time [s]', 'y': 'Time [s]',
        'yaw':  'Time [s]', 'vel': 'Time [s]', 'err': 'Time [s]',
    }
    ylabels = {
        'traj': 'Y [m]', 'x': 'X [m]', 'y': 'Y [m]',
        'yaw':  'Yaw [°]', 'vel': 'V [m/s]', 'err': 'Distance [m]',
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


def make_animator(node: StatePlotNoisyVsEkfNode, fig, axes):
    """Return a FuncAnimation updater closure."""

    lines = {}

    def _line(ax, color, label, lw=1.5, alpha=1.0, ls='-'):
        ln, = ax.plot([], [], color=color, label=label,
                      linewidth=lw, alpha=alpha, linestyle=ls)
        return ln

    # ---- Trajectory ----
    lines['traj_noisy'] = _line(axes['traj'], C_NOISY, 'Noisy Sensor', lw=2.0)
    lines['traj_ekf']   = _line(axes['traj'], C_EKF, 'EKF Estimate', lw=1.8, ls='--', alpha=0.9)

    # Current-position dots
    lines['dot_noisy'], = axes['traj'].plot([], [], 'o', color=C_NOISY, ms=7, zorder=5)
    lines['dot_ekf'],   = axes['traj'].plot([], [], 'o', color=C_EKF, ms=7, zorder=5)

    # ---- Time-series ----
    for key in ('x', 'y', 'yaw', 'vel'):
        lines[f'{key}_noisy'] = _line(axes[key], C_NOISY, 'Noisy Sensor', lw=2.0)
        lines[f'{key}_ekf']   = _line(axes[key], C_EKF, 'EKF Estimate', lw=1.8, ls='--', alpha=0.9)

    # Legends
    for key in ('traj', 'x', 'y', 'yaw', 'vel'):
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
        noisy, ekf = node._snapshot()

        if not noisy['t']:
            return list(lines.values()) + [stats_text]

        t_noisy = np.asarray(noisy['t'])
        t_ekf   = np.asarray(ekf['t']) if ekf['t'] else np.array([])

        # ---- Trajectory ----
        _update_line(lines['traj_noisy'], noisy['x'], noisy['y'])
        _update_line(lines['traj_ekf'],   ekf['x'], ekf['y'])

        if noisy['x']:
            lines['dot_noisy'].set_data([noisy['x'][-1]],  [noisy['y'][-1]])
        if ekf['x']:
            lines['dot_ekf'].set_data([ekf['x'][-1]], [ekf['y'][-1]])

        # ---- Time-series ----
        for key in ('x', 'y', 'yaw', 'vel'):
            _update_line(lines[f'{key}_noisy'], t_noisy, noisy[key])
            if len(t_ekf):
                _update_line(lines[f'{key}_ekf'], t_ekf, ekf[key])

        # ---- Auto-scale ----
        for ax in axes.values():
            ax.relim()
            ax.autoscale_view()

        # ---- Stats box ----
        if len(t_noisy) > 0 and len(t_ekf) > 0:
            # We still compute basic stats for the text box even though the line is gone
            min_len = min(len(t_noisy), len(t_ekf))
            err_diff = np.sqrt(
                (np.asarray(noisy['x'][:min_len]) - np.asarray(ekf['x'][:min_len]))**2 +
                (np.asarray(noisy['y'][:min_len]) - np.asarray(ekf['y'][:min_len]))**2
            )
            mean_diff = float(np.mean(err_diff))
            max_diff  = float(np.max(err_diff))
            stats_text.set_text(
                f'Difference (Noisy vs EKF)\n'
                f'  mean = {mean_diff:.4f} m\n'
                f'  max  = {max_diff:.4f} m\n'
                f'  pts  = {len(noisy["t"])}')
        else:
            stats_text.set_text(
                f'Collecting data …\n{len(noisy["t"])} Noisy pts received')

        return list(lines.values()) + [stats_text]

    return animate


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = StatePlotNoisyVsEkfNode()

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
