#!/usr/bin/env python3
"""
plot_perception_kpi.py

Generate 3 publication-ready KPI plots for the Zooba perception system final report.
Reads traffic_light_kpi.csv and sign_detection_kpi.csv from ~/zooba_kpi/ and saves
PNG plots to ~/zooba_kpi/plots/.
"""

import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filepath):
    data = {}
    if not os.path.exists(filepath):
        print(f"Warning: file not found {filepath}")
        return data
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return data
            
        for h in headers:
            data[h.strip()] = []
        for row in reader:
            if len(row) != len(headers):
                continue
            for i, val in enumerate(row):
                data[headers[i].strip()].append(val.strip())
    return data

def moving_average(a, n=50):
    if len(a) == 0:
        return np.array([])
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_perception_kpi(kpi_dir):
    tl_file = os.path.join(kpi_dir, 'traffic_light_kpi.csv')
    sign_file = os.path.join(kpi_dir, 'sign_detection_kpi.csv')
    
    tl_data = read_csv(tl_file)
    sg_data = read_csv(sign_file)
    
    if not tl_data and not sg_data:
        print("No CSV files found or empty in:", kpi_dir)
        return

    # Colors
    c_tl_det = '#348abd' # Steel blue
    c_tl_trk = '#1cba8b' # Emerald green
    c_sign = '#db604f'   # Coral red
    
    # Process TL
    if tl_data and 'timestamp' in tl_data:
        tl_time = np.array([float(x) for x in tl_data['timestamp']])
        if len(tl_time) > 0:
            tl_time = tl_time - tl_time[0]
        
        tl_latency = np.array([float(x) for x in tl_data['latency_ms']])
        tl_mode = np.array(tl_data['mode'])
        tl_det_lat = np.array([float(x) for x in tl_data['detection_latency_ms']])
        tl_trk_lat = np.array([float(x) for x in tl_data['tracking_latency_ms']])
        
        det_latencies = tl_det_lat[tl_mode == 'DETECTION']
        trk_latencies = tl_trk_lat[tl_mode == 'TRACKING']
        
        tl_rate = np.array([float(x) for x in tl_data['detection_rate_pct']])
        tl_ratio = np.array([float(x) for x in tl_data['best_color_ratio']])
    else:
        det_latencies, trk_latencies = np.array([]), np.array([])
        tl_time, tl_rate, tl_ratio = np.array([]), np.array([]), np.array([])

    # Process SG
    if sg_data and 'timestamp' in sg_data:
        sg_time = np.array([float(x) for x in sg_data['timestamp']])
        if len(sg_time) > 0:
            sg_time = sg_time - sg_time[0]
            
        sign_latencies = np.array([float(x) for x in sg_data['latency_ms']])
        sg_rate = np.array([float(x) for x in sg_data['detection_rate_pct']])
        sg_conf = np.array([float(x) for x in sg_data['vote_confidence']])
    else:
        sign_latencies = np.array([])
        sg_time, sg_rate, sg_conf = np.array([]), np.array([]), np.array([])

    out_dir = os.path.join(kpi_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    
    # Helper to compute stats safely
    def get_stats(arr):
        if len(arr) == 0: return 0.0, 0.0, 0.0, 0.0
        return np.mean(arr), np.std(arr), np.percentile(arr, 95), np.max(arr)
        
    det_mean, det_std, det_p95, det_max = get_stats(det_latencies)
    trk_mean, trk_std, trk_p95, trk_max = get_stats(trk_latencies)
    sign_mean, sign_std, sign_p95, sign_max = get_stats(sign_latencies)

    # ══════════════════════════════════════════════════════════════
    # FIGURE 1 - Latency Bar Chart
    # ══════════════════════════════════════════════════════════════
    plt.figure(figsize=(10, 6))
    
    labels = ['TL Detection', 'TL Tracking', 'Sign Detection']
    means = [det_mean, trk_mean, sign_mean]
    stds = [det_std, trk_std, sign_std]
    colors = [c_tl_det, c_tl_trk, c_sign]
    
    x_pos = np.arange(len(labels))
    plt.bar(x_pos, means, yerr=stds, color=colors, alpha=0.8, capsize=10, width=0.6)
    plt.axhline(y=50, color='#800000', linestyle='--', linewidth=1.5, label='50 ms real-time')
    
    plt.ylabel('Latency (ms)', fontweight='bold', fontsize=12)
    plt.title('Perception Pipeline Latency', fontweight='bold', fontsize=14)
    plt.xticks(x_pos, labels, fontweight='bold', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    stats_str = (
        f"TL Detection:\n  Mean={det_mean:.1f}  Std={det_std:.1f}  P95={det_p95:.1f}  Max={det_max:.1f}\n\n"
        f"TL Tracking:\n  Mean={trk_mean:.1f}  Std={trk_std:.1f}  P95={trk_p95:.1f}  Max={trk_max:.1f}\n\n"
        f"Sign Detection:\n  Mean={sign_mean:.1f}  Std={sign_std:.1f}  P95={sign_p95:.1f}  Max={sign_max:.1f}"
    )
    # Position the text box appropriately so it doesn't overlap too much
    plt.text(1.3, max([m+s for m,s in zip(means, stds)] + [50]) * 0.4, stats_str, 
             fontsize=10, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'latency_bar_chart.png'), dpi=300)
    plt.close()

    # ══════════════════════════════════════════════════════════════
    # FIGURE 2 - Detection Rate
    # ══════════════════════════════════════════════════════════════
    plt.figure(figsize=(10, 5))
    if len(tl_time) > 0:
        plt.plot(tl_time, tl_rate, label='Traffic Light', color=c_tl_det, linewidth=2)
        plt.plot(tl_time[-1], tl_rate[-1], 'o', color=c_tl_det, markersize=8)
        plt.annotate(f"{tl_rate[-1]:.1f}%", (tl_time[-1], tl_rate[-1]), 
                     xytext=(-10, 10), textcoords='offset points', color=c_tl_det, fontweight='bold')
        
    if len(sg_time) > 0:
        plt.plot(sg_time, sg_rate, label='Sign Detection', color=c_sign, linewidth=2)
        plt.plot(sg_time[-1], sg_rate[-1], 'o', color=c_sign, markersize=8)
        plt.annotate(f"{sg_rate[-1]:.1f}%", (sg_time[-1], sg_rate[-1]), 
                     xytext=(-10, -15), textcoords='offset points', color=c_sign, fontweight='bold')
        
    plt.axhline(y=80, color='gray', linestyle=':', linewidth=1.5, label='Target 80%')
    plt.ylim(0, 105)
    plt.xlabel('Time (s)', fontweight='bold', fontsize=12)
    plt.ylabel('Detection Rate (%)', fontweight='bold', fontsize=12)
    plt.title('Cumulative Detection Rate', fontweight='bold', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'detection_rate.png'), dpi=300)
    plt.close()

    # ══════════════════════════════════════════════════════════════
    # FIGURE 3 - Confidence
    # ══════════════════════════════════════════════════════════════
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    if len(tl_time) > 0:
        ax1.plot(tl_time, tl_ratio, color=c_tl_det, linewidth=1, alpha=0.5, label='TL Colour Ratio')
        if len(tl_ratio) >= 50:
            smooth_tl = moving_average(tl_ratio, 50)
            ax1.plot(tl_time[49:], smooth_tl, color=c_tl_det, linewidth=2, label='TL (smoothed)')
            
    ax1.set_xlabel('Time (s)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('TL Colour Ratio', fontweight='bold', fontsize=12, color=c_tl_det)
    ax1.tick_params(axis='y', labelcolor=c_tl_det)
    ax1.set_ylim(0, 1.05)
    
    ax2 = ax1.twinx()
    if len(sg_time) > 0:
        ax2.plot(sg_time, sg_conf, color=c_sign, linewidth=1, alpha=0.5, label='Sign Confidence')
        if len(sg_conf) >= 50:
            smooth_sg = moving_average(sg_conf, 50)
            ax2.plot(sg_time[49:], smooth_sg, color=c_sign, linewidth=2, label='Sign (smoothed)')
            
    ax2.set_ylabel('Sign Vote Confidence', fontweight='bold', fontsize=12, color=c_sign)
    ax2.tick_params(axis='y', labelcolor=c_sign)
    ax2.set_ylim(0, 1.05)
    
    plt.title('Detection Confidence', fontweight='bold', fontsize=14)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    ax1.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confidence.png'), dpi=300)
    plt.close()

    print(f"\n✓ Plots saved to {out_dir}")
    print("  - latency_bar_chart.png")
    print("  - detection_rate.png")
    print("  - confidence.png")

if __name__ == '__main__':
    kpi_dir = os.path.expanduser('~/zooba_workspace/zooba_kpi') # Adjust default dir
    if len(sys.argv) > 1:
        kpi_dir = sys.argv[1]
    plot_perception_kpi(kpi_dir)
