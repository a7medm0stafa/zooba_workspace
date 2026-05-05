function plot_perception_kpi(kpi_dir)
% PLOT_PERCEPTION_KPI  Generate 3 publication-ready KPI plots for the
%                      Zooba perception system final report.
%
%   plot_perception_kpi()          — uses ~/zooba_kpi/ as default
%   plot_perception_kpi(kpi_dir)   — custom path to KPI CSV folder
%
%   Reads:
%       <kpi_dir>/traffic_light_kpi.csv
%       <kpi_dir>/sign_detection_kpi.csv
%
%   Produces 3 figures (saved as PNG):
%       1. Latency bar chart   (TL Detection, TL Tracking, Sign Detection)
%       2. Detection rate      (TL and Sign over time)
%       3. Confidence          (TL colour ratio and Sign vote confidence)
%
%   Authors: Zooba Autonomous Vehicle Team

    if nargin < 1
        kpi_dir = fullfile(getenv('HOME'), 'zooba_kpi');
    end

    tl_file   = fullfile(kpi_dir, 'traffic_light_kpi.csv');
    sign_file = fullfile(kpi_dir, 'sign_detection_kpi.csv');

    % ── Colour palette ────────────────────────────────────────────
    C.tl_det   = [0.204 0.541 0.741];   % Steel blue
    C.tl_trk   = [0.110 0.725 0.545];   % Emerald green
    C.sign     = [0.859 0.380 0.310];   % Coral red
    C.grid     = [0.85  0.85  0.85 ];   % Light grey grid
    C.bg       = [0.98  0.98  0.98 ];   % Near-white background
    C.err      = [0.3   0.3   0.3  ];   % Dark grey for error bars

    % ── Read CSVs ─────────────────────────────────────────────────
    fprintf('Reading %s ...\n', tl_file);
    tl = readtable(tl_file);
    fprintf('  -> %d rows\n', height(tl));

    fprintf('Reading %s ...\n', sign_file);
    sg = readtable(sign_file);
    fprintf('  -> %d rows\n', height(sg));

    % ── Compute time axes (seconds from start) ────────────────────
    tl_t0 = tl.timestamp(1);
    tl_time = tl.timestamp - tl_t0;

    sg_t0 = sg.timestamp(1);
    sg_time = sg.timestamp - sg_t0;

    % ── Split TL latencies by mode ────────────────────────────────
    is_det = strcmp(tl.mode, 'DETECTION');
    is_trk = strcmp(tl.mode, 'TRACKING');

    det_latencies = tl.detection_latency_ms(is_det);
    trk_latencies = tl.tracking_latency_ms(is_trk);
    sign_latencies = sg.latency_ms;

    % ══════════════════════════════════════════════════════════════
    % FIGURE 1 — Latency Bar Chart
    % ══════════════════════════════════════════════════════════════
    fig1 = figure('Name', 'Perception Latency', ...
                  'Position', [100, 300, 700, 500], ...
                  'Color', 'w');

    means = [mean(det_latencies), mean(trk_latencies), mean(sign_latencies)];
    stds  = [std(det_latencies),  std(trk_latencies),  std(sign_latencies)];
    labels = {'TL Detection', 'TL Tracking', 'Sign Detection'};
    colors = [C.tl_det; C.tl_trk; C.sign];

    b = bar(1:3, means, 0.6, 'FaceColor', 'flat', 'EdgeColor', 'none');
    b.CData = colors;
    hold on;

    % Error bars (±1σ)
    er = errorbar(1:3, means, stds, 'LineStyle', 'none', ...
                  'Color', C.err, 'LineWidth', 1.5, 'CapSize', 12);

    % Real-time threshold line (50 ms)
    yline(50, '--', '50 ms real-time', ...
          'Color', [0.6 0.2 0.2], 'LineWidth', 1.2, ...
          'LabelHorizontalAlignment', 'left', ...
          'FontSize', 10, 'FontWeight', 'bold');

    % Stats annotation
    stats_str = '';
    for k = 1:3
        switch k
            case 1, data = det_latencies;
            case 2, data = trk_latencies;
            case 3, data = sign_latencies;
        end
        stats_str = [stats_str, sprintf('%s:\n', labels{k}), ...
                     sprintf('  Mean=%.1f  Std=%.1f  P95=%.1f  Max=%.1f\n', ...
                     mean(data), std(data), prctile(data, 95), max(data))];
    end
    annotation('textbox', [0.58, 0.60, 0.38, 0.30], ...
               'String', stats_str, ...
               'FitBoxToText', 'on', ...
               'BackgroundColor', [1 1 1 0.85], ...
               'EdgeColor', [0.7 0.7 0.7], ...
               'FontSize', 8, ...
               'FontName', 'Consolas', ...
               'Interpreter', 'none');

    set(gca, 'XTickLabel', labels, 'FontSize', 11, ...
             'Box', 'off', 'Color', C.bg);
    ylabel('Latency (ms)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Perception Pipeline Latency', ...
          'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    set(gca, 'GridColor', C.grid, 'GridAlpha', 0.7);
    hold off;

    % ══════════════════════════════════════════════════════════════
    % FIGURE 2 — Detection Rate Over Time
    % ══════════════════════════════════════════════════════════════
    fig2 = figure('Name', 'Detection Rate', ...
                  'Position', [150, 250, 900, 450], ...
                  'Color', 'w');

    plot(tl_time, tl.detection_rate_pct, '-', ...
         'Color', C.tl_det, 'LineWidth', 1.8);
    hold on;
    plot(sg_time, sg.detection_rate_pct, '-', ...
         'Color', C.sign, 'LineWidth', 1.8);

    % Final values annotation
    tl_final = tl.detection_rate_pct(end);
    sg_final = sg.detection_rate_pct(end);

    % Horizontal reference line
    yline(80, ':', 'Target 80%', ...
          'Color', [0.4 0.4 0.4], 'LineWidth', 1.0, ...
          'LabelHorizontalAlignment', 'left', ...
          'FontSize', 9);

    % Final value markers
    plot(tl_time(end), tl_final, 'o', ...
         'MarkerSize', 8, 'MarkerFaceColor', C.tl_det, ...
         'MarkerEdgeColor', 'w', 'LineWidth', 1.5);
    text(tl_time(end), tl_final + 3, sprintf('%.1f%%', tl_final), ...
         'Color', C.tl_det, 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'right');

    plot(sg_time(end), sg_final, 'o', ...
         'MarkerSize', 8, 'MarkerFaceColor', C.sign, ...
         'MarkerEdgeColor', 'w', 'LineWidth', 1.5);
    text(sg_time(end), sg_final - 3, sprintf('%.1f%%', sg_final), ...
         'Color', C.sign, 'FontSize', 10, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'right');

    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Detection Rate (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Cumulative Detection Rate', ...
          'FontSize', 14, 'FontWeight', 'bold');
    legend({'Traffic Light', 'Sign Detection'}, ...
           'Location', 'southeast', 'FontSize', 10);
    ylim([0, 105]);
    grid on;
    set(gca, 'GridColor', C.grid, 'GridAlpha', 0.7, ...
             'FontSize', 11, 'Box', 'off', 'Color', C.bg);
    hold off;

    % ══════════════════════════════════════════════════════════════
    % FIGURE 3 — Confidence Over Time
    % ══════════════════════════════════════════════════════════════
    fig3 = figure('Name', 'Detection Confidence', ...
                  'Position', [200, 200, 900, 450], ...
                  'Color', 'w');

    % ── Left y-axis: TL colour ratio ──
    yyaxis left;
    plot(tl_time, tl.best_color_ratio, '-', ...
         'Color', C.tl_det, 'LineWidth', 1.2);
    hold on;
    % Running mean (50-sample window)
    if height(tl) >= 50
        tl_smooth = movmean(tl.best_color_ratio, 50);
        plot(tl_time, tl_smooth, '-', ...
             'Color', C.tl_det * 0.7, 'LineWidth', 2.0);
    end
    ylabel('TL Colour Ratio', 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'YColor', C.tl_det);
    ylim([0, 1.05]);

    % ── Right y-axis: Sign vote confidence ──
    yyaxis right;
    plot(sg_time, sg.vote_confidence, '-', ...
         'Color', C.sign, 'LineWidth', 1.2);
    hold on;
    if height(sg) >= 50
        sg_smooth = movmean(sg.vote_confidence, 50);
        plot(sg_time, sg_smooth, '-', ...
             'Color', C.sign * 0.7, 'LineWidth', 2.0);
    end
    ylabel('Sign Vote Confidence', 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'YColor', C.sign);
    ylim([0, 1.05]);

    % ── Shared formatting ──
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Detection Confidence', ...
          'FontSize', 14, 'FontWeight', 'bold');
    legend({'TL Colour Ratio', 'TL (smoothed)', ...
            'Sign Confidence', 'Sign (smoothed)'}, ...
           'Location', 'northeast', 'FontSize', 9);
    grid on;
    set(gca, 'GridColor', C.grid, 'GridAlpha', 0.7, ...
             'FontSize', 11, 'Box', 'off');
    hold off;

    % ══════════════════════════════════════════════════════════════
    % Export to PNG
    % ══════════════════════════════════════════════════════════════
    out_dir = fullfile(kpi_dir, 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    exportgraphics(fig1, fullfile(out_dir, 'latency_bar_chart.png'), ...
                   'Resolution', 300);
    exportgraphics(fig2, fullfile(out_dir, 'detection_rate.png'), ...
                   'Resolution', 300);
    exportgraphics(fig3, fullfile(out_dir, 'confidence.png'), ...
                   'Resolution', 300);

    fprintf('\n✓ Plots saved to %s\n', out_dir);
    fprintf('  - latency_bar_chart.png\n');
    fprintf('  - detection_rate.png\n');
    fprintf('  - confidence.png\n');
end
