#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

logger = logging.getLogger(__name__)


def add_repo_import_paths():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


add_repo_import_paths()

XPOD_COLOR = '#2E86AB'
BASELINE_COLOR = '#A23B72'
ACCENT_COLOR = '#F18F01'
LIGHT_BG = '#F8F9FA'
DARK_TEXT = '#2D3748'

plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.edgecolor': '#CBD5E0',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'text.color': DARK_TEXT,
    'axes.labelcolor': DARK_TEXT,
    'xtick.color': DARK_TEXT,
    'ytick.color': DARK_TEXT,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})


def style_axes(ax, remove_top_right=True):
    ax.set_facecolor(LIGHT_BG)
    if remove_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)


def load_with_duration(csv_path: str, trace_csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    trace = pd.read_csv(trace_csv, usecols=['xpod_id', 'duration'])
    dur_map = trace.groupby('xpod_id')['duration'].max().to_dict()
    df['duration_s'] = df['xpod_id'].map(dur_map).fillna(0)
    return df


def jct_improvement_pct(x_df: pd.DataFrame, b_df: pd.DataFrame) -> float:
    return (1 - x_df['jct_s'].sum() / b_df['jct_s'].sum()) * 100


def save_figure(fig: Figure, name: str, output_dir: str) -> None:
    fig.patch.set_facecolor('white')
    svg_path = os.path.join(output_dir, f'{name}.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info('Saved: %s.svg', name)


def make_fig1_cold_start_convergence(xp_df: pd.DataFrame, bs_df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    style_axes(ax)

    xp_sorted = xp_df.sort_values('idx').reset_index(drop=True)
    bs_sorted = bs_df.sort_values('idx').reset_index(drop=True)
    xp_cum = xp_sorted['cold_start'].cumsum()
    bs_cum = bs_sorted['cold_start'].cumsum()

    ax.fill_between(xp_sorted['idx'][:200], xp_cum[:200], alpha=0.15, color=XPOD_COLOR)
    ax.fill_between(bs_sorted['idx'][:200], bs_cum[:200], alpha=0.15, color=BASELINE_COLOR)

    ax.plot(xp_sorted['idx'][:200], xp_cum[:200], color=XPOD_COLOR, label='xPod', lw=2.5, solid_capstyle='round')
    ax.plot(bs_sorted['idx'][:200], bs_cum[:200], color=BASELINE_COLOR, label='Random Baseline', lw=2.5, 
            linestyle='--', solid_capstyle='round')

    ax.axhline(5, color=XPOD_COLOR, alpha=0.4, linestyle=':', lw=1.5)
    ax.axhline(10, color=BASELINE_COLOR, alpha=0.4, linestyle=':', lw=1.5)

    ax.set_xlabel('Task Index', fontweight='medium')
    ax.set_ylabel('Cumulative Cold Starts', fontweight='medium')
    ax.set_title('Cold-start Convergence (First 200 Tasks)', pad=15)
    
    legend = ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, borderpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#E2E8F0')

    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=XPOD_COLOR, alpha=0.9, lw=1.2)
    ax.annotate('xPod converges at idx=5',
                xy=(5, 5), xytext=(45, 3),
                fontsize=9, color=XPOD_COLOR, fontweight='medium',
                arrowprops=dict(arrowstyle='->', color=XPOD_COLOR, alpha=0.8, lw=1.5),
                bbox=bbox_props)
    
    bbox_props2 = dict(boxstyle="round,pad=0.3", fc="white", ec=BASELINE_COLOR, alpha=0.9, lw=1.2)
    ax.annotate('Baseline converges at idx=64',
                xy=(64, 10), xytext=(85, 7),
                fontsize=9, color=BASELINE_COLOR, fontweight='medium',
                arrowprops=dict(arrowstyle='->', color=BASELINE_COLOR, alpha=0.8, lw=1.5),
                bbox=bbox_props2)

    return fig


def make_fig2_cumulative_transfer(xp_df: pd.DataFrame, bs_df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    style_axes(ax)

    xp_sorted = xp_df.sort_values('idx').reset_index(drop=True)
    bs_sorted = bs_df.sort_values('idx').reset_index(drop=True)

    xp_cum_gb = xp_sorted['bytes_to_load'].cumsum() / 1024**3
    bs_cum_gb = bs_sorted['bytes_to_load'].cumsum() / 1024**3

    n = len(xp_sorted)
    idx_sample = np.unique(np.concatenate([
        np.arange(0, 200),
        np.linspace(200, n-1, 800).astype(int)
    ]))

    ax.fill_between(xp_sorted['idx'].values[idx_sample], xp_cum_gb.values[idx_sample], alpha=0.15, color=XPOD_COLOR)
    ax.fill_between(bs_sorted['idx'].values[idx_sample], bs_cum_gb.values[idx_sample], alpha=0.15, color=BASELINE_COLOR)

    ax.plot(xp_sorted['idx'].values[idx_sample], xp_cum_gb.values[idx_sample],
            color=XPOD_COLOR, label=f'xPod ({xp_cum_gb.iloc[-1]:.0f} GB total)', lw=2.5, solid_capstyle='round')
    ax.plot(bs_sorted['idx'].values[idx_sample], bs_cum_gb.values[idx_sample],
            color=BASELINE_COLOR, label=f'Random Baseline ({bs_cum_gb.iloc[-1]:.0f} GB total)', lw=2.5, 
            linestyle='--', solid_capstyle='round')

    ax.set_xscale('log')
    ax.set_xlabel('Task Index (log scale)', fontweight='medium')
    ax.set_ylabel('Cumulative Data Transfer (GB)', fontweight='medium')
    ax.set_title('Cumulative Data Transfer (2x Reduction)', pad=15)
    
    legend = ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, borderpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#E2E8F0')

    return fig


def make_fig3_contention_cdf(xp_df: pd.DataFrame, bs_df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    style_axes(ax)

    xp_gpu = xp_df[xp_df['compute_node'].str.startswith('gpu-')]['contention_factor']
    bs_gpu = bs_df[bs_df['compute_node'].str.startswith('gpu-')]['contention_factor']

    xp_sorted_c = np.sort(xp_gpu.values)
    bs_sorted_c = np.sort(bs_gpu.values)
    xp_cdf = np.arange(1, len(xp_sorted_c)+1) / len(xp_sorted_c)
    bs_cdf = np.arange(1, len(bs_sorted_c)+1) / len(bs_sorted_c)

    ax.fill_between(xp_sorted_c, xp_cdf, alpha=0.12, color=XPOD_COLOR)
    ax.fill_between(bs_sorted_c, bs_cdf, alpha=0.12, color=BASELINE_COLOR)

    ax.plot(xp_sorted_c, xp_cdf, color=XPOD_COLOR, label=f'xPod (σ={xp_gpu.std():.1f})', lw=3, solid_capstyle='round')
    ax.plot(bs_sorted_c, bs_cdf, color=BASELINE_COLOR, label=f'Random Baseline (σ={bs_gpu.std():.1f})',
            lw=3, linestyle='--', solid_capstyle='round', dashes=(5, 2))

    xp_p95 = xp_gpu.quantile(0.95)
    bs_p95 = bs_gpu.quantile(0.95)
    
    ax.axvline(x=xp_p95, color=XPOD_COLOR, linestyle=':', lw=2, alpha=0.7)
    ax.axvline(x=bs_p95, color=BASELINE_COLOR, linestyle=':', lw=2, alpha=0.7)
    
    ax.text(xp_p95 + 0.5, 0.93, f'p95 = {xp_p95:.1f}', color=XPOD_COLOR, 
            fontweight='bold', fontsize=10, ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=XPOD_COLOR, alpha=0.9))
    ax.text(bs_p95 - 0.5, 0.85, f'p95 = {bs_p95:.1f}', color=BASELINE_COLOR, 
            fontweight='bold', fontsize=10, ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=BASELINE_COLOR, alpha=0.9))

    ax.set_xlabel('Contention Factor', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontweight='bold', fontsize=12)
    ax.set_title('GPU Contention Distribution: xPod Reduces Tail Risk', pad=20, fontweight='bold', fontsize=14)
    
    legend = ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, borderpad=1.2, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#E2E8F0')
    
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')

    return fig


def make_fig4_jct_percentile(xp_df: pd.DataFrame, bs_df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    style_axes(ax)

    percentiles = [90, 95, 99]
    xp_pcts = [xp_df['jct_s'].quantile(p/100) / 3600 for p in percentiles]
    bs_pcts = [bs_df['jct_s'].quantile(p/100) / 3600 for p in percentiles]

    x = np.arange(len(percentiles))
    width = 0.32

    bars1 = ax.bar(x - width/2, xp_pcts, width, label='xPod', color=XPOD_COLOR, 
                   edgecolor='white', linewidth=2, zorder=3)
    bars2 = ax.bar(x + width/2, bs_pcts, width, label='Random Baseline', color=BASELINE_COLOR, 
                   edgecolor='white', linewidth=2, zorder=3)

    for bar in bars1:
        bar.set_alpha(0.9)
    for bar in bars2:
        bar.set_alpha(0.9)

    for i, (xv, bv) in enumerate(zip(xp_pcts, bs_pcts)):
        pct = (1 - xv/bv)*100 if bv > 0 else 0
        if pct > 0:
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT_COLOR, alpha=0.95, lw=1.5)
            ax.text(i, max(xv, bv)*1.08, f'{pct:.1f}% ↓', ha='center', fontsize=11, color=ACCENT_COLOR, 
                    fontweight='bold', bbox=bbox_props, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'p{p}' for p in percentiles], fontweight='bold', fontsize=12)
    ax.set_xlabel('Percentile', fontweight='bold', fontsize=12)
    ax.set_ylabel('JCT (hours)', fontweight='bold', fontsize=12)
    ax.set_title('Tail JCT Reduction (High Percentiles)', pad=20, fontweight='bold', fontsize=14)
    
    legend = ax.legend(frameon=True, fancybox=True, shadow=True, borderpad=1.2, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#E2E8F0')

    ax.grid(axis='y', zorder=0, alpha=0.3)
    ax.set_ylim(bottom=0)

    for i, (xv, bv) in enumerate(zip(xp_pcts, bs_pcts)):
        ax.text(i - width/2, xv * 0.5, f'{xv:.1f}h', ha='center', fontsize=9, 
                color='white', fontweight='bold', zorder=6)
        ax.text(i + width/2, bv * 0.5, f'{bv:.1f}h', ha='center', fontsize=9, 
                color='white', fontweight='bold', zorder=6)

    return fig


def make_fig5_sensitivity(sensitivity_dir: str, base_xp: pd.DataFrame, base_bs: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('white')

    def load_pair(name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x_path = os.path.join(sensitivity_dir, f'{name}_xpod.csv')
        b_path = os.path.join(sensitivity_dir, f'{name}_baseline.csv')
        x = pd.read_csv(x_path)
        b = pd.read_csv(b_path)
        return x, b

    base_xp_100k = base_xp.head(100000)
    base_bs_100k = base_bs.head(100000)

    axA, axB, axC = axes
    for ax in axes:
        style_axes(ax)

    caps = [30, 50, 100, 200]
    jct_imps_A = []
    for cap in caps:
        if cap == 50:
            jct_imps_A.append(jct_improvement_pct(base_xp_100k, base_bs_100k))
        else:
            x, b = load_pair(f'A_gpucap{cap}')
            jct_imps_A.append(jct_improvement_pct(x, b))

    axA.plot(caps, jct_imps_A, marker='o', color=XPOD_COLOR, lw=2.5, markersize=12, 
             markerfacecolor=XPOD_COLOR, markeredgecolor='white', markeredgewidth=2, zorder=5)
    axA.set_xlabel('GPU vGPU Capacity', fontweight='medium')
    axA.set_ylabel('JCT Improvement (%)', fontweight='medium')
    axA.set_title('(A) Sensitivity to GPU Capacity', pad=15, fontweight='bold')
    axA.set_ylim(0, max(jct_imps_A)*1.5 + 1)
    axA.grid(True, alpha=0.3, linestyle='--')

    vgpus = [2, 4, 8]
    jct_imps_B = []
    for v in vgpus:
        if v == 4:
            jct_imps_B.append(jct_improvement_pct(base_xp_100k, base_bs_100k))
        else:
            x, b = load_pair(f'B_vgpu{v}')
            jct_imps_B.append(jct_improvement_pct(x, b))

    axB.plot(vgpus, jct_imps_B, marker='o', color=XPOD_COLOR, lw=2.5, markersize=12, 
             markerfacecolor=XPOD_COLOR, markeredgecolor='white', markeredgewidth=2, zorder=5)
    axB.set_xlabel('Number of vGPU Slots', fontweight='medium')
    axB.set_ylabel('JCT Improvement (%)', fontweight='medium')
    axB.set_title('(B) Sensitivity to vGPU Count', pad=15, fontweight='bold')
    axB.set_xticks(vgpus)
    axB.set_ylim(0, max(jct_imps_B)*1.3 + 1)
    axB.grid(True, alpha=0.3, linestyle='--')

    data_n = [2, 3, 4]
    transfer_xp = []
    transfer_bs = []
    for n in data_n:
        if n == 2:
            x, b = base_xp_100k, base_bs_100k
        else:
            x, b = load_pair(f'C_data{n}')
        transfer_xp.append(x['bytes_to_load'].sum() / 1024**3)
        transfer_bs.append(b['bytes_to_load'].sum() / 1024**3)

    axC.plot(data_n, transfer_xp, marker='o', color=XPOD_COLOR, lw=2.5, markersize=12, 
             markerfacecolor=XPOD_COLOR, markeredgecolor='white', markeredgewidth=2, label='xPod', zorder=5)
    axC.plot(data_n, transfer_bs, marker='s', color=BASELINE_COLOR, lw=2.5, markersize=12,
             markerfacecolor=BASELINE_COLOR, markeredgecolor='white', markeredgewidth=2,
             linestyle='--', label='Random Baseline', zorder=5)
    axC.set_xlabel('Number of Data Nodes', fontweight='medium')
    axC.set_ylabel('Total Data Transfer (GB)', fontweight='medium')
    axC.set_title('(C) Storage Cost vs Data Node Count', pad=15, fontweight='bold')
    axC.set_xticks(data_n)
    
    legend = axC.legend(frameon=True, fancybox=True, shadow=True, borderpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#E2E8F0')
    
    axC.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout(pad=2.0)
    return fig


def make_fig6_load_heatmap(xp_df: pd.DataFrame, bs_df: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')

    def make_load_matrix(df: pd.DataFrame, nodes: List[str], n_bins: int = 100) -> np.ndarray:
        df = df.sort_values('idx').reset_index(drop=True)
        bin_size = len(df) // n_bins
        mat = np.zeros((len(nodes), n_bins))
        for b_idx in range(n_bins):
            seg = df.iloc[b_idx*bin_size:(b_idx+1)*bin_size]
            counts = seg['compute_node'].value_counts()
            for i, node in enumerate(nodes):
                mat[i, b_idx] = counts.get(node, 0)
        return mat

    gpu_nodes = ['gpu-vnode-1', 'gpu-vnode-2', 'gpu-vnode-3', 'gpu-vnode-4']
    xp_mat = make_load_matrix(xp_df, gpu_nodes)
    bs_mat = make_load_matrix(bs_df, gpu_nodes)
    vmax = max(xp_mat.max(), bs_mat.max())

    ax1, ax2 = axes

    cmap = plt.cm.get_cmap('YlOrRd')
    cmap.set_bad(color='white')

    im1 = ax1.imshow(xp_mat, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    ax1.set_yticks(range(len(gpu_nodes)))
    ax1.set_yticklabels(gpu_nodes, fontweight='medium')
    ax1.set_xlabel('Time Bin (% of trace)', fontweight='medium')
    ax1.set_title('xPod: vGPU Task Distribution Over Time', pad=15, fontweight='bold')
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.outline.set_visible(False)
    style_axes(ax1, remove_top_right=False)

    im2 = ax2.imshow(bs_mat, aspect='auto', cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    ax2.set_yticks(range(len(gpu_nodes)))
    ax2.set_yticklabels(gpu_nodes, fontweight='medium')
    ax2.set_xlabel('Time Bin (% of trace)', fontweight='medium')
    ax2.set_title('Random Baseline: vGPU Task Distribution Over Time', pad=15, fontweight='bold')
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.outline.set_visible(False)
    style_axes(ax2, remove_top_right=False)

    fig.tight_layout(pad=2.0)
    return fig


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate xPod paper figures')
    parser.add_argument('--xpod-csv', default='results/raw/v1_final/xpod_4vgpu_cap50.csv',
                        help='Path to xPod experiment CSV')
    parser.add_argument('--baseline-csv', default='results/raw/v1_final/baseline_4vgpu_cap50.csv',
                        help='Path to baseline experiment CSV')
    parser.add_argument('--sensitivity-dir', default='results/raw/sensitivity',
                        help='Directory containing sensitivity experiment CSVs')
    parser.add_argument('--trace-csv', default='datasets/alibaba-cluster-trace-gpu-v2020/processed/scene4-joint/xpod_requests.csv',
                        help='Path to trace CSV for duration data')
    parser.add_argument('--output-dir', default='results/figures',
                        help='Directory to save output figures')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--figures', default='all',
                        help='Comma-separated list of figures to generate (1-6) or "all"')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(message)s'
    )

    logger.info('Starting figure generation')
    logger.info('xPod CSV: %s', args.xpod_csv)
    logger.info('Baseline CSV: %s', args.baseline_csv)
    logger.info('Sensitivity directory: %s', args.sensitivity_dir)
    logger.info('Trace CSV: %s', args.trace_csv)
    logger.info('Output directory: %s', args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info('Loading main experiment data...')
    xp_df = load_with_duration(args.xpod_csv, args.trace_csv)
    bs_df = load_with_duration(args.baseline_csv, args.trace_csv)
    logger.info('Data loaded: xPod %d rows, Baseline %d rows', len(xp_df), len(bs_df))

    figures_to_gen = []
    if args.figures == 'all':
        figures_to_gen = [1, 2, 3, 4, 5, 6]
    else:
        figures_to_gen = [int(f.strip()) for f in args.figures.split(',')]

    logger.info('Generating figures: %s', figures_to_gen)

    if 1 in figures_to_gen:
        logger.info('Generating Fig 1: Cold Start Convergence')
        fig1 = make_fig1_cold_start_convergence(xp_df, bs_df)
        save_figure(fig1, 'fig1_cold_start_convergence', args.output_dir)

    if 2 in figures_to_gen:
        logger.info('Generating Fig 2: Cumulative Data Transfer')
        fig2 = make_fig2_cumulative_transfer(xp_df, bs_df)
        save_figure(fig2, 'fig2_cumulative_transfer', args.output_dir)

    if 3 in figures_to_gen:
        logger.info('Generating Fig 3: GPU Contention CDF')
        fig3 = make_fig3_contention_cdf(xp_df, bs_df)
        save_figure(fig3, 'fig3_contention_cdf', args.output_dir)

    if 4 in figures_to_gen:
        logger.info('Generating Fig 4: JCT Percentiles')
        fig4 = make_fig4_jct_percentile(xp_df, bs_df)
        save_figure(fig4, 'fig4_jct_percentile', args.output_dir)

    if 5 in figures_to_gen:
        logger.info('Generating Fig 5: Sensitivity Analysis')
        fig5 = make_fig5_sensitivity(args.sensitivity_dir, xp_df, bs_df)
        save_figure(fig5, 'fig5_sensitivity', args.output_dir)

    if 6 in figures_to_gen:
        logger.info('Generating Fig 6: Load Heatmap')
        fig6 = make_fig6_load_heatmap(xp_df, bs_df)
        save_figure(fig6, 'fig6_load_heatmap', args.output_dir)

    logger.info('All figures generated successfully!')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
