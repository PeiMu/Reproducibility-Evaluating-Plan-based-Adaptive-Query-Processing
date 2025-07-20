import csv
import re

import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import numpy as np
import seaborn as sns
import math
from math import e
import sys
import plot_duckdb_end2end_results
from matplotlib.collections import PolyCollection
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec

if not os.path.exists("../figures"):
    os.makedirs("../figures")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

benchmarks = ["JOB", "DSB_10", "DSB_100"]

bar_colors = [
    '#006400',  # vanilla duckdb
    '#00FF00',  # AQP duckdb
    '#4D9296',  # vanilla common duckdb
    '#B4F4F7',  # AQP common duckdb
    '#165DC7',  # vanilla postgres
    '#03BAFC',  # AQP postgres
    '#03fcd7',  # AQP duckdb without join reordering
]

compare_colors = ['#D81B60', '#FFC107', '#004D40']


def plot_end2end_combined(col1_data, col1_color,
                          col2_data, col2_color,
                          col3_data, col3_color,
                          col4_data, col4_color,
                          col5_data, col5_color,
                          col6_data, col6_color,
                          legend_names, legend_colors,
                          legend_patterns,
                          benchmark_names, title,
                          output_name):
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    # Organize data by benchmark
    all_data = [col1_data, col2_data, col3_data,
                col4_data, col5_data, col6_data]

    # Plot each benchmark
    for benchmark_idx, (ax, benchmark_name) in enumerate(zip(axes, benchmark_names)):
        # Extract data for this benchmark (index corresponds to JOB=0, DSB_10=1, DSB_100=2)
        values = [
            col1_data[benchmark_idx],  # col1_data
            col2_data[benchmark_idx],  # col2_data
            col3_data[benchmark_idx],  # col3_data
            col4_data[benchmark_idx],  # col4_data
            col5_data[benchmark_idx],  # col5_data
            col6_data[benchmark_idx]  # col6_data
        ]

        # Find global maximum for consistent y-axis scaling
        global_max = max(values)

        colors = [col1_color, col2_color, col3_color,
                  col4_color, col5_color, col6_color]
        patterns = [legend_patterns[0], legend_patterns[1], legend_patterns[0],
                    legend_patterns[1], legend_patterns[0], legend_patterns[1]]

        x_positions = [0, 0.5, 2, 2.5, 4, 4.5]

        # Create bars
        bars = []
        for i in range(len(values)):
            bar = ax.bar(x_positions[i], values[i], width=0.4, color=colors[i], edgecolor='black')
            if patterns[i]:
                bar[0].set_hatch(patterns[i])
            bars.append(bar)

        # Add speedup/slowdown annotations
        for i in range(3):  # Three system groups
            x1, x2 = x_positions[i * 2], x_positions[i * 2 + 1]
            y1, y2 = values[i * 2], values[i * 2 + 1]  # vanilla, aqp

            # Determine which is higher
            swap_flag = False
            if y1 > y2:
                y_higher, y_lower = y1, y2
                swap_flag = True
            else:
                y_higher, y_lower = y2, y1

            # Draw vertical dotted line
            ax.plot([(x1 + x2) / 2, (x1 + x2) / 2], [y_lower, y_higher], color='red', linestyle=':', linewidth=1)

            # Draw horizontal solid lines
            ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y_higher, y_higher], color='red', linestyle='-',
                    linewidth=1)
            ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y_lower, y_lower], color='red', linestyle='-',
                    linewidth=1)

            # Add speedup/slowdown text
            y_speedup = y_higher + global_max / 10

            # if swap_flag:  # vanilla > aqp (speedup)
            #     speedup = y1 / y2
            #     ax.text((x1 + x2) / 2, y_speedup, f'{speedup:.2f}x ↑',
            #             ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
            # else:  # aqp > vanilla (slowdown)
            #     slowdown = y2 / y1
            #     ax.text((x1 + x2) / 2, y_speedup, f'{slowdown:.2f}x ↓',
            #             ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
            speedup = y1 / y2
            ax.text((x1 + x2) / 2, y_speedup, f'{speedup:.2f}x',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')

        # Set x-axis labels and formatting
        ax.set_xticks([0.25, 2.25, 4.25])
        ax.set_xticklabels(['DuckDB', 'DuckDB\n(common)', 'PostgreSQL\n(common)'])
        ax.set_title(benchmark_name, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', labelsize=12)
        # Make major tick labels bold
        # for tick in ax.xaxis.get_major_ticks():
        #     tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, value in enumerate(values):
            ax.text(x_positions[i] - 0.01, value + global_max * 0.01, f'{value:.1f}',
                    ha='center', fontsize=10, fontweight='bold')

        # Set consistent y-axis limits
        ax.set_ylim(0, global_max + global_max / 5)
        ax.tick_params(axis='y', labelsize=10)

    # Set shared y-axis label
    fig.text(-0.005, 0.5, title, va='center', rotation='vertical',
             fontsize=14, fontweight='bold')

    # Create shared legend at the top
    legend_patches = []
    for i, (name, color, pattern) in enumerate(zip(legend_names, legend_colors, legend_patterns)):
        legend_patches.append(Patch(facecolor=color, edgecolor='black', hatch=pattern, label=name))

    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.95),
               ncol=len(legend_names), fontsize=12, handleheight=2, handlelength=3)

    # # Set consistent y-axis limits
    # for ax in axes:
    #     ax.set_ylim(0, global_max + global_max / 20 + 15)
    #     ax.tick_params(axis='y', labelsize=10)

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.7)  # Make room for the legend
    plt.savefig(f'../figures/{output_name}.pdf', bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def plot_end2end_wo_reorder(col1_data, col1_color,
                            col2_data, col2_color,
                            group_names,
                            legend_names, legend_colors, title,
                            name):
    assert len(col1_data) == len(col2_data) == len(group_names), "Data and group_names must match in length"
    n = len(group_names)

    fig, ax = plt.subplots(figsize=(6, 3))

    bar_width = 0.35
    indices = list(range(n))

    bars = []
    for i in range(n):
        if i == 2:
            col2_data[i] = 12000
        # Plot col1 and col2 bars side-by-side
        bar1 = ax.bar(i - bar_width / 2, col1_data[i], width=bar_width, color=col1_color, edgecolor='black')
        bar2 = ax.bar(i + bar_width / 2, col2_data[i], width=bar_width, color=col2_color, edgecolor='black')
        bars.append((bar1, bar2))

        # Draw vertical and horizontal comparison lines between col1_data[i] and col2_data[i]
        y1 = col1_data[i]
        y2 = col2_data[i]
        x1 = i - bar_width / 2
        x2 = i + bar_width / 2

        if i != n - 1:
            # Ensure y1 is the smaller
            swap_flag = False
            if y1 > y2:
                y_higher, y_lower = y1, y2
            else:
                y_higher, y_lower = y2, y1

            # Add speedup/slowdown text
            y_speedup = y_higher + max(col1_data + col2_data) / 10
            speedup = y1 / y2
            ax.text((x1 + x2) / 2, y_speedup, f'{speedup:.2f}x',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')

    # X-axis settings
    ax.set_xticks(indices)
    ax.set_xticklabels(group_names, fontsize=10, fontweight='bold')

    # Add legend
    legend_patches = [Patch(facecolor=legend_colors[0], edgecolor='black', label=legend_names[0]),
                      Patch(facecolor=legend_colors[1], edgecolor='black', label=legend_names[1])]
    ax.legend(handles=legend_patches, loc='upper left', handleheight=3, handlelength=4)

    # Y-label and title
    plt.ylabel(title, fontsize=14, fontweight='bold')

    # Add values above bars
    for i in range(n):
        plt.text(i - bar_width / 2, col1_data[i] + max(col1_data + col2_data) * 0.01,
                 f'{col1_data[i]:.2f}', ha='center', fontsize=9)
        if i == n - 1:
            plt.text(i + bar_width / 2, col2_data[i] + max(col1_data + col2_data) * 0.01,
                     f'TO', ha='center', fontsize=9, fontweight='bold', color='black')
            plt.hlines(col2_data[i], 0, i + bar_width / 2, linestyle='dashed', color='black')
        else:
            plt.text(i + bar_width / 2, col2_data[i] + max(col1_data + col2_data) * 0.01,
                     f'{col2_data[i]:.2f}', ha='center', fontsize=9)

    plt.ylim(0, 14000)
    plt.tight_layout()
    plt.savefig('../figures/' + name + '.pdf')
    plt.clf()
    plt.close()


def plot_violin_dataframe(col1_data,
                          col2_data,
                          col3_data,
                          col4_data,
                          col5_data,
                          col6_data,
                          colors, legend_patterns,
                          benchmark_names, output_name):
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

    # Process each benchmark
    for benchmark_idx, (ax, benchmark_name) in enumerate(zip(axes, benchmark_names)):
        sql_name = col1_data[benchmark_idx]["sql_name"]
        # Extract data for this benchmark
        col1_data_df = col1_data[benchmark_idx]["mean"]
        col2_data_df = col2_data[benchmark_idx]["mean"]
        col3_data_df = col3_data[benchmark_idx]["mean"]
        col4_data_df = col4_data[benchmark_idx]["mean"]
        col5_data_df = col5_data[benchmark_idx]["mean"]
        col6_data_df = col6_data[benchmark_idx]["mean"]

        # Create DataFrame for seaborn
        plot_data = []

        # Add DuckDB data
        for i, (v_val, a_val) in enumerate(zip(col1_data_df, col2_data_df)):
            plot_data.append({'System': 'DuckDB', 'Type': 'Vanilla', 'Value': v_val, 'Query': sql_name[i]})
            plot_data.append({'System': 'DuckDB', 'Type': 'AQP', 'Value': a_val, 'Query': sql_name[i]})

        # Add DuckDB Common data
        for i, (v_val, a_val) in enumerate(zip(col3_data_df, col4_data_df)):
            plot_data.append({'System': 'DuckDB\n(common)', 'Type': 'Vanilla', 'Value': v_val, 'Query': sql_name[i]})
            plot_data.append({'System': 'DuckDB\n(common)', 'Type': 'AQP', 'Value': a_val, 'Query': sql_name[i]})

        # Add PostgreSQL data
        for i, (v_val, a_val) in enumerate(zip(col5_data_df, col6_data_df)):
            plot_data.append(
                {'System': 'PostgreSQL\n(common)', 'Type': 'Vanilla', 'Value': v_val, 'Query': sql_name[i]})
            plot_data.append({'System': 'PostgreSQL\n(common)', 'Type': 'AQP', 'Value': a_val, 'Query': sql_name[i]})

        df = pd.DataFrame(plot_data)

        # Create split violin plot
        sns.violinplot(data=df, x='System', y='Value', hue='Type',
                       split=True, gap=.1,
                       density_norm="width", inner="box", inner_kws=dict(box_width=12, whis_width=10),
                       palette='Set1', cut=0,
                       ax=ax, linewidth=1.5)
        # sns.stripplot(data=df, x="System", y="Value", ax=ax,
        #               color='white', size=3, jitter=True, alpha=0.6, edgecolor="black", linewidth=1)
        # Build a mapping from system name to x-position
        x_categories = df['System'].unique()
        x_pos_map = {name: i for i, name in enumerate(x_categories)}
        # Define manual offsets
        offset = 0.15  # Amount to nudge left/right
        # Plot strip points manually
        for i, row in df.iterrows():
            base_x = x_pos_map[row['System']]
            if row['Type'] == 'Vanilla':
                x = base_x - offset
            else:  # AQP
                x = base_x + offset
            ax.scatter(x, row['Value'], color='white', s=20, edgecolor='black', linewidth=1, alpha=0.6, zorder=5)

        for ind, violin in enumerate(ax.findobj(PolyCollection)):
            violin.set_facecolor(colors[ind])

        # Apply patterns to violins
        pattern1 = legend_patterns[0]  # For DuckDB and DuckDB(common)
        pattern2 = legend_patterns[1]  # For PostgreSQL
        hatches = [pattern1, pattern2, pattern1, pattern2, pattern1, pattern2]

        # Apply hatches to violin bodies (first 6 collections)
        for i in range(6):
            ax.collections[i].set_hatch(hatches[i])
            ax.collections[i].set_edgecolor('black')  # Enhance pattern visibility

        # Customize subplot
        # ax.set_title(benchmark_name, fontsize=14, fontweight='bold')
        # ax.tick_params(axis='x', labelsize=12)
        ax.set_xticklabels([])
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, alpha=0.3)

        # Remove individual subplot legends
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_ylabel("")
        ax.set_xlabel("")

        # Build a mapping from system name to x-position
        x_categories = df['System'].unique()
        x_pos_map = {name: i for i, name in enumerate(x_categories)}
        offset = 0.15  # Must match strip plot offset

        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_offset = y_range * 0.08  # Text vertical offset

        for system in x_categories:
            sys_df = df[df['System'] == system]
            if sys_df.empty:
                continue

            # Calculate average and outlier thresholds
            avg_val = sys_df['Value'].mean()
            high_threshold = 15 * avg_val

            # Identify outliers
            outliers = sys_df[
                (sys_df['Value'] > high_threshold)
            ]

            # Find the most extreme outlier in each direction
            max_deviation = 0
            min_deviation = float('inf')
            extreme_query = None

            for _, row in outliers.iterrows():
                # Calculate deviation ratio
                if row['Value'] > high_threshold:
                    deviation = row['Value'] / avg_val
                    if deviation > max_deviation:
                        max_deviation = deviation
                        extreme_max_row = row
                        extreme_query = row['Query']

            # If we found an extreme query
            if extreme_query:
                # Get both points for this query (Vanilla and AQP)
                query_points = sys_df[sys_df['Query'] == extreme_query]

                # Annotate both points
                for _, row in query_points.iterrows():
                    base_x = x_pos_map[system]
                    xpos = base_x + (offset if row['Type'] == 'AQP' else -offset)
                    value = row['Value']

                    is_left = (row['Type'] == 'Vanilla')
                    # Determine annotation direction based on value
                    color = 'darkred'
                    # Calculate annotation position
                    if is_left:
                        text_y = value + y_offset
                        va = 'bottom'
                        arrow_rad = 0.2
                    else:
                        text_y = value - y_offset
                        va = 'top'
                        arrow_rad = -0.2

                    # Position text to avoid overlap
                    if is_left:
                        text_x = xpos - 0.3
                        ha = 'right'
                        arrow_rad = -abs(arrow_rad)  # Curve left
                    else:
                        text_x = xpos + 0.3
                        ha = 'left'
                        arrow_rad = abs(arrow_rad)  # Curve right

                    ax.annotate(f"{row['Query']}: {row['Value']:.2f}s",
                                xy=(xpos, row['Value']),
                                xytext=(text_x, text_y),
                                fontsize=8,
                                fontweight='bold',
                                color=color,
                                arrowprops=dict(
                                    arrowstyle="->",
                                    color=color,
                                    lw=1,
                                    connectionstyle=f"arc3,rad={arrow_rad}"
                                ),
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    fc="white",
                                    ec=color,
                                    lw=1,
                                    alpha=0.9
                                ),
                                horizontalalignment='center',
                                verticalalignment=va)

    # Set shared y-axis label
    fig.text(-0.005, 0.5, 'E2E Time for Each Query [s]', va='center', rotation='vertical',
             fontsize=14, fontweight='bold')

    # Adjust layout and save
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    plt.savefig(f'../figures/{output_name}.pdf', bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def plot_boxplot_dataframe(col1_data,
                           col2_data,
                           col3_data,
                           col4_data,
                           col5_data,
                           col6_data,
                           colors, legend_patterns,
                           broken_axis_range,
                           benchmark_names, output_name):
    # Create figure with subplots
    # fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

    fig = plt.figure(figsize=(15, 3))
    gs = GridSpec(1, 3, figure=fig)

    # # Create axes for first two benchmarks
    # ax1 = fig.add_subplot(gs[0])
    # ax2 = fig.add_subplot(gs[1])
    # axes = [ax1, ax2]

    axes = []
    for idx, [lower_range, upper_range] in enumerate(broken_axis_range):
        bax = brokenaxes(
            ylims=(lower_range, upper_range),
            subplot_spec=gs[idx],
            despine=False,
            wspace=0.05,
            d=0.015  # Break mark size
        )
        axes.append(bax)

    box_positions = {
        'DuckDB': {'Vanilla': 0, 'AQP': 0.5},
        'DuckDB\n(common)': {'Vanilla': 2, 'AQP': 2.5},
        'PostgreSQL\n(common)': {'Vanilla': 4, 'AQP': 4.5}
    }
    box_width = 0.3

    # Process each benchmark
    for benchmark_idx, (ax, benchmark_name) in enumerate(zip(axes, benchmark_names)):
        sql_name = col1_data[benchmark_idx]["sql_name"]
        # Extract data for this benchmark
        col1_data_df = col1_data[benchmark_idx]["mean"]
        col2_data_df = col2_data[benchmark_idx]["mean"]
        col3_data_df = col3_data[benchmark_idx]["mean"]
        col4_data_df = col4_data[benchmark_idx]["mean"]
        col5_data_df = col5_data[benchmark_idx]["mean"]
        col6_data_df = col6_data[benchmark_idx]["mean"]

        # Create DataFrame for seaborn
        plot_data = []

        # Add DuckDB data
        for i, (v_val, a_val) in enumerate(zip(col1_data_df, col2_data_df)):
            plot_data.append({'System': 'DuckDB', 'Type': 'Vanilla', 'Value': v_val, 'Query': sql_name[i]})
            plot_data.append({'System': 'DuckDB', 'Type': 'AQP', 'Value': a_val, 'Query': sql_name[i]})

        # Add DuckDB Common data
        for i, (v_val, a_val) in enumerate(zip(col3_data_df, col4_data_df)):
            plot_data.append({'System': 'DuckDB\n(common)', 'Type': 'Vanilla', 'Value': v_val, 'Query': sql_name[i]})
            plot_data.append({'System': 'DuckDB\n(common)', 'Type': 'AQP', 'Value': a_val, 'Query': sql_name[i]})

        # Add PostgreSQL data
        for i, (v_val, a_val) in enumerate(zip(col5_data_df, col6_data_df)):
            plot_data.append(
                {'System': 'PostgreSQL\n(common)', 'Type': 'Vanilla', 'Value': v_val, 'Query': sql_name[i]})
            plot_data.append({'System': 'PostgreSQL\n(common)', 'Type': 'AQP', 'Value': a_val, 'Query': sql_name[i]})

        df = pd.DataFrame(plot_data)

        # Create box plot with manual positioning
        systems = df['System'].unique()
        for system in box_positions:
            sys_df = df[df['System'] == system]
            if sys_df.empty:
                continue

            sys_df = df[df['System'] == system]

            # Vanilla box
            vanilla_data = sys_df[sys_df['Type'] == 'Vanilla']['Value']

            # Plot on each sub-axis of brokenaxes
            box = {}
            for ax_part in ax.axs:
                box = ax_part.boxplot(
                    vanilla_data,
                    positions=[box_positions[system]['Vanilla']],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=True
                )

            # box = current_ax.boxplot(
            #     vanilla_data,
            #     positions=[box_positions[system]['Vanilla']],
            #     widths=box_width,
            #     patch_artist=True,
            #     showfliers=True
            # )

            # Set color and style for Vanilla
            color_idx = list(box_positions.keys()).index(system) * 2
            for patch in box['boxes']:
                patch.set_facecolor(colors[color_idx])
                patch.set_edgecolor('black')
                patch.set_hatch(legend_patterns[0])

            # AQP box
            aqp_data = sys_df[sys_df['Type'] == 'AQP']['Value']
            for ax_part in ax.axs:
                box = ax_part.boxplot(
                    aqp_data,
                    positions=[box_positions[system]['AQP']],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=True
                )
            # box = current_ax.boxplot(
            #     aqp_data,
            #     positions=[box_positions[system]['AQP']],
            #     widths=box_width,
            #     patch_artist=True,
            #     showfliers=True
            # )

            # Set color and style for AQP
            color_idx = list(box_positions.keys()).index(system) * 2 + 1
            for patch in box['boxes']:
                patch.set_facecolor(colors[color_idx])
                patch.set_edgecolor('black')
                patch.set_hatch(legend_patterns[1])

        # for tick in ax.yaxis.get_major_ticks():
        #     tick.label1.set_fontweight('bold')
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("")
        ax.set_xlabel("")

        # Brokenaxes handling
        ax.set_xticks([0.25, 2.25, 4.25])
        ax.set_xticklabels([])

        # Apply settings to all internal axes
        for ax_part in ax.axs:
            ax_part.grid(True, alpha=0.3)
            ax_part.tick_params(axis='y', labelsize=10)
            for tick in ax_part.yaxis.get_major_ticks():
                tick.label1.set_fontweight('bold')
            # Hide x-axis on top part
            if ax_part != ax.axs[0]:
                ax_part.set_xticks([])

        y_offset = 1

        for system in systems:
            sys_df = df[df['System'] == system]
            if sys_df.empty:
                continue

            # Calculate average and outlier thresholds
            avg_val = sys_df['Value'].mean()
            high_threshold = 15 * avg_val

            # Identify outliers
            outliers = sys_df[
                (sys_df['Value'] > high_threshold)
            ]

            # Find the most extreme outlier in each direction
            max_deviation = 0
            min_deviation = float('inf')
            extreme_max_row = None
            extreme_min_row = None
            extreme_query = None

            for _, row in outliers.iterrows():
                # Calculate deviation ratio
                if row['Value'] > high_threshold:
                    deviation = row['Value'] / avg_val
                    if deviation > max_deviation:
                        max_deviation = deviation
                        extreme_max_row = row
                        extreme_query = row['Query']
                else:
                    deviation = avg_val / row['Value']
                    if deviation > min_deviation:
                        min_deviation = deviation
                        extreme_min_row = row
                        extreme_query = row['Query']

            # If we found an extreme query
            if extreme_query:
                # Get both points for this query (Vanilla and AQP)
                query_points = sys_df[sys_df['Query'] == extreme_query]

                # Annotate both points
                for _, row in query_points.iterrows():
                    xpos = box_positions[system][row['Type']]
                    value = row['Value']

                    # Get matching color
                    system_idx = list(box_positions.keys()).index(system)

                    # Determine position based on value and type
                    is_high = value > avg_val
                    is_left = (row['Type'] == 'Vanilla')

                    # Calculate annotation position
                    if is_left:
                        text_y = value + y_offset
                        va = 'bottom'
                        arrow_rad = 0.2
                    else:
                        text_y = value - y_offset
                        va = 'top'
                        arrow_rad = -0.2

                    # Position text to avoid overlap
                    if is_left:
                        text_x = xpos - 0.3
                        ha = 'right'
                        arrow_rad = -abs(arrow_rad)  # Curve left
                    else:
                        text_x = xpos + 0.3
                        ha = 'left'
                        arrow_rad = abs(arrow_rad)  # Curve right

                    # Create annotation
                    ax.annotate(f"{row['Query']}: {row['Value']:.2f}s",
                                xy=(xpos, value),
                                xytext=(text_x, text_y),
                                fontsize=8,
                                fontweight='bold',
                                color='darkred',
                                arrowprops=dict(
                                    arrowstyle="->",
                                    color='darkred',
                                    lw=1,
                                    connectionstyle=f"arc3,rad={arrow_rad}"
                                ),
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    fc="white",
                                    ec='darkred',
                                    lw=1,
                                    alpha=0.9
                                ),
                                horizontalalignment=ha,
                                verticalalignment=va)

    # Set shared y-axis label
    fig.text(0.093, 0.5, 'E2E Time for Each Query [s]', va='center', rotation='vertical',
             fontsize=12, fontweight='bold')

    # Adjust layout and save
    # plt.tight_layout()
    plt.savefig(f'../figures/{output_name}.pdf', bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def plot_boxplot_dataframe_wo_reorder(col1_data, col1_color,
                                      col2_data, col2_color,
                                      broken_axis_range,
                                      group_names,
                                      legend_names, legend_colors, title,
                                      name):
    # assert len(col1_data) == len(col2_data) == len(group_names), "Data and group_names must match in length"
    lower_range, upper_range = broken_axis_range

    fig = plt.figure(figsize=(6, 3))
    bax = brokenaxes(
        ylims=(lower_range, upper_range),
        despine=False,
        wspace=0.05,
        d=0.015  # Break mark size
    )

    box_positions = {
        'JOB': {'Vanilla': 0, 'AQP': 0.5},
        'DSB_10': {'Vanilla': 2, 'AQP': 2.5},
        'DSB_100': {'Vanilla': 4, 'AQP': 4.5}
    }

    box_width = 0.3
    positions = [1, 1.5, 3, 4.5, 5, 5.5]
    box_data = []

    for i, group in enumerate(group_names):
        box = {}
        for ax_part in bax.axs:
            box = ax_part.boxplot(
                col1_data[i]['mean'],
                positions=[box_positions[group]['Vanilla']],
                widths=box_width,
                patch_artist=True,
                showfliers=True
            )

        # Set color and style for Vanilla
        for patch in box['boxes']:
            patch.set_facecolor(col1_color)
            patch.set_edgecolor('black')

        for ax_part in bax.axs:
            box = ax_part.boxplot(
                col2_data[i]['mean'],
                positions=[box_positions[group]['AQP']],
                widths=box_width,
                patch_artist=True,
                showfliers=True
            )

        # Set color and style for Vanilla
        for patch in box['boxes']:
            patch.set_facecolor(col2_color)
            patch.set_edgecolor('black')

        # Compute side-by-side positions
        # pos1 = i + box_width / 2
        # pos2 = i + box_width / 2
        # positions.extend([pos1, pos2])

        # Add data for Vanilla and AQP
        box_data.append(col1_data[i]['mean'].values)
        box_data.append(col2_data[i]['mean'].values)

    bax.tick_params(axis='y', labelsize=10)
    bax.grid(True, alpha=0.3)
    bax.set_ylabel("")
    bax.set_xlabel("")

    # # Brokenaxes handling
    # bax.set_xticks([0.25, 2.25, 4.25])
    # bax.set_xticklabels([])

    # Apply settings to all internal axes
    for ax_part in bax.axs:
        ax_part.grid(True, alpha=0.3)
        ax_part.tick_params(axis='y', labelsize=10)
        for tick in ax_part.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        # Hide x-axis on top part
        if ax_part != bax.axs[0]:
            ax_part.set_xticks([])

    y_offset = 1

    # Legend
    legend_patches = [
        Patch(facecolor=legend_colors[0], edgecolor='black', label=legend_names[0]),
        Patch(facecolor=legend_colors[1], edgecolor='black', label=legend_names[1])
    ]
    bax.legend(handles=legend_patches, loc='upper left', handleheight=3, handlelength=4)

    # Y-label
    fig.text(0.04, 0.5, title, va='center', rotation='vertical',
             fontsize=14, fontweight='bold')

    plt.savefig(f'../figures/{name}.pdf', bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def plot_compare_violin_dataframe(col1_data,
                                  col2_data,
                                  col3_data,
                                  col4_data,
                                  col5_data,
                                  col6_data,
                                  colors,
                                  benchmark_names, output_name):
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

    # Process each benchmark
    for benchmark_idx, (ax, benchmark_name) in enumerate(zip(axes, benchmark_names)):
        # Extract data for this benchmark
        col1_data_df = col1_data[benchmark_idx]["mean"]
        col2_data_df = col2_data[benchmark_idx]["mean"]
        col3_data_df = col3_data[benchmark_idx]["mean"]
        col4_data_df = col4_data[benchmark_idx]["mean"]
        col5_data_df = col5_data[benchmark_idx]["mean"]
        col6_data_df = col6_data[benchmark_idx]["mean"]

        # Create DataFrame for seaborn
        plot_data = []

        # Add DuckDB speedup
        for i, (v_val, a_val) in enumerate(zip(col1_data_df, col2_data_df)):
            speedup = v_val / a_val
            plot_data.append({'System': 'DuckDB', 'Speedup': speedup, 'Query': i})

        # Add DuckDB Common speedup
        for i, (v_val, a_val) in enumerate(zip(col3_data_df, col4_data_df)):
            speedup = v_val / a_val
            plot_data.append({'System': 'DuckDB\n(common)', 'Speedup': speedup, 'Query': i})

        # Add PostgreSQL speedup
        for i, (v_val, a_val) in enumerate(zip(col5_data_df, col6_data_df)):
            speedup = v_val / a_val
            plot_data.append({'System': 'PostgreSQL\n(common)', 'Speedup': speedup, 'Query': i})

        df = pd.DataFrame(plot_data)

        # Create split violin plot
        sns.violinplot(data=df, x='System', y='Speedup', log_scale=True, density_norm="width",
                       inner="box", inner_kws=dict(box_width=12, whis_width=10),
                       ax=ax, linewidth=1.5)
        sns.stripplot(data=df, x="System", y="Speedup", ax=ax,
                      color='white', size=3, jitter=True, alpha=0.6, edgecolor="black", linewidth=1)

        for ind, violin in enumerate(ax.findobj(PolyCollection)):
            violin.set_facecolor(colors[ind])

        # # Apply patterns to violins
        # pattern1 = legend_patterns[0]  # For DuckDB and DuckDB(common)
        # pattern2 = legend_patterns[1]  # For PostgreSQL
        # hatches = [pattern1, pattern2, pattern1, pattern2, pattern1, pattern2]
        #
        # # Apply hatches to violin bodies (first 6 collections)
        # for i in range(6):
        #     ax.collections[i].set_hatch(hatches[i])
        #     ax.collections[i].set_edgecolor('black')  # Enhance pattern visibility

        # Customize subplot
        # ax.set_title(benchmark_name, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', labelsize=12)
        ax.set_xticklabels([])
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, alpha=0.3)

        # Remove individual subplot legends
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_ylabel("")
        ax.set_xlabel("")

    # Set shared y-axis label
    fig.text(-0.005, 0.5, 'Speedup for Each Query [x]', va='center', rotation='vertical',
             fontsize=14, fontweight='bold')

    # Adjust layout and save
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    plt.savefig(f'../figures/{output_name}.pdf', bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def plot_violin_dict(col1_data,
                     col2_data,
                     col3_data,
                     col4_data,
                     col5_data,
                     col6_data,
                     colors, legend_patterns,
                     benchmark_names, output_name):
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

    # Process each benchmark
    for benchmark_idx, (ax, benchmark_name) in enumerate(zip(axes, benchmark_names)):
        # Extract data for this benchmark
        col1_data_dict = col1_data[benchmark_idx]
        col2_data_dict = col2_data[benchmark_idx]
        col3_data_dict = col3_data[benchmark_idx]
        col4_data_dict = col4_data[benchmark_idx]
        col5_data_dict = col5_data[benchmark_idx]
        col6_data_dict = col6_data[benchmark_idx]

        # Create DataFrame for seaborn
        plot_data = []

        # Add DuckDB data
        for qid in col1_data_dict:
            if qid in col2_data_dict:
                plot_data.append({'System': 'DuckDB', 'Type': 'Vanilla', 'Value': col1_data_dict[qid], 'Query': qid})
                plot_data.append({'System': 'DuckDB', 'Type': 'AQP', 'Value': col2_data_dict[qid], 'Query': qid})

        # Add DuckDB Common data
        for qid in col3_data_dict:
            if qid in col4_data_dict:
                plot_data.append(
                    {'System': 'DuckDB\n(common)', 'Type': 'Vanilla', 'Value': col3_data_dict[qid], 'Query': qid})
                plot_data.append(
                    {'System': 'DuckDB\n(common)', 'Type': 'AQP', 'Value': col4_data_dict[qid], 'Query': qid})

        # Add PostgreSQL data
        for qid in col5_data_dict:
            if qid in col6_data_dict:
                plot_data.append(
                    {'System': 'PostgreSQL\n(common)', 'Type': 'Vanilla', 'Value': col5_data_dict[qid], 'Query': qid})
                plot_data.append(
                    {'System': 'PostgreSQL\n(common)', 'Type': 'AQP', 'Value': col6_data_dict[qid], 'Query': qid})

        df = pd.DataFrame(plot_data)

        # Create split violin plot
        sns.violinplot(data=df, x='System', y='Value', hue='Type',
                       split=True, gap=.1,
                       density_norm="width", inner="box", inner_kws=dict(box_width=12, whis_width=10),
                       palette='Set1', cut=0,
                       ax=ax, linewidth=1.5)
        # sns.stripplot(data=df, x="System", y="Value", ax=ax,
        #               color='white', size=3, jitter=True, alpha=0.6, edgecolor="black", linewidth=1)
        # Build a mapping from system name to x-position
        x_categories = df['System'].unique()
        x_pos_map = {name: i for i, name in enumerate(x_categories)}
        # Define manual offsets
        offset = 0.15  # Amount to nudge left/right
        # Plot strip points manually
        for i, row in df.iterrows():
            base_x = x_pos_map[row['System']]
            if row['Type'] == 'Vanilla':
                x = base_x - offset
            else:  # AQP
                x = base_x + offset
            ax.scatter(x, row['Value'], color='white', s=20, edgecolor='black', linewidth=1, alpha=0.6, zorder=5)

        for ind, violin in enumerate(ax.findobj(PolyCollection)):
            violin.set_facecolor(colors[ind])

        # Apply patterns to violins
        pattern1 = legend_patterns[0]  # For DuckDB and DuckDB(common)
        pattern2 = legend_patterns[1]  # For PostgreSQL
        hatches = [pattern1, pattern2, pattern1, pattern2, pattern1, pattern2]

        # Apply hatches to violin bodies (first 6 collections)
        for i in range(6):
            ax.collections[i].set_hatch(hatches[i])
            ax.collections[i].set_edgecolor('black')  # Enhance pattern visibility

        # Customize subplot
        # ax.set_title(benchmark_name, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', labelsize=12)
        ax.set_xticklabels([])
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, alpha=0.3)

        # Remove individual subplot legends
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_ylabel("")
        ax.set_xlabel("")

        # Build a mapping from system name to x-position
        x_categories = df['System'].unique()
        x_pos_map = {name: i for i, name in enumerate(x_categories)}
        offset = 0.15  # Must match strip plot offset

        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_offset = y_range * 0.08  # Text vertical offset

        for system in x_categories:
            sys_df = df[df['System'] == system]
            if sys_df.empty:
                continue

            # Calculate average and outlier thresholds
            avg_val = sys_df['Value'].mean()
            high_threshold = 15 * avg_val

            # Identify outliers
            outliers = sys_df[
                (sys_df['Value'] > high_threshold)
            ]

            # Find the most extreme outlier in each direction
            max_deviation = 0
            min_deviation = float('inf')
            extreme_query = None

            for _, row in outliers.iterrows():
                # Calculate deviation ratio
                if row['Value'] > high_threshold:
                    deviation = row['Value'] / avg_val
                    if deviation > max_deviation:
                        max_deviation = deviation
                        extreme_max_row = row
                        extreme_query = row['Query']

            # If we found an extreme query
            if extreme_query:
                # Get both points for this query (Vanilla and AQP)
                query_points = sys_df[sys_df['Query'] == extreme_query]

                # Annotate both points
                for _, row in query_points.iterrows():
                    base_x = x_pos_map[system]
                    xpos = base_x + (offset if row['Type'] == 'AQP' else -offset)
                    value = row['Value']

                    is_left = (row['Type'] == 'Vanilla')
                    # Determine annotation direction based on value
                    color = 'darkred'
                    # Calculate annotation position
                    if not is_left:
                        text_y = value + y_offset
                        va = 'bottom'
                        arrow_rad = 0.2
                    else:
                        text_y = value - y_offset
                        va = 'top'
                        arrow_rad = -0.2

                    # Position text to avoid overlap
                    if is_left:
                        text_x = xpos - 0.3
                        ha = 'right'
                        arrow_rad = -abs(arrow_rad)  # Curve left
                    else:
                        text_x = xpos + 0.3
                        ha = 'left'
                        arrow_rad = abs(arrow_rad)  # Curve right

                    ax.annotate(f"{row['Query']}: {row['Value']:.2f}s",
                                xy=(xpos, row['Value']),
                                xytext=(text_x, text_y),
                                fontsize=8,
                                fontweight='bold',
                                color=color,
                                arrowprops=dict(
                                    arrowstyle="->",
                                    color=color,
                                    lw=1,
                                    connectionstyle=f"arc3,rad={arrow_rad}"
                                ),
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    fc="white",
                                    ec=color,
                                    lw=1,
                                    alpha=0.9
                                ),
                                horizontalalignment='center',
                                verticalalignment=va)

    # Set shared y-axis label
    fig.text(-0.005, 0.5, 'Exe Time for Each Query [s]', va='center', rotation='vertical',
             fontsize=14, fontweight='bold')

    # Adjust layout and save
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85, left=0.08)
    plt.savefig(f'../figures/{output_name}.pdf', bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def plot_boxplot_dict(col1_data,
                      col2_data,
                      col3_data,
                      col4_data,
                      col5_data,
                      col6_data,
                      colors, legend_patterns,
                      broken_axis_range,
                      benchmark_names, output_name):
    # Create figure with subplots
    # fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

    fig = plt.figure(figsize=(15, 3))
    gs = GridSpec(1, 3, figure=fig)

    # # Create axes for first two benchmarks
    # ax1 = fig.add_subplot(gs[0])
    # ax2 = fig.add_subplot(gs[1])
    # axes = [ax1, ax2]

    axes = []
    for idx, [lower_range, upper_range] in enumerate(broken_axis_range):
        bax = brokenaxes(
            ylims=(lower_range, upper_range),
            subplot_spec=gs[idx],
            despine=False,
            wspace=0.05,
            d=0.015  # Break mark size
        )
        axes.append(bax)

    box_positions = {
        'DuckDB': {'Vanilla': 0, 'AQP': 0.5},
        'DuckDB\n(common)': {'Vanilla': 2, 'AQP': 2.5},
        'PostgreSQL\n(common)': {'Vanilla': 4, 'AQP': 4.5}
    }
    box_width = 0.3

    # Process each benchmark
    for benchmark_idx, (ax, benchmark_name) in enumerate(zip(axes, benchmark_names)):
        # Extract data for this benchmark
        col1_data_dict = col1_data[benchmark_idx]
        col2_data_dict = col2_data[benchmark_idx]
        col3_data_dict = col3_data[benchmark_idx]
        col4_data_dict = col4_data[benchmark_idx]
        col5_data_dict = col5_data[benchmark_idx]
        col6_data_dict = col6_data[benchmark_idx]

        # Create DataFrame for seaborn
        plot_data = []

        # Add DuckDB data
        for qid in col1_data_dict:
            if qid in col2_data_dict:
                plot_data.append({'System': 'DuckDB', 'Type': 'Vanilla', 'Value': col1_data_dict[qid], 'Query': qid})
                plot_data.append({'System': 'DuckDB', 'Type': 'AQP', 'Value': col2_data_dict[qid], 'Query': qid})

        # Add DuckDB Common data
        for qid in col3_data_dict:
            if qid in col4_data_dict:
                plot_data.append(
                    {'System': 'DuckDB\n(common)', 'Type': 'Vanilla', 'Value': col3_data_dict[qid],
                     'Query': qid})
                plot_data.append(
                    {'System': 'DuckDB\n(common)', 'Type': 'AQP', 'Value': col4_data_dict[qid], 'Query': qid})

        # Add PostgreSQL data
        for qid in col5_data_dict:
            if qid in col6_data_dict:
                plot_data.append(
                    {'System': 'PostgreSQL\n(common)', 'Type': 'Vanilla', 'Value': col5_data_dict[qid],
                     'Query': qid})
                plot_data.append(
                    {'System': 'PostgreSQL\n(common)', 'Type': 'AQP', 'Value': col6_data_dict[qid],
                     'Query': qid})

        df = pd.DataFrame(plot_data)

        # Create box plot with manual positioning
        systems = df['System'].unique()
        for system in box_positions:
            sys_df = df[df['System'] == system]
            if sys_df.empty:
                continue

            sys_df = df[df['System'] == system]

            # Vanilla box
            vanilla_data = sys_df[sys_df['Type'] == 'Vanilla']['Value']

            # Plot on each sub-axis of brokenaxes
            box = {}
            for ax_part in ax.axs:
                box = ax_part.boxplot(
                    vanilla_data,
                    positions=[box_positions[system]['Vanilla']],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=True
                )

            # box = current_ax.boxplot(
            #     vanilla_data,
            #     positions=[box_positions[system]['Vanilla']],
            #     widths=box_width,
            #     patch_artist=True,
            #     showfliers=True
            # )

            # Set color and style for Vanilla
            color_idx = list(box_positions.keys()).index(system) * 2
            for patch in box['boxes']:
                patch.set_facecolor(colors[color_idx])
                patch.set_edgecolor('black')
                patch.set_hatch(legend_patterns[0])

            # AQP box
            aqp_data = sys_df[sys_df['Type'] == 'AQP']['Value']
            for ax_part in ax.axs:
                box = ax_part.boxplot(
                    aqp_data,
                    positions=[box_positions[system]['AQP']],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=True
                )
            # box = current_ax.boxplot(
            #     aqp_data,
            #     positions=[box_positions[system]['AQP']],
            #     widths=box_width,
            #     patch_artist=True,
            #     showfliers=True
            # )

            # Set color and style for AQP
            color_idx = list(box_positions.keys()).index(system) * 2 + 1
            for patch in box['boxes']:
                patch.set_facecolor(colors[color_idx])
                patch.set_edgecolor('black')
                patch.set_hatch(legend_patterns[1])

        # for tick in ax.yaxis.get_major_ticks():
        #     tick.label1.set_fontweight('bold')
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("")
        ax.set_xlabel("")

        # Brokenaxes handling
        ax.set_xticks([0.25, 2.25, 4.25])
        ax.set_xticklabels([])

        # Apply settings to all internal axes
        for ax_part in ax.axs:
            ax_part.grid(True, alpha=0.3)
            ax_part.tick_params(axis='y', labelsize=10)
            for tick in ax_part.yaxis.get_major_ticks():
                tick.label1.set_fontweight('bold')
            # Hide x-axis on top part
            if ax_part != ax.axs[0]:
                ax_part.set_xticks([])

        # if benchmark_idx < len(benchmark_names) - 1:
        #     y_lim_max = ax.get_ylim()[1]
        #     y_lim_min = ax.get_ylim()[0]
        #     # y_offset = (y_lim_max - y_lim_min) * 0.08  # Text vertical offset
        # else:
        y_offset = 1

        for system in systems:
            sys_df = df[df['System'] == system]
            if sys_df.empty:
                continue

            # Calculate average and outlier thresholds
            avg_val = sys_df['Value'].mean()
            high_threshold = 11 * avg_val

            # Identify outliers
            outliers = sys_df[
                (sys_df['Value'] > high_threshold)
            ]

            # Find the most extreme outlier in each direction
            max_deviation = 0
            min_deviation = float('inf')
            extreme_max_row = None
            extreme_min_row = None
            extreme_query = None

            for _, row in outliers.iterrows():
                # Calculate deviation ratio
                if row['Value'] > high_threshold:
                    deviation = row['Value'] / avg_val
                    if deviation > max_deviation:
                        max_deviation = deviation
                        extreme_max_row = row
                        extreme_query = row['Query']
                else:
                    deviation = avg_val / row['Value']
                    if deviation > min_deviation:
                        min_deviation = deviation
                        extreme_min_row = row
                        extreme_query = row['Query']

            # If we found an extreme query
            if extreme_query:
                # Get both points for this query (Vanilla and AQP)
                query_points = sys_df[sys_df['Query'] == extreme_query]

                # Annotate both points
                for _, row in query_points.iterrows():
                    xpos = box_positions[system][row['Type']]
                    value = row['Value']

                    # Get matching color
                    system_idx = list(box_positions.keys()).index(system)

                    # Determine position based on value and type
                    is_high = value > avg_val
                    is_left = (row['Type'] == 'Vanilla')

                    # Calculate annotation position
                    if is_left:
                        text_y = value + y_offset
                        va = 'bottom'
                        arrow_rad = 0.2
                    else:
                        text_y = value - y_offset
                        va = 'top'
                        arrow_rad = -0.2

                    # Position text to avoid overlap
                    if is_left:
                        text_x = xpos - 0.3
                        ha = 'right'
                        arrow_rad = -abs(arrow_rad)  # Curve left
                    else:
                        text_x = xpos + 0.3
                        ha = 'left'
                        arrow_rad = abs(arrow_rad)  # Curve right

                    # Create annotation
                    ax.annotate(f"{row['Query']}: {row['Value']:.2f}s",
                                xy=(xpos, value),
                                xytext=(text_x, text_y),
                                fontsize=8,
                                fontweight='bold',
                                color='darkred',
                                arrowprops=dict(
                                    arrowstyle="->",
                                    color='darkred',
                                    lw=1,
                                    connectionstyle=f"arc3,rad={arrow_rad}"
                                ),
                                bbox=dict(
                                    boxstyle="round,pad=0.2",
                                    fc="white",
                                    ec='darkred',
                                    lw=1,
                                    alpha=0.9
                                ),
                                horizontalalignment=ha,
                                verticalalignment=va)

    # Set shared y-axis label
    fig.text(0.093, 0.5, 'Exe Time for Each Query [s]', va='center', rotation='vertical',
             fontsize=12, fontweight='bold')

    # Adjust layout and save
    # plt.tight_layout()
    plt.savefig(f'../figures/{output_name}.pdf', bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def plot_compare_violin_dict(col1_data,
                             col2_data,
                             col3_data,
                             col4_data,
                             col5_data,
                             col6_data,
                             colors,
                             benchmark_names, output_name):
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

    # Process each benchmark
    for benchmark_idx, (ax, benchmark_name) in enumerate(zip(axes, benchmark_names)):
        # Extract data for this benchmark
        col1_data_dict = col1_data[benchmark_idx]
        col2_data_dict = col2_data[benchmark_idx]
        col3_data_dict = col3_data[benchmark_idx]
        col4_data_dict = col4_data[benchmark_idx]
        col5_data_dict = col5_data[benchmark_idx]
        col6_data_dict = col6_data[benchmark_idx]

        # Create DataFrame for seaborn
        plot_data = []

        # Add DuckDB speedup
        for qid in col1_data_dict:
            if qid in col2_data_dict:
                speedup = col1_data_dict[qid] / col2_data_dict[qid]
                plot_data.append({'System': 'DuckDB', 'Speedup': speedup, 'Query': qid})

        # Add DuckDB Common speedup
        for qid in col3_data_dict:
            if qid in col4_data_dict:
                speedup = col3_data_dict[qid] / col4_data_dict[qid]
                plot_data.append({'System': 'DuckDB\n(common)', 'Speedup': speedup, 'Query': qid})

        # Add PostgreSQL speedup
        for qid in col5_data_dict:
            if qid in col6_data_dict:
                speedup = col5_data_dict[qid] / col6_data_dict[qid]
                plot_data.append({'System': 'PostgreSQL\n(common)', 'Speedup': speedup, 'Query': qid})

        df = pd.DataFrame(plot_data)

        # Create split violin plot
        sns.violinplot(data=df, x='System', y='Speedup', log_scale=True, density_norm="width",
                       inner="box", inner_kws=dict(box_width=12, whis_width=10),
                       ax=ax, linewidth=1.5)
        sns.stripplot(data=df, x="System", y="Speedup", ax=ax,
                      color='white', size=4, jitter=True, alpha=0.6, edgecolor="black", linewidth=1)

        for ind, violin in enumerate(ax.findobj(PolyCollection)):
            violin.set_facecolor(colors[ind])

        # # Apply patterns to violins
        # pattern1 = legend_patterns[0]  # For DuckDB and DuckDB(common)
        # pattern2 = legend_patterns[1]  # For PostgreSQL
        # hatches = [pattern1, pattern2, pattern1, pattern2, pattern1, pattern2]
        #
        # # Apply hatches to violin bodies (first 6 collections)
        # for i in range(6):
        #     ax.collections[i].set_hatch(hatches[i])
        #     ax.collections[i].set_edgecolor('black')  # Enhance pattern visibility

        # Customize subplot
        # ax.set_title(benchmark_name, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', labelsize=12)
        ax.set_xticklabels([])
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, alpha=0.3)

        # Remove individual subplot legends
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_ylabel("")
        ax.set_xlabel("")

    # Set shared y-axis label
    fig.text(-0.005, 0.5, 'Speedup for Each Query [x]', va='center', rotation='vertical',
             fontsize=14, fontweight='bold')

    # Adjust layout and save
    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    plt.savefig(f'../figures/{output_name}.pdf', bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def compare_query_by_query_violin_grouped(col1_data, col2_data, colors, group_labels, title):
    assert len(col1_data) == len(col2_data) == len(group_labels), "Expected 3 groups of paired DataFrames"
    all_data = []

    def add_pairwise_speedups(source_df, baseline_df, group_label):
        if len(source_df) > len(baseline_df):
            source_df = source_df[source_df['sub_sql_name'] != '072_spj'].reset_index(drop=True)
        source_mean_dict = source_df["mean"].to_dict()
        baseline_mean_dict = baseline_df["mean"].to_dict()
        speedups_dict = {}
        for idx, diff_value in source_mean_dict.items():
            source_mean = source_mean_dict[idx]
            baseline_mean = baseline_mean_dict[idx]
            speedups_dict[idx] = source_mean / baseline_mean

        speedups = pd.DataFrame.from_dict(speedups_dict, orient="index", columns=["speedup"])
        merged = source_df.copy()
        merged["speedup"] = speedups
        # Append results
        for _, row in merged.iterrows():
            all_data.append({
                "sql_name": row['sql_name'],
                "speedup": row['speedup'],
                "group": group_label
            })

    # Compute speedups per benchmark group
    for i in range(len(col1_data) - 1):
        add_pairwise_speedups(col1_data[i], col2_data[i], group_labels[i])

    df = pd.DataFrame(all_data)

    # Plot setup
    plt.figure(figsize=(6, 4))
    ax = sns.violinplot(data=df, x="group", y="speedup", log_scale=True, density_norm="width", hue="group",
                        inner="box", inner_kws=dict(box_width=12, whis_width=10),
                        palette=colors, linewidth=1.5, legend=True)
    sns.stripplot(data=df, x="group", y="speedup",
                  color='white', size=3, jitter=True, alpha=0.6, edgecolor="black", linewidth=1)

    # Annotate outliers for each group
    for i, group in enumerate(group_labels):
        group_data = df[df["group"] == group]
        if group_data.empty:
            continue

        max_val = group_data["speedup"].max()
        min_val = group_data["speedup"].min()
        max_query = group_data.loc[group_data["speedup"].idxmax(), "sql_name"]
        min_query = group_data.loc[group_data["speedup"].idxmin(), "sql_name"]

        # Annotate max
        plt.annotate(f"{max_query}: {max_val:.1f}x",
                     xy=(i, max_val),
                     xytext=(i - 0.1, max_val * 1.3),
                     fontsize=12, fontweight='bold', color='darkred',
                     arrowprops=dict(arrowstyle="->", color='darkred', lw=1.5,
                                     connectionstyle="arc3,rad=-0.3"),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=1, alpha=0.9),
                     horizontalalignment='right')

        # Annotate min
        plt.annotate(f"{min_query}: {min_val:.1f}x",
                     xy=(i, min_val),
                     xytext=(i - 0.1, min_val * 1.3),
                     fontsize=12, fontweight='bold', color='darkblue',
                     arrowprops=dict(arrowstyle="->", color='darkblue', lw=1.5,
                                     connectionstyle="arc3,rad=0.3"),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkblue", lw=1, alpha=0.9),
                     horizontalalignment='right')

    # Reference line and axis styling
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.ylabel("Speedup for Each Query [x]", fontsize=14, fontweight='bold')
    plt.xlabel("")
    plt.xticks(fontsize=14, fontweight='bold')

    yticks = plt.yticks()[0]
    # plt.yticks(ticks=yticks, labels=[f"$e^{{{int(t)}}}$" for t in yticks], fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()

    # Save
    plt.savefig("../figures/query-by-query_violin_" + title + ".pdf")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    vanilla_duckdb_datas = []
    aqp_duckdb_datas = []
    aqp_duckdb_wo_stats_datas = []
    aqp_duckdb_whole_plan_datas = []
    aqp_duckdb_wo_stats_whole_plan_datas = []

    vanilla_duckdb_sums = []
    aqp_duckdb_sums = []
    aqp_duckdb_wo_stats_sums = []
    aqp_duckdb_whole_plan_sums = []
    aqp_duckdb_wo_stats_whole_plan_sums = []
    aqp_duckdb_wo_reorder_sums = []

    vanilla_duckdb_execute_times = []
    vanilla_duckdb_exe_sums = []
    aqp_duckdb_execute_times = []
    aqp_duckdb_exe_sums = []
    aqp_duckdb_wo_stats_execute_times = []
    aqp_duckdb_wo_stats_exe_sums = []

    vanilla_duckdb_common_datas = []
    aqp_duckdb_common_datas = []
    aqp_duckdb_wo_stats_common_datas = []
    aqp_duckdb_whole_plan_common_datas = []
    aqp_duckdb_wo_stats_whole_plan_common_datas = []
    aqp_duckdb_wo_reorder_datas = []

    vanilla_duckdb_common_sums = []
    aqp_duckdb_common_sums = []
    aqp_duckdb_wo_stats_common_sums = []
    aqp_duckdb_whole_plan_common_sums = []
    aqp_duckdb_wo_stats_whole_plan_common_sums = []

    vanilla_duckdb_execute_common_times = []
    vanilla_duckdb_exe_common_sums = []
    aqp_duckdb_execute_common_times = []
    aqp_duckdb_exe_common_sums = []
    aqp_duckdb_wo_stats_execute_common_times = []
    aqp_duckdb_wo_stats_exe_common_sums = []

    vanilla_pg_datas = []
    aqp_pg_datas = []
    aqp_pg_wo_stats_datas = []
    aqp_pg_whole_plan_datas = []
    aqp_pg_wo_stats_whole_plan_datas = []

    vanilla_pg_sums = []
    aqp_pg_sums = []
    aqp_pg_wo_stats_sums = []
    aqp_pg_whole_plan_sums = []
    aqp_pg_wo_stats_whole_plan_sums = []

    vanilla_pg_execute_times = []
    vanilla_pg_exe_sums = []
    aqp_pg_execute_times = []
    aqp_pg_exe_sums = []
    aqp_pg_wo_stats_execute_times = []
    aqp_pg_wo_stats_exe_sums = []

    for benchmark in benchmarks:
        query_num = 113 if benchmark == "JOB" else 58

        common_queries = []
        if benchmark == "JOB":
            common_queries = ["10c", "12a", "12b", "12c", "13a", "13b", "13c", "13d", "14a",
                              "14b", "14c", "15a", "15b", "15c", "15d", "16b", "16c", "16d",
                              "18a", "18b", "18c", "1a", "1b", "1c", "1d", "20b", "20c",
                              "22d", "23a", "23b", "23c", "25a", "25b", "25c", "26a", "2a",
                              "2b", "2d", "30a", "30b", "30c", "31a", "31b", "31c", "3a",
                              "3b", "3c", "4a", "4b", "4c", "6a", "6b", "6c", "6d",
                              "6e", "6f", "8a", "8b", "8c", "8d", "9a", "9b", "9c"]
        else:
            common_queries = ["1_019_spj", "1_040_spj", "1_099_spj", "1_013_spj", "1_018_spj", "1_072_spj", "1_102_spj",
                              "2_019_spj", "2_040_spj", "2_099_spj", "2_013_spj", "2_018_spj", "2_072_spj", "2_102_spj"]

        # duckdb end to end time
        vanilla_duckdb_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_official_nan.csv"
        aqp_duckdb_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_query_split_rsj.csv"
        aqp_duckdb_wo_stats_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_query_split_rsj_wo_stats_stats.csv"
        aqp_duckdb_wo_reorder_log = os.getcwd() + f"/{benchmark}_result_wo_reorder_luigi/duckdb_query_split_rsj.csv"
        # duckdb breakdown time
        vanilla_duckdb_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_official_breakdown_time_log.csv"
        aqp_duckdb_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_rsj_breakdown_time_log.csv"
        aqp_duckdb_wo_stats_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_rsj_wo_stats_breakdown_time_log.csv"
        # duckdb whole plan time
        aqp_duckdb_whole_plan_exe_time_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_rsj_whole_plan_breakdown_time_log.csv"
        aqp_duckdb_wo_stats_whole_plan_exe_time_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_rsj_whole_plan_wo_stats_breakdown_time_log.csv"
        aqp_duckdb_wo_reorder_breakdown_log = os.getcwd() + f"/{benchmark}_result_wo_reorder_luigi/duckdb_rsj_breakdown_time_log.csv"

        ### prepare duckdb data ###
        vanilla_duckdb_data = plot_duckdb_end2end_results.analyze_csv_data(vanilla_duckdb_log, "duckdb", benchmark)
        # if benchmark == "DSB_100":
        #     vanilla_duckdb_data = vanilla_duckdb_data[vanilla_duckdb_data['sub_sql_name'] != '050'].reset_index(
        #         drop=True)
        aqp_duckdb_data = plot_duckdb_end2end_results.analyze_csv_data(aqp_duckdb_log, "duckdb", benchmark)
        # if benchmark == "DSB_100":
        #     aqp_duckdb_data = aqp_duckdb_data[aqp_duckdb_data['sub_sql_name'] != '050'].reset_index(drop=True)
        aqp_duckdb_wo_stats_data = plot_duckdb_end2end_results.analyze_csv_data(aqp_duckdb_wo_stats_log, "duckdb",
                                                                                benchmark)
        # if benchmark == "DSB_100":
        #     aqp_duckdb_wo_stats_data = aqp_duckdb_wo_stats_data[
        #         aqp_duckdb_wo_stats_data['sub_sql_name'] != '050'].reset_index(drop=True)
        # analyze whole plan execution time
        aqp_duckdb_wo_reorder_data = plot_duckdb_end2end_results.analyze_csv_data(aqp_duckdb_wo_reorder_log, "duckdb",
                                                                                  benchmark)
        aqp_duckdb_whole_plan_data = plot_duckdb_end2end_results.analyze_whole_plan_csv_data(
            aqp_duckdb_whole_plan_exe_time_log, benchmark)
        # if benchmark == "DSB_100":
        #     aqp_duckdb_whole_plan_data.pop('1_050')
        #     aqp_duckdb_whole_plan_data.pop('2_050')
        aqp_duckdb_wo_stats_whole_plan_data = plot_duckdb_end2end_results.analyze_whole_plan_csv_data(
            aqp_duckdb_wo_stats_whole_plan_exe_time_log,
            benchmark)
        # if benchmark == "DSB_100":
        #     aqp_duckdb_wo_stats_whole_plan_data.pop('1_050')
        #     aqp_duckdb_wo_stats_whole_plan_data.pop('2_050')

        vanilla_duckdb_datas.append(vanilla_duckdb_data)
        aqp_duckdb_datas.append(aqp_duckdb_data)
        aqp_duckdb_wo_stats_datas.append(aqp_duckdb_wo_stats_data)
        aqp_duckdb_wo_reorder_datas.append(aqp_duckdb_wo_reorder_data)
        aqp_duckdb_whole_plan_data = {key: aqp_duckdb_whole_plan_data[key] / 1000 for key in
                                      aqp_duckdb_whole_plan_data.keys()}
        aqp_duckdb_whole_plan_datas.append(aqp_duckdb_whole_plan_data)
        aqp_duckdb_wo_stats_whole_plan_data = {key: aqp_duckdb_wo_stats_whole_plan_data[key] / 1000 for key in
                                               aqp_duckdb_wo_stats_whole_plan_data.keys()}
        aqp_duckdb_wo_stats_whole_plan_datas.append(aqp_duckdb_wo_stats_whole_plan_data)

        # analyze duckdb end-to-end time
        vanilla_duckdb_sum = sum(vanilla_duckdb_data['mean'])
        aqp_duckdb_sum = sum(aqp_duckdb_data['mean'])
        aqp_duckdb_wo_stats_sum = sum(aqp_duckdb_wo_stats_data['mean'])
        aqp_duckdb_wo_reorder_sum = sum(aqp_duckdb_wo_reorder_data['mean'])
        aqp_duckdb_whole_plan_sum = sum(aqp_duckdb_whole_plan_data.values())
        aqp_duckdb_wo_stats_whole_plan_sum = sum(aqp_duckdb_wo_stats_whole_plan_data.values())

        vanilla_duckdb_sums.append(vanilla_duckdb_sum)
        aqp_duckdb_sums.append(aqp_duckdb_sum)
        aqp_duckdb_wo_stats_sums.append(aqp_duckdb_wo_stats_sum)
        aqp_duckdb_wo_reorder_sums.append(aqp_duckdb_wo_reorder_sum)
        aqp_duckdb_whole_plan_sums.append(aqp_duckdb_whole_plan_sum)
        aqp_duckdb_wo_stats_whole_plan_sums.append(aqp_duckdb_wo_stats_whole_plan_sum)

        # analyze vanilla breakdown time
        vanilla_duckdb_breakdown_data = plot_duckdb_end2end_results.analyze_vanilla_breakdown(
            vanilla_duckdb_breakdown_log, benchmark)
        # if benchmark == "DSB_100":
        #     vanilla_duckdb_breakdown_data.pop('1_050')
        #     vanilla_duckdb_breakdown_data.pop('2_050')
        vanilla_duckdb_execute_time = {query_name: metrics[' Execute'] / 1000 for query_name, metrics in
                                       vanilla_duckdb_breakdown_data.items()}
        vanilla_duckdb_exe_sum = sum(sub_dict[' Execute'] for sub_dict in vanilla_duckdb_breakdown_data.values()) / 1000
        # analyze AQP breakdown time
        aqp_duckdb_breakdown_data = plot_duckdb_end2end_results.analyze_duckdb_breakdown(aqp_duckdb_breakdown_log,
                                                                                         benchmark)
        # if benchmark == "DSB_100":
        #     aqp_duckdb_breakdown_data.pop('1_050')
        #     aqp_duckdb_breakdown_data.pop('2_050')
        aqp_duckdb_execute_time = {query_name: (metrics['execute'] + metrics['final_exe']) / 1000 for
                                   query_name, metrics in
                                   aqp_duckdb_breakdown_data.items()}
        aqp_duckdb_exe_sum = sum(sub_dict['execute'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000 + \
                             sum(sub_dict['final_exe'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000
        aqp_duckdb_wo_stats_breakdown_data = plot_duckdb_end2end_results.analyze_duckdb_breakdown(
            aqp_duckdb_wo_stats_breakdown_log, benchmark)
        # if benchmark == "DSB_100":
        #     aqp_duckdb_wo_stats_breakdown_data.pop('1_050')
        #     aqp_duckdb_wo_stats_breakdown_data.pop('2_050')
        aqp_duckdb_wo_stats_execute_time = {query_name: (metrics['execute'] + metrics['final_exe']) / 1000 for
                                            query_name, metrics in
                                            aqp_duckdb_wo_stats_breakdown_data.items()}
        aqp_duckdb_wo_stats_exe_sum = sum(
            sub_dict['execute'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data.values()) / 1000 + \
                                      sum(sub_dict['final_exe'] for sub_dict in
                                          aqp_duckdb_wo_stats_breakdown_data.values()) / 1000

        vanilla_duckdb_execute_times.append(vanilla_duckdb_execute_time)
        vanilla_duckdb_exe_sums.append(vanilla_duckdb_exe_sum)
        aqp_duckdb_execute_times.append(aqp_duckdb_execute_time)
        aqp_duckdb_exe_sums.append(aqp_duckdb_exe_sum)
        aqp_duckdb_wo_stats_execute_times.append(aqp_duckdb_wo_stats_execute_time)
        aqp_duckdb_wo_stats_exe_sums.append(aqp_duckdb_wo_stats_exe_sum)

        ### prepare duckdb data of the common queries ###
        vanilla_duckdb_data_common = vanilla_duckdb_data[vanilla_duckdb_data['sql_name'].isin(common_queries)]
        aqp_duckdb_data_common = aqp_duckdb_data[aqp_duckdb_data['sql_name'].isin(common_queries)]
        aqp_duckdb_wo_stats_data_common = aqp_duckdb_wo_stats_data[
            aqp_duckdb_wo_stats_data['sql_name'].isin(common_queries)]
        # analyze whole plan execution time
        aqp_duckdb_whole_plan_data_common = {k: v for k, v in aqp_duckdb_whole_plan_data.items() if
                                             k in common_queries}
        aqp_duckdb_wo_stats_whole_plan_data_common = {k: v for k, v in aqp_duckdb_wo_stats_whole_plan_data.items() if
                                                      k in common_queries}

        vanilla_duckdb_common_datas.append(vanilla_duckdb_data_common)
        aqp_duckdb_common_datas.append(aqp_duckdb_data_common)
        aqp_duckdb_wo_stats_common_datas.append(aqp_duckdb_wo_stats_data_common)
        aqp_duckdb_whole_plan_common_datas.append(aqp_duckdb_whole_plan_data_common)
        aqp_duckdb_wo_stats_whole_plan_common_datas.append(aqp_duckdb_wo_stats_whole_plan_data_common)

        vanilla_duckdb_common_sum = sum(vanilla_duckdb_data_common['mean'])
        aqp_duckdb_common_sum = sum(aqp_duckdb_data_common['mean'])
        aqp_duckdb_wo_stats_common_sum = sum(aqp_duckdb_wo_stats_data_common['mean'])
        aqp_duckdb_whole_plan_common_sum = sum(aqp_duckdb_whole_plan_data_common.values())
        aqp_duckdb_wo_stats_whole_plan_common_sum = sum(aqp_duckdb_wo_stats_whole_plan_data_common.values())

        vanilla_duckdb_common_sums.append(vanilla_duckdb_common_sum)
        aqp_duckdb_common_sums.append(aqp_duckdb_common_sum)
        aqp_duckdb_wo_stats_common_sums.append(aqp_duckdb_wo_stats_common_sum)
        aqp_duckdb_whole_plan_common_sums.append(aqp_duckdb_whole_plan_common_sum)
        aqp_duckdb_wo_stats_whole_plan_common_sums.append(aqp_duckdb_wo_stats_whole_plan_common_sum)

        # analyze vanilla breakdown time
        vanilla_duckdb_breakdown_data_common = {k: v for k, v in vanilla_duckdb_breakdown_data.items() if
                                                k in common_queries}
        vanilla_duckdb_execute_time_common = {query_name: metrics[' Execute'] / 1000 for query_name, metrics in
                                              vanilla_duckdb_breakdown_data_common.items()}
        vanilla_duckdb_exe_sum_common = sum(
            sub_dict[' Execute'] for sub_dict in vanilla_duckdb_breakdown_data_common.values()) / 1000
        # analyze AQP breakdown time
        aqp_duckdb_breakdown_data_common = {k: v for k, v in aqp_duckdb_breakdown_data.items() if
                                            k in common_queries}
        aqp_duckdb_execute_time_common = {query_name: (metrics['execute'] + metrics['final_exe']) / 1000 for
                                          query_name, metrics
                                          in
                                          aqp_duckdb_breakdown_data_common.items()}
        aqp_duckdb_exe_sum_common = sum(
            sub_dict['execute'] for sub_dict in aqp_duckdb_breakdown_data_common.values()) / 1000 + sum(
            sub_dict['final_exe'] for sub_dict in aqp_duckdb_breakdown_data_common.values()) / 1000
        aqp_duckdb_wo_stats_breakdown_data_common = {k: v for k, v in aqp_duckdb_wo_stats_breakdown_data.items() if
                                                     k in common_queries}
        aqp_duckdb_wo_stats_exe_sum_common = sum(
            sub_dict['execute'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data_common.values()) / 1000 + sum(
            sub_dict['final_exe'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data_common.values()) / 1000
        vanilla_duckdb_execute_common_times.append(vanilla_duckdb_execute_time_common)
        vanilla_duckdb_exe_common_sums.append(vanilla_duckdb_exe_sum_common)
        aqp_duckdb_execute_common_times.append(aqp_duckdb_execute_time_common)
        aqp_duckdb_exe_common_sums.append(aqp_duckdb_exe_sum_common)
        aqp_duckdb_wo_stats_exe_common_sums.append(aqp_duckdb_wo_stats_exe_sum_common)

        # postgres end to end time
        vanilla_pg_log = os.getcwd() + f"/{benchmark}_result_luigi/pg_Official.csv"
        aqp_pg_log = os.getcwd() + f"/{benchmark}_result_luigi/pg_QuerySplit_with_stats.csv"
        aqp_pg_wo_stats_log = os.getcwd() + f"/{benchmark}_result_luigi/pg_QuerySplit.csv"
        # postgres breakdown time
        vanilla_pg_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/pg_Official_breakdown_time_log.csv"
        aqp_pg_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/QuerySplit_with_stats_breakdown_time_log.csv"
        aqp_pg_wo_stats_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/QuerySplit_wo_stats_breakdown_time_log.csv"
        # postgres whole plan time
        aqp_pg_whole_plan_exe_time_log = os.getcwd() + f"/{benchmark}_result_luigi/QuerySplit_whole_plan_breakdown_time_log.csv"
        aqp_pg_wo_stats_whole_plan_exe_time_log = os.getcwd() + f"/{benchmark}_result_luigi/QuerySplit_whole_plan_wo_stats_breakdown_time_log.csv"

        ### prepare postgres data ###
        vanilla_pg_data = plot_duckdb_end2end_results.analyze_csv_data(vanilla_pg_log, "postgres", benchmark)
        aqp_pg_data = plot_duckdb_end2end_results.analyze_csv_data(aqp_pg_log, "postgres", benchmark)
        aqp_pg_wo_stats_data = plot_duckdb_end2end_results.analyze_csv_data(aqp_pg_wo_stats_log, "postgres", benchmark)
        aqp_pg_whole_plan_data = plot_duckdb_end2end_results.analyze_whole_plan_csv_data(aqp_pg_whole_plan_exe_time_log,
                                                                                         benchmark)
        aqp_pg_wo_stats_whole_plan_data = plot_duckdb_end2end_results.analyze_whole_plan_csv_data(
            aqp_pg_wo_stats_whole_plan_exe_time_log,
            benchmark)

        vanilla_pg_datas.append(vanilla_pg_data)
        aqp_pg_datas.append(aqp_pg_data)
        aqp_pg_wo_stats_datas.append(aqp_pg_wo_stats_data)
        aqp_pg_whole_plan_datas.append(aqp_pg_whole_plan_data)
        aqp_pg_wo_stats_whole_plan_datas.append(aqp_pg_wo_stats_whole_plan_data)

        vanilla_pg_sum = sum(vanilla_pg_data['mean'])
        aqp_pg_sum = sum(aqp_pg_data['mean'])
        aqp_pg_wo_stats_sum = sum(aqp_pg_wo_stats_data['mean'])
        aqp_pg_whole_plan_sum = sum(aqp_pg_whole_plan_data.values())
        aqp_pg_wo_stats_whole_plan_sum = sum(aqp_pg_wo_stats_whole_plan_data.values())

        vanilla_pg_sums.append(vanilla_pg_sum)
        aqp_pg_sums.append(aqp_pg_sum)
        aqp_pg_wo_stats_sums.append(aqp_pg_wo_stats_sum)
        aqp_pg_whole_plan_sums.append(aqp_pg_whole_plan_sum)
        aqp_pg_wo_stats_whole_plan_sums.append(aqp_pg_wo_stats_whole_plan_sum)

        # analyze vanilla breakdown time
        vanilla_pg_breakdown_data = plot_duckdb_end2end_results.analyze_vanilla_breakdown(vanilla_pg_breakdown_log,
                                                                                          benchmark)
        vanilla_pg_execute_time = {query_name: metrics[' Execute'] for query_name, metrics in
                                   vanilla_pg_breakdown_data.items()}
        vanilla_pg_exe_sum = sum(sub_dict[' Execute'] for sub_dict in vanilla_pg_breakdown_data.values())
        # analyze AQP breakdown time
        aqp_pg_breakdown_data = plot_duckdb_end2end_results.analyze_pg_breakdown(aqp_pg_breakdown_log, True, benchmark)
        aqp_pg_execute_time = {query_name: metrics['execute'] + metrics['final_exe'] for query_name, metrics in
                               aqp_pg_breakdown_data.items()}
        aqp_pg_exe_sum = sum(sub_dict['execute'] for sub_dict in aqp_pg_breakdown_data.values()) + \
                         sum(sub_dict['final_exe'] for sub_dict in aqp_pg_breakdown_data.values())
        aqp_pg_wo_stats_breakdown_data = plot_duckdb_end2end_results.analyze_pg_breakdown(aqp_pg_wo_stats_breakdown_log,
                                                                                          False, benchmark)
        aqp_pg_wo_stats_execute_time = {query_name: metrics['execute'] + metrics['final_exe'] for query_name, metrics in
                                        aqp_pg_wo_stats_breakdown_data.items()}
        aqp_pg_wo_stats_exe_sum = sum(sub_dict['execute'] for sub_dict in aqp_pg_wo_stats_breakdown_data.values()) + \
                                  sum(sub_dict['final_exe'] for sub_dict in aqp_pg_wo_stats_breakdown_data.values())

        vanilla_pg_execute_times.append(vanilla_pg_execute_time)
        vanilla_pg_exe_sums.append(vanilla_pg_exe_sum)
        aqp_pg_execute_times.append(aqp_pg_execute_time)
        aqp_pg_exe_sums.append(aqp_pg_exe_sum)
        aqp_pg_wo_stats_execute_times.append(aqp_pg_wo_stats_execute_time)
        aqp_pg_wo_stats_exe_sums.append(aqp_pg_wo_stats_exe_sum)

    plot_end2end_combined(vanilla_duckdb_sums, bar_colors[0],
                          aqp_duckdb_sums, bar_colors[1],
                          vanilla_duckdb_common_sums, bar_colors[2],
                          aqp_duckdb_common_sums, bar_colors[3],
                          vanilla_pg_sums, bar_colors[4],
                          aqp_pg_sums, bar_colors[5],
                          ["Vanilla", "Plan-based AQP"],
                          ["darkgray", "lightgray"], [None, 'x'],
                          benchmarks,
                          "Total End-to-end Time (s)",
                          f'q1_fig_common')

    plot_end2end_combined(aqp_duckdb_wo_stats_sums, bar_colors[1],
                          aqp_duckdb_sums, bar_colors[1],
                          aqp_duckdb_wo_stats_common_sums, bar_colors[3],
                          aqp_duckdb_common_sums, bar_colors[3],
                          aqp_pg_wo_stats_sums, bar_colors[5],
                          aqp_pg_sums, bar_colors[5],
                          ["Plan-based AQP w/o Updating Cardinality", "Plan-based AQP with Updating Cardinality"],
                          ["lightgray", "lightgray"], ['/', 'x'],
                          benchmarks,
                          "Total End-to-end Time (s)",
                          f'q2_fig_common')

    plot_end2end_combined(vanilla_duckdb_exe_sums, bar_colors[0],
                          aqp_duckdb_wo_stats_whole_plan_sums, bar_colors[0],
                          vanilla_duckdb_exe_common_sums, bar_colors[2],
                          aqp_duckdb_wo_stats_whole_plan_common_sums, bar_colors[2],
                          vanilla_pg_exe_sums, bar_colors[4],
                          aqp_pg_wo_stats_whole_plan_sums, bar_colors[4],
                          ["Vanilla", "Vanilla with Specified Join Order and Operator"],
                          ["darkgray", "darkgray"], [None, '+'],
                          benchmarks,
                          "Total Execution Time (s)",
                          f'q3_fig_common')

    plot_end2end_combined(aqp_duckdb_whole_plan_sums, bar_colors[1],
                          aqp_duckdb_exe_sums, bar_colors[1],
                          aqp_duckdb_whole_plan_common_sums, bar_colors[3],
                          aqp_duckdb_exe_common_sums, bar_colors[3],
                          aqp_pg_whole_plan_sums, bar_colors[5],
                          aqp_pg_exe_sums, bar_colors[5],
                          ["Plan-based AQP w/o Splitting Plan", "Plan-based AQP with Splitting Plan"],
                          ["lightgray", "lightgray"], ['\\', 'x'],
                          benchmarks,
                          "Total Execution Time (s)",
                          f'q4_fig_common')

    # vanilla VS AQP without join reordering
    plot_end2end_wo_reorder(vanilla_duckdb_sums, bar_colors[0],
                            aqp_duckdb_wo_reorder_sums, bar_colors[6],
                            benchmarks,
                            ["Vanilla DuckDB", "AQP-DuckDB w/o join reorder module"],
                            [bar_colors[0], bar_colors[6]], "Total End-to-end Time (s)",
                            'vanilla_vs_AQP_wo_reordering_join')

    plot_violin_dataframe(vanilla_duckdb_datas, aqp_duckdb_datas, vanilla_duckdb_common_datas,
                          aqp_duckdb_common_datas, vanilla_pg_datas, aqp_pg_datas,
                          [bar_colors[0], bar_colors[1], bar_colors[2], bar_colors[3], bar_colors[4], bar_colors[5]],
                          [None, 'x'], benchmarks,
                          f'qbq_q1_fig')

    plot_violin_dataframe(aqp_duckdb_wo_stats_datas, aqp_duckdb_datas, aqp_duckdb_wo_stats_common_datas,
                          aqp_duckdb_common_datas, aqp_pg_wo_stats_datas, aqp_pg_datas,
                          [bar_colors[1], bar_colors[1], bar_colors[3], bar_colors[3], bar_colors[5], bar_colors[5]],
                          ['/', 'x'], benchmarks,
                          f'qbq_q2_fig')

    plot_violin_dict(vanilla_duckdb_execute_times, aqp_duckdb_wo_stats_whole_plan_datas,
                     vanilla_duckdb_execute_common_times, aqp_duckdb_wo_stats_whole_plan_common_datas,
                     vanilla_pg_execute_times, aqp_pg_wo_stats_whole_plan_datas,
                     [bar_colors[0], bar_colors[0], bar_colors[2], bar_colors[2], bar_colors[4], bar_colors[4]],
                     [None, '+'], benchmarks,
                     f'qbq_q3_fig')

    plot_violin_dict(aqp_duckdb_whole_plan_datas, aqp_duckdb_execute_times,
                     aqp_duckdb_whole_plan_common_datas, aqp_duckdb_execute_common_times,
                     aqp_pg_whole_plan_datas, aqp_pg_execute_times,
                     [bar_colors[1], bar_colors[1], bar_colors[3], bar_colors[3], bar_colors[5], bar_colors[5]],
                     ['\\', 'x'], benchmarks,
                     f'qbq_q4_fig')

    plot_boxplot_dataframe(vanilla_duckdb_datas, aqp_duckdb_datas, vanilla_duckdb_common_datas,
                           aqp_duckdb_common_datas, vanilla_pg_datas, aqp_pg_datas,
                           [bar_colors[0], bar_colors[1], bar_colors[2], bar_colors[3], bar_colors[4], bar_colors[5]],
                           [None, 'x'],
                           [[(-1, 15), (30, 35)], [(-0.5, 3), (10, 12)], [(-5, 100), (3775, 3800)]], benchmarks,
                           f'box_qbq_q1_fig')

    plot_boxplot_dataframe(aqp_duckdb_wo_stats_datas, aqp_duckdb_datas, aqp_duckdb_wo_stats_common_datas,
                           aqp_duckdb_common_datas, aqp_pg_wo_stats_datas, aqp_pg_datas,
                           [bar_colors[1], bar_colors[1], bar_colors[3], bar_colors[3], bar_colors[5], bar_colors[5]],
                           ['/', 'x'],
                           [[(-0.5, 4), (7, 11)], [(-0.5, 3), (7, 8)], [(-5, 35), (80, 85)]], benchmarks,
                           f'box_qbq_q2_fig')

    plot_boxplot_dict(vanilla_duckdb_execute_times, aqp_duckdb_wo_stats_whole_plan_datas,
                      vanilla_duckdb_execute_common_times, aqp_duckdb_wo_stats_whole_plan_common_datas,
                      vanilla_pg_execute_times, aqp_pg_wo_stats_whole_plan_datas,
                      [bar_colors[0], bar_colors[0], bar_colors[2], bar_colors[2], bar_colors[4], bar_colors[4]],
                      [None, '+'],
                      [[(-0.5, 8.5), (34, 35)], [(-0.5, 2), (7, 11)], [(-5, 90), (3800, 3810)]], benchmarks,
                      f'box_qbq_q3_fig')

    plot_boxplot_dict(aqp_duckdb_whole_plan_datas, aqp_duckdb_execute_times,
                      aqp_duckdb_whole_plan_common_datas, aqp_duckdb_execute_common_times,
                      aqp_pg_whole_plan_datas, aqp_pg_execute_times,
                      [bar_colors[1], bar_colors[1], bar_colors[3], bar_colors[3], bar_colors[5], bar_colors[5]],
                      ['\\', 'x'],
                      [[(-0.5, 3), (10, 11)], [(-0.5, 1.5), (2, 2.5)], [(-5, 25), (80, 82)]], benchmarks,
                      f'box_qbq_q4_fig')
    #

    plot_boxplot_dataframe_wo_reorder(vanilla_duckdb_datas, bar_colors[0],
                                      aqp_duckdb_wo_reorder_datas, bar_colors[6],
                                      [(-0.5, 18), (63, 70)],
                                      benchmarks,
                                      ["Vanilla DuckDB", "AQP-DuckDB w/o join reorder module"],
                                      [bar_colors[0], bar_colors[6]], "E2E Time for Each Query [s]",
                                      f'box_qbq_q5_fig')

    plot_compare_violin_dataframe(vanilla_duckdb_datas, aqp_duckdb_datas, vanilla_duckdb_common_datas,
                                  aqp_duckdb_common_datas, vanilla_pg_datas, aqp_pg_datas,
                                  [bar_colors[1], bar_colors[3], bar_colors[5]], benchmarks,
                                  f'qbq_compare_q1_fig')

    plot_compare_violin_dataframe(aqp_duckdb_wo_stats_datas, aqp_duckdb_datas, aqp_duckdb_wo_stats_common_datas,
                                  aqp_duckdb_common_datas, aqp_pg_wo_stats_datas, aqp_pg_datas,
                                  [bar_colors[1], bar_colors[3], bar_colors[5]], benchmarks,
                                  f'qbq_compare_q2_fig')

    plot_compare_violin_dict(vanilla_duckdb_execute_times, aqp_duckdb_wo_stats_whole_plan_datas,
                             vanilla_duckdb_execute_common_times, aqp_duckdb_wo_stats_whole_plan_common_datas,
                             vanilla_pg_execute_times, aqp_pg_wo_stats_whole_plan_datas,
                             [bar_colors[0], bar_colors[2], bar_colors[4]], benchmarks,
                             f'qbq_compare_q3_fig')

    plot_compare_violin_dict(aqp_duckdb_whole_plan_datas, aqp_duckdb_execute_times,
                             aqp_duckdb_whole_plan_common_datas, aqp_duckdb_execute_common_times,
                             aqp_pg_whole_plan_datas, aqp_pg_execute_times,
                             [bar_colors[1], bar_colors[3], bar_colors[5]], benchmarks,
                             f'qbq_compare_q4_fig')

    compare_query_by_query_violin_grouped(vanilla_duckdb_datas, aqp_duckdb_wo_reorder_datas,
                                          ['#5A6F6A', '#DABDAE', '#03BAFC'], benchmarks,
                                          title=f'vanilla_vs_AQP_wo_reordering_join')
