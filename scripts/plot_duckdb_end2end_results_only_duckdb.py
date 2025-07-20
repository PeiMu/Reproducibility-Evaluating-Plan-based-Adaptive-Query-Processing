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

if not os.path.exists("../figures"):
    os.makedirs("../figures")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

benchmarks = ["job", "dsb_10", "dsb_100"]

bar_colors = [
    "#006400",  # dark green, vanilla duckdb
    '#00FF00',  # bright green, AQP duckdb
    '#03fcd7',  # bright blue-green, AQP duckdb without join reordering
]


# bar_colors = [
#     "darkgray",
#     "lightgray",
#     "darkgray",
#     "lightgray"
# ]


def plot_end2end(col1_data, col1_color,
                 col2_data, col2_color,
                 group_names,
                 legend_names, legend_colors, title,
                 name):
    assert len(col1_data) == len(col2_data) == len(group_names), "Data and group_names must match in length"
    n = len(group_names)

    fig, ax = plt.subplots(figsize=(6, 4))

    bar_width = 0.35
    indices = list(range(n))

    bars = []
    for i in range(n):
        # Plot col1 and col2 bars side-by-side
        bar1 = ax.bar(i - bar_width / 2, col1_data[i], width=bar_width, color=col1_color, edgecolor='black')
        bar2 = ax.bar(i + bar_width / 2, col2_data[i], width=bar_width, color=col2_color, edgecolor='black')
        bars.append((bar1, bar2))

        # Draw vertical and horizontal comparison lines between col1_data[i] and col2_data[i]
        y1 = col1_data[i]
        y2 = col2_data[i]
        x1 = i - bar_width / 2
        x2 = i + bar_width / 2

        # Ensure y1 is the smaller
        swap_flag = False
        if y1 > y2:
            y1, y2 = y2, y1
            x1, x2 = x2, x1
            swap_flag = True

        # Vertical dotted line
        ax.plot([(x1 + x2) / 2, (x1 + x2) / 2], [y1, y2], color='red', linestyle=':', linewidth=1)
        # Horizontal solid lines
        ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y1, y1], color='red', linestyle='-', linewidth=1)
        ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y2, y2], color='red', linestyle='-', linewidth=1)

        # Speedup or slowdown annotation
        y_speedup = y2 + max(col1_data + col2_data) * 0.05
        if swap_flag:
            speedup = y2 / y1
            ax.text((x1 + x2) / 2, y_speedup, f'{speedup:.2f}x ↑',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
        else:
            slowdown = y1 / y2
            ax.text((x1 + x2) / 2, y_speedup, f'{slowdown:.2f}x ↓',
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
        plt.text(i + bar_width / 2, col2_data[i] + max(col1_data + col2_data) * 0.01,
                 f'{col2_data[i]:.2f}', ha='center', fontsize=9)

    plt.ylim(0, max(col1_data + col2_data) * 2)
    plt.tight_layout()
    plt.savefig('../figures/' + name + '.pdf')
    plt.clf()
    plt.close()


def plot_end2end_stacked(col1_data, col1_pattern,
                         col2_data, col2_pattern,
                         legend_names, legend_colors, title,
                         name):
    values = [sum(col1_data.values()), sum(col2_data.values())]
    colors = ['lightblue', '#fcdb00', '#ff954a', 'lightgreen', '#e695f5']
    patterns = [col1_pattern, col2_pattern]
    labels = ["Other", "Analyze", "Optimization", "AQP-Process", "Execution"]

    other_time = {0: col1_data["Other"], 1: col2_data["Other"]}
    analyze_time = {0: col1_data["Analyze"], 1: col2_data["Analyze"]}
    optimization_time = {0: col1_data["Optimization"], 1: col2_data["Optimization"]}
    aqp_process_time = {0: col1_data["AQP-Process"], 1: col2_data["AQP-Process"]}
    execution_time = {0: col1_data["Execution"], 1: col2_data["Execution"]}

    x_positions = [0, 0.5]

    fig, ax = plt.subplots()

    bars = []
    for i in range(len(values)):
        bar_sections = []
        bar_sections.append(
            ax.bar(x_positions[i], other_time[i], width=0.4, color=colors[0], edgecolor='black', label=labels[0]))
        bar_sections.append(
            ax.bar(x_positions[i], analyze_time[i], bottom=other_time[i], width=0.4, color=colors[1], edgecolor='black',
                   label=labels[1]))
        bar_sections.append(
            ax.bar(x_positions[i], optimization_time[i], bottom=other_time[i] + analyze_time[i], width=0.4,
                   color=colors[2], edgecolor='black', label=labels[2]))
        bar_sections.append(
            ax.bar(x_positions[i], aqp_process_time[i], bottom=other_time[i] + analyze_time[i] + optimization_time[i],
                   width=0.4, color=colors[3], edgecolor='black', label=labels[3]))
        bar_sections.append(
            ax.bar(x_positions[i], execution_time[i],
                   bottom=other_time[i] + analyze_time[i] + optimization_time[i] + aqp_process_time[i],
                   width=0.4, color=colors[4], edgecolor='black', label=labels[4]))
        for section in bar_sections:
            section[0].set_hatch(patterns[i])
        bars.append(bar_sections)

    # Add labels and formatting
    ax.set_xticks([0.25])  # Center labels for groups
    ax.set_xticklabels(['DuckDB'])

    # Add vertical dotted lines, horizontal solid lines, and speedup representation
    x1, x2 = x_positions[0], x_positions[1]  # Get x positions of bars in the group
    y1, y2 = values[0], values[1]  # Get heights of bars in the group

    # Ensure y1 is the smaller value and y2 is the larger value
    swap_flag = False
    if y1 > y2:
        y1, y2 = y2, y1
        x1, x2 = x2, x1  # Swap x positions accordingly
        swap_flag = True

    # Draw a vertical dotted line at the middle of x position of the lower bar
    ax.plot([(x1 + x2) / 2, (x1 + x2) / 2], [y1, y2], color='red', linestyle=':', linewidth=1)

    # Draw a horizontal solid line at the height of the higher bar, centered at the middle
    ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y2, y2], color='red', linestyle='-', linewidth=1)
    # Draw a horizontal solid line at the height of the lower bar, centered at the middle
    ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y1, y1], color='red', linestyle='-', linewidth=1)

    # Adjust the position of the speedup representation slightly above the higher bar
    y_speedup = y2 * 1.1  # Slightly above the higher bar

    if swap_flag:
        # Compute speedup
        speedup = y2 / y1
        # Draw speedup annotation in red
        ax.text((x1 + x2) / 2, y_speedup, f'Speedup: {speedup:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
    else:
        # # Compute slowdown
        # slowdown = y2 / y1 - 1
        # # Draw speedup annotation in red
        # ax.text((x1 + x2) / 2, y_speedup, f'Slowdown: {slowdown * 100:.2f}%',
        #         ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
        # Compute slowdown
        slowdown = y1 / y2
        # Draw speedup annotation in red
        ax.text((x1 + x2) / 2, y_speedup, f'Slowdown: {slowdown:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')

    # Add the pattern legend
    pattern_patches = [Patch(facecolor=legend_colors[0], edgecolor='black', hatch=patterns[0], label=legend_names[0]),
                       Patch(facecolor=legend_colors[1], edgecolor='black', hatch=patterns[1], label=legend_names[1])]
    pattern_legend = ax.legend(handles=pattern_patches, loc='upper left', handleheight=2, handlelength=3)

    # Add the color legend
    color_patches = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in reversed(range(5))]
    color_legend = ax.legend(handles=color_patches, loc='upper right', handleheight=2, handlelength=3)

    ax.add_artist(pattern_legend)

    plt.ylabel(title, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10, fontweight='bold')
    for i, value in enumerate(values):
        plt.text(x_positions[i], value + max(values) * 0.01, f'{value:.2f}', ha='center', fontsize=10)
    plt.ylim(0, max(values) * 2)
    plt.tight_layout()
    plt.savefig('../figures/' + name + '.pdf')
    plt.clf()
    plt.close()


def plot_query_by_query(col1_data, col1_color,
                        col2_data, col2_color):
    x = np.arange(len(col1_data))
    plt.figure(figsize=(12, 6))

    col1_data["perf_diff"] = col1_data['mean'] - col2_data['mean']
    perf_diff_sorted = col1_data.sort_values(by="perf_diff", ascending=False)
    plt.bar(perf_diff_sorted['sql_name'], perf_diff_sorted['perf_diff'], color='skyblue')
    plt.xticks(x, labels=perf_diff_sorted['sql_name'], rotation=90, ha='right')
    plt.xlabel("Query ID")
    plt.ylabel("Query by query time difference (s)")

    plt.tight_layout()
    plt.savefig("../figures/plot_query_by_query.pdf")
    plt.clf()


def plot_query_by_query_box_chart_with_deviation(data, category):
    # Calculate the Q1 and Q3
    median = data['median']
    q1 = median + data['stddev'] * stats.norm.ppf(0.25)
    q3 = median + data['stddev'] * stats.norm.ppf(0.75)
    whislo = q1 - (q3 - q1) * 1.5
    whishi = q3 + (q3 - q1) * 1.5

    keys = ['med', 'mean', 'q1', 'q3', 'whislo', 'whishi']
    boxplot_stats = [dict(zip(keys, vals)) for vals in zip(median, data['mean'], q1, q3, whislo, whishi)]
    fig, ax = plt.subplots()
    ax.bxp(boxplot_stats, showmeans=False, showfliers=False)
    ax.set_xlabel('query id')
    ax.set_xticks(range(1, len(data) + 1))  # Set tick positions (1-based index for bxp)
    ax.set_xticklabels(data['sql_name'], rotation=90, fontsize=4)  # Set labels and rotate for better visibility
    ax.set_ylabel('End to end Time (s)')
    ax.set_ylim(0, max(data['max']) + 1)

    # plt.title(title)
    # plt.show()
    plt.savefig('../figures/' + category + '_with_deviation.pdf')
    plt.clf()
    plt.close()

    return boxplot_stats


def plot_query_by_query_box_chart_compare_with_deviation(official_stats, rsj_stats):
    # Create positions for the two sets
    positions1 = np.arange(1, query_num + 1)  # Positions for Set 1
    positions2 = positions1 + 0.4  # Offset for Set 2

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(20, 8))  # Adjust the size for query_num categories

    # Plot the first set of boxplots
    ax.bxp(official_stats, positions=positions1, showmeans=False, showfliers=False, widths=0.3, patch_artist=True,
           boxprops=dict(facecolor="yellow", edgecolor="black"))

    # Plot the second set of boxplots
    ax.bxp(rsj_stats, positions=positions2, showmeans=False, showfliers=False, widths=0.3, patch_artist=True,
           boxprops=dict(facecolor="blue", edgecolor="black"))

    # Customize the x-axis
    ax.set_xticks((positions1 + positions2) / 2)  # Set ticks between the two sets
    ax.set_xticklabels([f'Category {i}' for i in range(1, query_num + 1)], rotation=90)  # Rotate for readability

    # Add labels and title
    ax.set_ylabel('Values')
    ax.set_title(f"Comparison of Two Sets Across {query_num} Queries")

    # Add a legend
    ax.legend(['Set 1', 'Set 2'], loc='upper left')

    # Tight layout to handle label spacing
    plt.tight_layout()

    # Show the plot
    # plt.show()
    plt.savefig('../figures/compare_with_deviation.pdf')
    plt.clf()
    plt.close()


def compare_query_by_query_violin_grouped(col1_data, col2_data, colors, group_labels, title):
    assert len(col1_data) == len(col2_data) == len(group_labels), "Expected 3 groups of paired DataFrames"
    all_data = []

    def add_pairwise_speedups(source_df, baseline_df, group_label):
        diff = source_df["mean"] - baseline_df["mean"]
        diff_dict = diff.to_dict()
        source_mean_dict = source_df["mean"].to_dict()
        baseline_mean_dict = baseline_df["mean"].to_dict()
        speedups_dict = {}
        for idx, diff_value in diff_dict.items():
            source_mean = source_mean_dict[idx]
            baseline_mean = baseline_mean_dict[idx]
            if diff_value > 0:
                speedups_dict[idx] = diff_value / baseline_mean * 100
            else:
                speedups_dict[idx] = diff_value / source_mean * 100
            if -e < speedups_dict[idx] < e:
                speedups_dict[idx] = 0
            elif speedups_dict[idx] >= e:
                speedups_dict[idx] = np.log(speedups_dict[idx])
            else:
                speedups_dict[idx] = -np.log(-speedups_dict[idx])

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
    for i in range(len(col1_data)):
        add_pairwise_speedups(col1_data[i], col2_data[i], group_labels[i])

    df = pd.DataFrame(all_data)

    # Plot setup
    plt.figure(figsize=(12, 8))
    ax = sns.violinplot(data=df, x="group", y="speedup", hue="group",
                        inner="box", inner_kws=dict(box_width=25, whis_width=5),
                        palette=colors, legend=False)
    sns.stripplot(data=df, x="group", y="speedup",
                  color='black', size=4, jitter=True, alpha=0.6, edgecolor="auto", linewidth=0.3)

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
        plt.annotate(f"{max_query}: {math.exp(max_val):.1f}%",
                     xy=(i, max_val),
                     xytext=(i - 0.1, max_val * 1.3),
                     fontsize=12, fontweight='bold', color='darkred',
                     arrowprops=dict(arrowstyle="->", color='darkred', lw=1.5,
                                     connectionstyle="arc3,rad=-0.3"),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=1, alpha=0.9),
                     horizontalalignment='right')

        # Annotate min
        plt.annotate(f"{min_query}: {-math.exp(-min_val):.1f}%",
                     xy=(i, min_val),
                     xytext=(i - 0.1, min_val * 1.3),
                     fontsize=12, fontweight='bold', color='darkblue',
                     arrowprops=dict(arrowstyle="->", color='darkblue', lw=1.5,
                                     connectionstyle="arc3,rad=0.3"),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkblue", lw=1, alpha=0.9),
                     horizontalalignment='right')

    # Reference line and axis styling
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.ylabel("Log-Scaled End2end Speedup/Slowdown (%)", fontsize=14, fontweight='bold')
    plt.xlabel("")
    plt.xticks(fontsize=14, fontweight='bold')

    yticks = plt.yticks()[0]
    plt.yticks(ticks=yticks, labels=[f"$e^{{{int(t)}}}$" for t in yticks], fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()

    # Save
    plt.savefig("../figures/query-by-query_violin_" + title + ".pdf")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    vanilla_duckdb_sums = []
    aqp_duckdb_sums = []
    aqp_duckdb_wo_stats_sums = []
    aqp_duckdb_wo_reorder_sums = []
    vanilla_duckdb_datas = []
    aqp_duckdb_datas = []
    aqp_duckdb_wo_stats_datas = []
    aqp_duckdb_wo_reorder_datas = []
    for benchmark in benchmarks:
        query_num = 113 if benchmark == "job" else 58

        # duckdb end to end time
        vanilla_duckdb_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_official_nan.csv"
        aqp_duckdb_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_query_split_rsj.csv"
        aqp_duckdb_wo_stats_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_query_split_rsj_wo_stats_stats.csv"
        aqp_duckdb_wo_reorder_log = os.getcwd() + f"/{benchmark}_result_wo_reorder_luigi/duckdb_query_split_rsj.csv"

        # duckdb breakdown time
        vanilla_duckdb_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_official_breakdown_time_log.csv"
        aqp_duckdb_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_rsj_breakdown_time_log.csv"
        aqp_duckdb_wo_stats_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_rsj_wo_stats_breakdown_time_log.csv"
        aqp_duckdb_whole_plan_exe_time_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_rsj_whole_plan_breakdown_time_log.csv"
        aqp_duckdb_wo_stats_whole_plan_exe_time_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_rsj_whole_plan_wo_stats_breakdown_time_log.csv"
        aqp_duckdb_wo_reorder_breakdown_log = os.getcwd() + f"/{benchmark}_result_wo_reorder_luigi/duckdb_rsj_breakdown_time_log.csv"

        ### prepare duckdb data ###
        vanilla_duckdb_data = plot_duckdb_end2end_results.analyze_csv_data(vanilla_duckdb_log, "duckdb", benchmark)
        if benchmark == "dsb_100":
            vanilla_duckdb_data = vanilla_duckdb_data[vanilla_duckdb_data['sub_sql_name'] != '072_spj'].reset_index(
                drop=True)
        aqp_duckdb_data = plot_duckdb_end2end_results.analyze_csv_data(aqp_duckdb_log, "duckdb", benchmark)
        aqp_duckdb_wo_stats_data = plot_duckdb_end2end_results.analyze_csv_data(aqp_duckdb_wo_stats_log, "duckdb",
                                                                                benchmark)
        aqp_duckdb_wo_reorder_data = plot_duckdb_end2end_results.analyze_csv_data(aqp_duckdb_wo_reorder_log, "duckdb",
                                                                                  benchmark)
        vanilla_duckdb_datas.append(vanilla_duckdb_data)
        aqp_duckdb_datas.append(aqp_duckdb_data)
        aqp_duckdb_wo_stats_datas.append(aqp_duckdb_wo_stats_data)
        aqp_duckdb_wo_reorder_datas.append(aqp_duckdb_wo_reorder_data)

        vanilla_duckdb_sum = sum(vanilla_duckdb_data['mean'])
        aqp_duckdb_sum = sum(aqp_duckdb_data['mean'])
        aqp_duckdb_wo_stats_sum = sum(aqp_duckdb_wo_stats_data['mean'])
        aqp_duckdb_wo_reorder_sum = sum(aqp_duckdb_wo_reorder_data['mean'])

        vanilla_duckdb_sums.append(vanilla_duckdb_sum)
        aqp_duckdb_sums.append(aqp_duckdb_sum)
        aqp_duckdb_wo_stats_sums.append(aqp_duckdb_wo_stats_sum)
        aqp_duckdb_wo_reorder_sums.append(aqp_duckdb_wo_reorder_sum)

        # # analyze vanilla breakdown time
        # vanilla_duckdb_breakdown_data = plot_duckdb_end2end_results.analyze_vanilla_breakdown(vanilla_duckdb_breakdown_log)
        # vanilla_duckdb_execute_time = {query_name: metrics[' Execute'] for query_name, metrics in
        #                                vanilla_duckdb_breakdown_data.items()}
        # vanilla_exe_sum = sum(sub_dict[' Execute'] for sub_dict in vanilla_duckdb_breakdown_data.values()) / 1000
        # vanilla_opt_sum = sum(sub_dict['PreOptimize'] for sub_dict in vanilla_duckdb_breakdown_data.values()) / 1000 + \
        #                   sum(sub_dict[' final-PostOptimize'] for sub_dict in vanilla_duckdb_breakdown_data.values()) / 1000
        # # we only need 1. execution time, 2. AQP-process time, 3. optimization time, and 4. other time (including parse, etc.)
        # vanilla_duckdb_breakdown = dict()
        # vanilla_duckdb_breakdown['Execution'] = vanilla_exe_sum
        # vanilla_duckdb_breakdown['AQP-Process'] = 0
        # vanilla_duckdb_breakdown['Optimization'] = vanilla_opt_sum
        # vanilla_duckdb_breakdown['Analyze'] = 0
        # vanilla_duckdb_breakdown['Other'] = vanilla_duckdb_sum - vanilla_exe_sum - vanilla_opt_sum
        #
        # # analyze whole plan execution time
        # aqp_duckdb_whole_plan_data = plot_duckdb_end2end_results.analyze_whole_plan_csv_data(aqp_duckdb_whole_plan_exe_time_log)
        # aqp_duckdb_wo_stats_whole_plan_data = plot_duckdb_end2end_results.analyze_whole_plan_csv_data(aqp_duckdb_wo_stats_whole_plan_exe_time_log)
        #
        # aqp_duckdb_whole_plan_sum = sum(aqp_duckdb_whole_plan_data.values()) / 1000
        # aqp_duckdb_wo_stats_whole_plan_sum = sum(aqp_duckdb_wo_stats_whole_plan_data.values()) / 1000
        #
        # # analyze AQP breakdown time
        # aqp_duckdb_breakdown_data = plot_duckdb_end2end_results.analyze_duckdb_breakdown(aqp_duckdb_breakdown_log)
        # aqp_duckdb_execute_time = {query_name: metrics['execute'] + metrics['final_exe'] for query_name, metrics in
        #                            aqp_duckdb_breakdown_data.items()}
        # aqp_duckdb_exe_sum = sum(sub_dict['execute'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000 + \
        #                      sum(sub_dict['final_exe'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000
        # aqp_duckdb_opt_sum = sum(sub_dict['pre_opt'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000 + \
        #                      sum(sub_dict['post-opt'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000 + \
        #                      sum(sub_dict['final_post_opt'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000
        # aqp_duckdb_process_sum = sum(
        #     sub_dict['AQP-pre-process'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000 + \
        #                          sum(sub_dict['adapt-select'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000 + \
        #                          sum(sub_dict['AQP-post-process'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000
        # aqp_duckdb_breakdown = dict()
        # aqp_duckdb_breakdown['Execution'] = aqp_duckdb_exe_sum
        # aqp_duckdb_breakdown['AQP-Process'] = aqp_duckdb_process_sum
        # aqp_duckdb_breakdown['Optimization'] = aqp_duckdb_opt_sum
        # aqp_duckdb_breakdown['Analyze'] = 0
        # aqp_duckdb_breakdown['Other'] = aqp_duckdb_sum - aqp_duckdb_exe_sum - aqp_duckdb_process_sum - aqp_duckdb_opt_sum
        #
        # aqp_duckdb_wo_stats_breakdown_data = plot_duckdb_end2end_results.analyze_duckdb_breakdown(aqp_duckdb_wo_stats_breakdown_log)
        # aqp_duckdb_wo_stats_execute_time = {query_name: metrics['execute'] + metrics['final_exe'] for query_name, metrics in
        #                                     aqp_duckdb_wo_stats_breakdown_data.items()}
        # aqp_duckdb_wo_stats_exe_sum = sum(
        #     sub_dict['execute'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data.values()) / 1000 + \
        #                               sum(sub_dict['final_exe'] for sub_dict in
        #                                   aqp_duckdb_wo_stats_breakdown_data.values()) / 1000
        # aqp_duckdb_wo_stats_opt_sum = sum(
        #     sub_dict['pre_opt'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data.values()) / 1000 + \
        #                               sum(sub_dict['post-opt'] for sub_dict in
        #                                   aqp_duckdb_wo_stats_breakdown_data.values()) / 1000 + \
        #                               sum(sub_dict['final_post_opt'] for sub_dict in
        #                                   aqp_duckdb_wo_stats_breakdown_data.values()) / 1000
        # aqp_duckdb_wo_stats_process_sum = sum(
        #     sub_dict['AQP-pre-process'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data.values()) / 1000 + \
        #                                   sum(sub_dict['adapt-select'] for sub_dict in
        #                                       aqp_duckdb_wo_stats_breakdown_data.values()) / 1000 + \
        #                                   sum(sub_dict['AQP-post-process'] for sub_dict in
        #                                       aqp_duckdb_wo_stats_breakdown_data.values()) / 1000
        # aqp_duckdb_wo_stats_breakdown = dict()
        # aqp_duckdb_wo_stats_breakdown['Execution'] = aqp_duckdb_wo_stats_exe_sum
        # aqp_duckdb_wo_stats_breakdown['AQP-Process'] = aqp_duckdb_wo_stats_process_sum
        # aqp_duckdb_wo_stats_breakdown['Optimization'] = aqp_duckdb_wo_stats_opt_sum
        # aqp_duckdb_wo_stats_breakdown['Analyze'] = 0
        # aqp_duckdb_wo_stats_breakdown[
        #     'Other'] = aqp_duckdb_wo_stats_sum - aqp_duckdb_wo_stats_exe_sum - aqp_duckdb_wo_stats_process_sum - aqp_duckdb_wo_stats_opt_sum
        #
        # aqp_duckdb_wo_reorder_breakdown_data = plot_duckdb_end2end_results.analyze_duckdb_breakdown(aqp_duckdb_wo_reorder_breakdown_log)
        # aqp_duckdb_wo_reorder_execute_time = {query_name: metrics['execute'] + metrics['final_exe'] for query_name, metrics in
        #                                     aqp_duckdb_wo_reorder_breakdown_data.items()}
        # aqp_duckdb_wo_reorder_exe_sum = sum(
        #     sub_dict['execute'] for sub_dict in aqp_duckdb_wo_reorder_breakdown_data.values()) / 1000 + \
        #                               sum(sub_dict['final_exe'] for sub_dict in
        #                                   aqp_duckdb_wo_reorder_breakdown_data.values()) / 1000
        # aqp_duckdb_wo_reorder_opt_sum = sum(
        #     sub_dict['pre_opt'] for sub_dict in aqp_duckdb_wo_reorder_breakdown_data.values()) / 1000 + \
        #                               sum(sub_dict['post-opt'] for sub_dict in
        #                                   aqp_duckdb_wo_reorder_breakdown_data.values()) / 1000 + \
        #                               sum(sub_dict['final_post_opt'] for sub_dict in
        #                                   aqp_duckdb_wo_reorder_breakdown_data.values()) / 1000
        # aqp_duckdb_wo_reorder_process_sum = sum(
        #     sub_dict['AQP-pre-process'] for sub_dict in aqp_duckdb_wo_reorder_breakdown_data.values()) / 1000 + \
        #                                   sum(sub_dict['adapt-select'] for sub_dict in
        #                                       aqp_duckdb_wo_reorder_breakdown_data.values()) / 1000 + \
        #                                   sum(sub_dict['AQP-post-process'] for sub_dict in
        #                                       aqp_duckdb_wo_reorder_breakdown_data.values()) / 1000
        # aqp_duckdb_wo_reorder_breakdown = dict()
        # aqp_duckdb_wo_reorder_breakdown['Execution'] = aqp_duckdb_wo_reorder_exe_sum
        # aqp_duckdb_wo_reorder_breakdown['AQP-Process'] = aqp_duckdb_wo_reorder_process_sum
        # aqp_duckdb_wo_reorder_breakdown['Optimization'] = aqp_duckdb_wo_reorder_opt_sum
        # aqp_duckdb_wo_reorder_breakdown['Analyze'] = 0
        # aqp_duckdb_wo_reorder_breakdown[
        #     'Other'] = aqp_duckdb_wo_reorder_sum - aqp_duckdb_wo_reorder_exe_sum - aqp_duckdb_wo_reorder_process_sum - aqp_duckdb_wo_reorder_opt_sum

    # # fig for q1: vanilla VS AQP (with updating cardinality)
    # plot_end2end(vanilla_duckdb_sum, bar_colors[0], None,
    #              aqp_duckdb_sum, bar_colors[1], None,
    #              ["Vanilla", "Plan-based AQP"],
    #              [bar_colors[0], bar_colors[1]], "Total End-to-end Time (s)",
    #              f'{benchmark}_q1_fig_duckdb_only')
    #
    # plot_end2end_stacked(vanilla_duckdb_breakdown, None,
    #                      aqp_duckdb_breakdown, 'x',
    #                      ["Vanilla", "Plan-based AQP"],
    #                      ["lightgray", "lightgray"], "Total End-to-end Breakdown Time (s)",
    #                      f'{benchmark}_q1_fig_breakdown_duckdb_only')
    #
    # # # query by query compare
    # # vanilla_duckdb_data_stats = plot_query_by_query_box_chart_with_deviation(vanilla_duckdb_data, 'vanilla DuckDB')
    # # aqp_duckdb_data_stats = plot_query_by_query_box_chart_with_deviation(aqp_duckdb_data, 'AQP-DuckDB')
    # # plot_query_by_query_box_chart_compare_with_deviation(vanilla_duckdb_data_stats, aqp_duckdb_data_stats)
    # # plot_query_by_query(vanilla_duckdb_data, bar_colors[0], aqp_duckdb_data, bar_colors[1])
    # # aqp_duckdb_wo_stats_data_stats = plot_query_by_query_box_chart_with_deviation(aqp_duckdb_wo_stats_data,
    # #                                                                'AQP-DuckDB without updating statistics')
    #
    #
    # # fig for q2: w/wo updating statistics
    # plot_end2end(aqp_duckdb_wo_stats_sum, bar_colors[1], '/',
    #              aqp_duckdb_sum, bar_colors[1], None,
    #              ["Plan-based AQP w/o Updating Statistics", "Plan-based AQP with Updating Statistics"],
    #              [bar_colors[1], bar_colors[1]], "Total End-to-end Time (s)",
    #              f'{benchmark}_q2_fig_duckdb_only')
    # plot_end2end_stacked(aqp_duckdb_wo_stats_breakdown, '/',
    #                      aqp_duckdb_breakdown, 'x',
    #                      ["Plan-based AQP w/o Updating Cardinality", "Plan-based AQP with Updating Cardinality"],
    #                      ["lightgray", "lightgray"], "Total End-to-end Breakdown Time (s)",
    #                      f'{benchmark}_q2_fig_breakdown_duckdb_only')
    #
    # # fig for q3: w/wo specifying join order and operator (wo split and wo updating cardinality), only execution time
    # plot_end2end(vanilla_duckdb_breakdown['Execution'], bar_colors[0], None,
    #              aqp_duckdb_wo_stats_whole_plan_sum, bar_colors[0], 'o',
    #              ["Vanilla", "Vanilla with Specified Join Order and Operator"],
    #              [bar_colors[0], bar_colors[0]], "Execution Time (s)",
    #              f'{benchmark}_q3_fig_duckdb_only')
    #
    # # fig for q4: C5 VS C7: w/wo splitting the plan (with updating statistics), only execution time
    # plot_end2end(aqp_duckdb_breakdown['Execution'], bar_colors[1], 'x',
    #              aqp_duckdb_whole_plan_sum, bar_colors[1], '\\',
    #              ["Plan-based AQP with Splitting Plan", "Plan-based AQP w/o Splitting Plan"],
    #              [bar_colors[1], bar_colors[1]], "Execution Time (s)",
    #              f'{benchmark}_q4_fig_duckdb_only')

    # vanilla VS AQP without join reordering
    plot_end2end(vanilla_duckdb_sums, bar_colors[0],
                 aqp_duckdb_wo_reorder_sums, bar_colors[2],
                 benchmarks,
                 ["Vanilla DuckDB", "AQP-DuckDB"],
                 [bar_colors[0], bar_colors[2]], "Total End-to-end Time (s)",
                 'vanilla_vs_AQP_wo_reordering_join')

    # plot_end2end_stacked(vanilla_duckdb_breakdown, None,
    #                      aqp_duckdb_wo_reorder_breakdown, 'x',
    #                      ["Vanilla", "Plan-based AQP w/o reordering join"],
    #                      ["lightgray", "lightgray"], "Total End-to-end Breakdown Time (s)",
    #                      f'{benchmark}_q5_fig_breakdown_duckdb_only')

    compare_query_by_query_violin_grouped(vanilla_duckdb_datas, aqp_duckdb_wo_reorder_datas,
                                          ['#00FF00', '#03fc98', '#03BAFC'], benchmarks,
                                          title=f'vanilla_vs_AQP_wo_reordering_join')
