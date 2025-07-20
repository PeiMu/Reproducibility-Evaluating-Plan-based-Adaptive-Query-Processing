import csv
import re

import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import numpy as np
import sys

if not os.path.exists("../figures"):
    os.makedirs("../figures")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

benchmark = "job"
if len(sys.argv) > 1:
    benchmark = sys.argv[1]

query_num = 113 if benchmark == "job" else 58

# postgres end to end time
vanilla_pg_log = os.getcwd() + f"/{benchmark}_result_luigi/pg_Official.csv"
aqp_pg_log = os.getcwd() + f"/{benchmark}_result_luigi/pg_QuerySplit_with_stats.csv"
aqp_pg_wo_stats_log = os.getcwd() + f"/{benchmark}_result_luigi/pg_QuerySplit.csv"

# postgres breakdown time
vanilla_pg_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/pg_Official_breakdown_time_log.csv"
aqp_pg_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/QuerySplit_with_stats_breakdown_time_log.csv"
aqp_pg_wo_stats_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/QuerySplit_wo_stats_breakdown_time_log.csv"
aqp_pg_whole_plan_exe_time_log = os.getcwd() + f"/{benchmark}_result_luigi/QuerySplit_whole_plan_breakdown_time_log.csv"
aqp_pg_wo_stats_whole_plan_exe_time_log = os.getcwd() + f"/{benchmark}_result_luigi/QuerySplit_whole_plan_wo_stats_breakdown_time_log.csv"

pg_log = os.getcwd() + f"/DB_CompetitorPerformance.xlsx"

strategies = ["Vanilla", "Vanilla with Specified Join Order and Operator",
              "Plan-based AQP", "Plan-based AQP without Updating Cardinality",
              "Plan-based AQP without Splitting Plan"]
bar_colors = [
    '#165DC7',  # dark blue, vanilla postgres
    '#03BAFC',  # bright blue, AQP postgres
]


# bar_colors = [
#     "darkgray",
#     "lightgray"
# ]

common_queries = []
if benchmark == "job":
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


def analyze_csv_data(csv_file):
    data = pd.read_csv(csv_file, skiprows=lambda x: x > 0 and x % 2 == 0)

    # Extract the SQL name from the command column
    if benchmark == "job":
        data['sql_name'] = data['command'].str.extract(r'/([0-9a-z]+)\.sql')
    else:
        data['instance'] = data['command'].str.extract(r'/1_instance_out_wo_multi_block/(\d+)/query')
        data['sub_sql_name'] = data['command'].str.extract(r'/query([^/]+)/[^/]+\.sql')
        data['sql_name'] = data['instance'] + '_' + data['sub_sql_name']

    # Convert to float type for calculation
    columns = ['mean', 'stddev', 'median', 'user', 'system', 'min', 'max']
    data[columns] = data[columns].astype(float)
    return data


def analyze_vanilla_breakdown(csv_file):
    results = {}
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip the header

        for row in reader:
            if row and row[0].startswith("execute"):
                command = row[0].strip()
                if benchmark == "job":
                    match = re.search(r'/([0-9a-z]+)\.sql', command)
                    if not match:
                        raise ValueError(f"Pattern not found in command: {command}")
                    sql_name = match.group(1)
                else:
                    instance_match = re.search(r'/(\d+)/query', command)
                    if not instance_match:
                        raise ValueError(f"Instance pattern not found in command: {command}")
                    instance = instance_match.group(1)
                    sub_sql_name_match = re.search(r'/query([^/]+)/[^/]+\.sql', command)
                    if not sub_sql_name_match:
                        raise ValueError(f"Sub SQL name pattern not found in command: {command}")
                    sub_sql_name = sub_sql_name_match.group(1)
                    sql_name = instance + '_' + sub_sql_name

                perf_rows = []
                for _ in range(15):
                    try:
                        perf_row = next(reader)
                        values = [float(item.strip()) for item in perf_row]
                        perf_rows.append(values)
                    except StopIteration:
                        break

                assert len(perf_rows) == 15
                valid_rows = perf_rows[5:]
                averages = {}
                # Calculate average for each column
                for i, col_name in enumerate(header):
                    col_values = [row[i] for row in valid_rows]
                    avg = sum(col_values) / len(col_values)
                    averages[col_name] = avg
                results[sql_name] = averages
        return results


def analyze_whole_plan_csv_data(csv_file):
    results = {}
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)

        # 5 warm up + 10 execution
        for row in reader:
            if row and row[0].startswith("execute"):
                command = row[0].strip()
                if benchmark == "job":
                    match = re.search(r'/([0-9a-z]+)\.sql', command)
                    if not match:
                        raise ValueError(f"Pattern not found in command: {command}")
                    sql_name = match.group(1)
                else:
                    instance_match = re.search(r'/(\d+)/query', command)
                    if not instance_match:
                        raise ValueError(f"Instance pattern not found in command: {command}")
                    instance = instance_match.group(1)
                    sub_sql_name_match = re.search(r'/query([^/]+)/[^/]+\.sql', command)
                    if not sub_sql_name_match:
                        raise ValueError(f"Sub SQL name pattern not found in command: {command}")
                    sub_sql_name = sub_sql_name_match.group(1)
                    sql_name = instance + '_' + sub_sql_name

                perf_values = []
                for _ in range(15):
                    try:
                        perf_row = next(reader)
                        value = float(perf_row[-2].strip() if perf_row[-1].strip() == '' else perf_row[-1].strip())
                        perf_values.append(value)
                    except StopIteration:
                        break

                assert len(perf_values) == 15
                mean_value = sum(perf_values[5:]) / len(perf_values[5:])
                results[sql_name] = mean_value
    return results


# analyze AQP-PG breakdown time: pre-opt, [opt, execute, (analyze), AQP-post-process], opt, execute
def analyze_pg_breakdown(csv_file, have_analyze):
    results = {}
    group_columns = []
    if have_analyze:
        group_columns = ["opt", "execute", "analyze", "AQP-post-process"]
    else:
        group_columns = ["opt", "execute", "AQP-post-process"]

    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)

        for row in reader:
            if row and row[0].startswith("execute"):
                command = row[0].strip()
                if benchmark == "job":
                    match = re.search(r'/([0-9a-z]+)\.sql', command)
                    if not match:
                        raise ValueError(f"Pattern not found in command: {command}")
                    sql_name = match.group(1)
                else:
                    instance_match = re.search(r'/(\d+)/query', command)
                    if not instance_match:
                        raise ValueError(f"Instance pattern not found in command: {command}")
                    instance = instance_match.group(1)
                    sub_sql_name_match = re.search(r'/query([^/]+)/[^/]+\.sql', command)
                    if not sub_sql_name_match:
                        raise ValueError(f"Sub SQL name pattern not found in command: {command}")
                    sub_sql_name = sub_sql_name_match.group(1)
                    sql_name = instance + '_' + sub_sql_name

                perf_rows = []
                for _ in range(15):
                    try:
                        perf_row = next(reader)
                        pre_opt_time = float(perf_row[0].strip())
                        final_opt_time = float(perf_row[-3].strip())
                        final_exe_time = float(perf_row[-2].strip())
                        group_values = perf_row[1:-3]
                        groups = []
                        assert len(group_values) % len(group_columns) == 0
                        num_groups = len(group_values) // len(group_columns)
                        for i in range(num_groups):
                            group = {}
                            for j, col_name in enumerate(group_columns):
                                # Compute the index within group_values
                                index = i * len(group_columns) + j
                                group[col_name] = float(group_values[index].strip())
                            groups.append(group)

                        # Save all extracted data from the row in a dictionary.
                        row_data = {
                            "pre_opt": pre_opt_time,
                            "groups": groups,
                            "final_opt": final_opt_time,
                            "final_exe": final_exe_time
                        }
                        perf_rows.append(row_data)
                    except StopIteration:
                        break

                assert len(perf_rows) == 15
                valid_rows = perf_rows[5:]
                averages = {}
                # calculate the average of each column
                avg_pre_opt = sum(r["pre_opt"] for r in valid_rows) / len(valid_rows)

                group_sums = {col: 0.0 for col in group_columns}
                group_counts = {col: 0 for col in group_columns}
                for r in valid_rows:
                    for grp in r["groups"]:
                        for col in group_columns:
                            group_sums[col] += grp[col]
                avg_groups = {col: (group_sums[col] / len(valid_rows)) for col in group_columns}

                avg_final_opt = sum(r["final_opt"] for r in valid_rows) / len(valid_rows)
                avg_final_exe = sum(r["final_exe"] for r in valid_rows) / len(valid_rows)

                averages["pre_opt"] = avg_pre_opt
                averages["opt"] = avg_groups["opt"]
                averages["execute"] = avg_groups["execute"]
                if have_analyze:
                    averages["analyze"] = avg_groups["analyze"]
                averages["AQP-post-process"] = avg_groups["AQP-post-process"]
                averages["final_opt"] = avg_final_opt
                averages["final_exe"] = avg_final_exe
                results[sql_name] = averages

    return results


# temp impl. todo: use `analyze_csv_data`
def analyze_pg_excel_data(excel_file):
    data = pd.read_excel(excel_file, sheet_name="QuerySplit_reproduce_analyze", usecols="A,C,E,M",
                         names=['query_id', strategies[3], strategies[0], strategies[2]], skiprows=1, nrows=64,
                         header=None)
    vanilla_pg = data[strategies[0]]
    aqp_pg_wo_stat = data[strategies[3]]
    aqp_pg = data[strategies[2]]
    return vanilla_pg, aqp_pg_wo_stat, aqp_pg


def plot_end2end(col1_data, col1_color, col1_pattern,
                 col2_data, col2_color, col2_pattern,
                 legend_names, legend_colors, title,
                 name):
    col1_sum = col1_data
    col2_sum = col2_data

    values = [col1_sum, col2_sum]
    colors = [col1_color, col2_color]
    patterns = [col1_pattern, col2_pattern]

    x_positions = [0, 0.5]

    fig, ax = plt.subplots()

    bars = []
    for i in range(len(values)):
        bar = ax.bar(x_positions[i], values[i], width=0.4, color=colors[i], edgecolor='black')
        if patterns[i]:
            bar[0].set_hatch(patterns[i])
        bars.append(bar)

    # Add labels and formatting
    ax.set_xticks([0.25])  # Center labels for groups
    ax.set_xticklabels(['PostgreSQL'])

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
    y_speedup = y2 + ax.get_ylim()[1] / 10  # Slightly above the higher bar

    if swap_flag:
        # Compute speedup
        speedup = y2 / y1
        # Draw speedup annotation in red
        ax.text((x1 + x2) / 2, y_speedup, f'{speedup:.2f}x ↑',
                ha='center', va='bottom', fontsize=20, fontweight='bold', color='red')
    else:
        # # Compute slowdown
        # slowdown = y2 / y1 - 1
        # # Draw speedup annotation in red
        # ax.text((x1 + x2) / 2, y_speedup, f'Slowdown: {slowdown * 100:.2f}%',
        #         ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
        # Compute slowdown
        slowdown = y1 / y2
        # Draw speedup annotation in red
        ax.text((x1 + x2) / 2, y_speedup, f'{slowdown:.2f}x ↓',
                ha='center', va='bottom', fontsize=20, fontweight='bold', color='red')

    # Add a legend
    # legend_patches = [Patch(facecolor=colors[i], edgecolor='black', hatch=patterns[i], label=legend_names[i])
    #                   for i in range(len(legend_names))]
    legend_patches = [Patch(facecolor=legend_colors[0], edgecolor='black', hatch=patterns[0], label=legend_names[0]),
                      Patch(facecolor=legend_colors[1], edgecolor='black', hatch=patterns[1], label=legend_names[1])]
    ax.legend(handles=legend_patches, loc='upper left', handleheight=3, handlelength=4)

    plt.ylabel(title)
    for i, value in enumerate(values):
        plt.text(x_positions[i], value + max(values) * 0.01, f'{value:.2f}', ha='center', fontsize=10)
    plt.ylim(0, max(values) * 2)
    plt.tight_layout()
    # plt.name("Performance Comparison of JOB")
    # plt.show()
    plt.savefig('../figures/' + name + '.pdf')
    plt.clf()
    plt.close()


def plot_end2end_stacked(col1_data, col1_pattern,
                         col2_data, col2_pattern,
                         legend_names, legend_colors, title,
                         name):
    values = [sum(col1_data.values()), sum(col2_data.values())]
    colors = ['lightgreen', '#fcdb00', '#e695f5', '#ff954a', 'lightblue']
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
            ax.bar(x_positions[i], execution_time[i], width=0.4, color=colors[4], edgecolor='black', label=labels[4]))
        bar_sections.append(
            ax.bar(x_positions[i], aqp_process_time[i], bottom=execution_time[i], width=0.4, color=colors[3],
                   edgecolor='black', label=labels[3]))
        bar_sections.append(
            ax.bar(x_positions[i], optimization_time[i], bottom=execution_time[i] + aqp_process_time[i], width=0.4,
                   color=colors[2], edgecolor='black', label=labels[2]))
        bar_sections.append(
            ax.bar(x_positions[i], analyze_time[i],
                   bottom=optimization_time[i] + execution_time[i] + aqp_process_time[i],
                   width=0.4, color=colors[1], edgecolor='black', label=labels[1]))
        bar_sections.append(
            ax.bar(x_positions[i], other_time[i],
                   bottom=analyze_time[i] + optimization_time[i] + execution_time[i] + aqp_process_time[i],
                   width=0.4, color=colors[0], edgecolor='black', label=labels[0]))
        for section in bar_sections:
            section[0].set_hatch(patterns[i])
        bars.append(bar_sections)

    # Add labels and formatting
    ax.set_xticks([0.25])  # Center labels for groups
    ax.set_xticklabels(['PostgreSQL'])

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

    plt.ylabel(title)
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

    # width = 0.3
    # plt.bar(x - width / 2, col1_data['mean'], width, label=strategies[0], color=col1_color)
    # plt.bar(x + width / 2, col2_data['mean'], width, label=strategies[1], color=col2_color)
    # plt.xlabel("Query ID")
    # plt.ylabel("Query by query time (ms)")
    # plt.legend()
    # plt.xticks(x, labels=col1_data['sql_name'], rotation=90)
    # plt.xlim(-0.5, len(col1_data) - 0.5)
    # max_y = max(max(col1_data['mean']), max(col2_data['mean']))
    # plt.ylim(0, max_y * 1.01)

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


def plot_query_by_query_box_chart_compare(official_stats, rsj_stats):
    # Create positions for the two sets
    positions1 = np.arange(1, 113 + 1)  # Positions for Set 1
    positions2 = positions1 + 0.4  # Offset for Set 2

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(20, 8))  # Adjust the size for 113 categories

    # Plot the first set of boxplots
    ax.bxp(official_stats, positions=positions1, showmeans=False, showfliers=False, widths=0.3, patch_artist=True,
           boxprops=dict(facecolor="yellow", edgecolor="black"))

    # Plot the second set of boxplots
    ax.bxp(rsj_stats, positions=positions2, showmeans=False, showfliers=False, widths=0.3, patch_artist=True,
           boxprops=dict(facecolor="blue", edgecolor="black"))

    # Customize the x-axis
    ax.set_xticks((positions1 + positions2) / 2)  # Set ticks between the two sets
    ax.set_xticklabels([f'Category {i}' for i in range(1, 113 + 1)], rotation=90)  # Rotate for readability

    # Add labels and title
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Two Sets Across 113 Categories')

    # Add a legend
    ax.legend(['Set 1', 'Set 2'], loc='upper left')

    # Tight layout to handle label spacing
    plt.tight_layout()

    # Show the plot
    # plt.show()
    plt.savefig('../figures/compare_with_deviation.pdf')
    plt.clf()
    plt.close()


if __name__ == "__main__":
    ### prepare postgres data ###
    vanilla_pg_data = analyze_csv_data(vanilla_pg_log)
    aqp_pg_data = analyze_csv_data(aqp_pg_log)
    aqp_pg_wo_stats_data = analyze_csv_data(aqp_pg_wo_stats_log)

    vanilla_pg_sum = sum(vanilla_pg_data['mean'])
    aqp_pg_sum = sum(aqp_pg_data['mean'])
    aqp_pg_wo_stats_sum = sum(aqp_pg_wo_stats_data['mean'])

    # analyze vanilla breakdown time
    vanilla_pg_breakdown_data = analyze_vanilla_breakdown(vanilla_pg_breakdown_log)
    vanilla_pg_exe_sum = sum(sub_dict[' Execute'] for sub_dict in vanilla_pg_breakdown_data.values())
    vanilla_opt_sum = sum(sub_dict['Optimize'] for sub_dict in vanilla_pg_breakdown_data.values())
    # we only need 1. execution time, 2. AQP-process time, 3. optimization time, and 4. other time (including parse, etc.)
    vanilla_pg_breakdown = dict()
    vanilla_pg_breakdown['Execution'] = vanilla_pg_exe_sum
    vanilla_pg_breakdown['AQP-Process'] = 0
    vanilla_pg_breakdown['Optimization'] = vanilla_opt_sum
    vanilla_pg_breakdown['Analyze'] = 0
    vanilla_pg_breakdown['Other'] = vanilla_pg_sum - vanilla_pg_exe_sum - vanilla_opt_sum

    # analyze whole plan execution time
    aqp_pg_whole_plan_data = analyze_whole_plan_csv_data(aqp_pg_whole_plan_exe_time_log)
    aqp_pg_wo_stats_whole_plan_data = analyze_whole_plan_csv_data(aqp_pg_wo_stats_whole_plan_exe_time_log)

    aqp_pg_whole_plan_sum = sum(aqp_pg_whole_plan_data.values())
    aqp_pg_wo_stats_whole_plan_sum = sum(aqp_pg_wo_stats_whole_plan_data.values())

    # analyze AQP breakdown time
    aqp_pg_breakdown_data = analyze_pg_breakdown(aqp_pg_breakdown_log, True)
    aqp_pg_exe_sum = sum(sub_dict['execute'] for sub_dict in aqp_pg_breakdown_data.values()) + \
                     sum(sub_dict['final_exe'] for sub_dict in aqp_pg_breakdown_data.values())
    aqp_pg_opt_sum = sum(sub_dict['pre_opt'] for sub_dict in aqp_pg_breakdown_data.values()) + \
                     sum(sub_dict['opt'] for sub_dict in aqp_pg_breakdown_data.values()) + \
                     sum(sub_dict['final_opt'] for sub_dict in aqp_pg_breakdown_data.values())
    aqp_pg_process_sum = sum(sub_dict['AQP-post-process'] for sub_dict in aqp_pg_breakdown_data.values())
    aqp_pg_analyze_sum = sum(sub_dict['analyze'] for sub_dict in aqp_pg_breakdown_data.values())
    aqp_pg_breakdown = dict()
    aqp_pg_breakdown['Execution'] = aqp_pg_exe_sum
    aqp_pg_breakdown['AQP-Process'] = aqp_pg_process_sum
    aqp_pg_breakdown['Optimization'] = aqp_pg_opt_sum
    aqp_pg_breakdown['Analyze'] = aqp_pg_analyze_sum
    aqp_pg_breakdown['Other'] = aqp_pg_sum - aqp_pg_exe_sum - aqp_pg_process_sum - aqp_pg_opt_sum - aqp_pg_analyze_sum

    aqp_pg_wo_stats_breakdown_data = analyze_pg_breakdown(aqp_pg_wo_stats_breakdown_log, False)
    aqp_pg_wo_stats_exe_sum = sum(sub_dict['execute'] for sub_dict in aqp_pg_wo_stats_breakdown_data.values()) + \
                              sum(sub_dict['final_exe'] for sub_dict in aqp_pg_wo_stats_breakdown_data.values())
    aqp_pg_wo_stats_opt_sum = sum(sub_dict['pre_opt'] for sub_dict in aqp_pg_wo_stats_breakdown_data.values()) + \
                              sum(sub_dict['opt'] for sub_dict in aqp_pg_wo_stats_breakdown_data.values()) + \
                              sum(sub_dict['final_opt'] for sub_dict in aqp_pg_wo_stats_breakdown_data.values())
    aqp_pg_wo_stats_process_sum = sum(
        sub_dict['AQP-post-process'] for sub_dict in aqp_pg_wo_stats_breakdown_data.values())
    aqp_pg_wo_stats_breakdown = dict()
    aqp_pg_wo_stats_breakdown['Execution'] = aqp_pg_wo_stats_exe_sum
    aqp_pg_wo_stats_breakdown['AQP-Process'] = aqp_pg_wo_stats_process_sum
    aqp_pg_wo_stats_breakdown['Optimization'] = aqp_pg_wo_stats_opt_sum
    aqp_pg_wo_stats_breakdown['Analyze'] = 0
    aqp_pg_wo_stats_breakdown[
        'Other'] = aqp_pg_wo_stats_sum - aqp_pg_wo_stats_exe_sum - aqp_pg_wo_stats_process_sum - aqp_pg_wo_stats_opt_sum

    # (vanilla_duckdb, vanilla_duckdb_specify_join,
    #  aqp_duckdb, aqp_duckdb_wo_stat, apq_duckdb_wo_split) = analyze_duckdb_excel_data(duckdb_log)
    # # vanilla_duckdb_sum = sum(vanilla_duckdb) / 1000
    # vanilla_duckdb_specify_join_sum = sum(vanilla_duckdb_specify_join) / 1000
    # # aqp_duckdb_sum = sum(aqp_duckdb) / 1000
    # # aqp_duckdb_wo_stats_sum = sum(aqp_duckdb_wo_stat) / 1000
    # apq_duckdb_wo_split_sum = sum(apq_duckdb_wo_split) / 1000
    #
    # vanilla_pg, aqp_pg_wo_stat, aqp_pg = analyze_pg_excel_data(pg_log)
    # vanilla_pg_sum = sum(vanilla_pg) / 1000
    # aqp_pg_wo_stat_sum = sum(aqp_pg_wo_stat) / 1000
    # aqp_pg_sum = sum(aqp_pg) / 1000

    # fig for q1: vanilla VS AQP (with updating Cardinality)
    plot_end2end(vanilla_pg_sum, bar_colors[0], None,
                 aqp_pg_sum, bar_colors[1], None,
                 ["Vanilla", "Plan-based AQP"],
                 [bar_colors[0], bar_colors[1]], "Total End-to-end Time (s)",
                 'q1_fig')

    plot_end2end_stacked(vanilla_pg_breakdown, None,
                         aqp_pg_breakdown, 'x',
                         ["Vanilla", "Plan-based AQP"],
                         ["lightgray", "lightgray"], "Total End-to-end Breakdown Time (s)",
                         'q1_fig_breakdown')

    vanilla_pg_data_stats = plot_query_by_query_box_chart_with_deviation(vanilla_pg_data, 'vanilla PostgreSQL')
    aqp_pg_data_stats = plot_query_by_query_box_chart_with_deviation(aqp_pg_data, 'AQP-PostgreSQL')
    aqp_pg_wo_stats_data_stats = plot_query_by_query_box_chart_with_deviation(aqp_pg_wo_stats_data,
                                                                   'AQP-PostgreSQL without updating Cardinality')

    # fig for q2: w/wo updating Cardinality
    plot_end2end(aqp_pg_wo_stats_sum, bar_colors[1], '/',
                 aqp_pg_sum, bar_colors[1], None,
                 ["Plan-based AQP w/o Updating Cardinality", "Plan-based AQP with Updating Cardinality"],
                 [bar_colors[1], bar_colors[1]], "Total End-to-end Time (s)",
                 'q2_fig')
    plot_end2end_stacked(aqp_pg_wo_stats_breakdown, '/',
                         aqp_pg_breakdown, 'x',
                         ["Plan-based AQP w/o Updating Cardinality", "Plan-based AQP with Updating Cardinality"],
                         ["lightgray", "lightgray"], "Total End-to-end Breakdown Time (s)",
                         'q2_fig_breakdown')

    # fig for q3: w/wo specifying join order and operator (wo split and wo updating Cardinality), only execution time
    plot_end2end(vanilla_pg_breakdown['Execution'], bar_colors[0], None,
                 aqp_pg_wo_stats_whole_plan_sum, bar_colors[0], 'o',
                 ["Vanilla", "Vanilla with Specified Join Order and Operator"],
                 [bar_colors[0], bar_colors[0]], "Execution Time (s)",
                 'q3_fig')

    # fig for q4: C5 VS C7: w/wo splitting the plan (with updating Cardinality), only execution time
    plot_end2end(aqp_pg_breakdown['Execution'], bar_colors[1], 'x',
                 aqp_pg_whole_plan_sum, bar_colors[1], '\\',
                 ["Plan-based AQP with Splitting Plan", "Plan-based AQP w/o Splitting Plan"],
                 [bar_colors[1], bar_colors[1]], "Execution Time (s)",
                 'q4_fig')
