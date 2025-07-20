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
import seaborn as sns
import math
from math import e


if not os.path.exists("../figures"):
    os.makedirs("../figures")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

benchmark = "DSB_10"
if len(sys.argv) > 1:
    benchmark = sys.argv[1]

query_num = 113 if benchmark == "JOB" else 58

# duckdb end to end time
vanilla_duckdb_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_official_nan.csv"
aqp_duckdb_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_query_split_rsj.csv"
aqp_duckdb_wo_stats_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_query_split_rsj_wo_stats_stats.csv"

# duckdb breakdown time
vanilla_duckdb_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_official_breakdown_time_log.csv"
aqp_duckdb_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_rsj_breakdown_time_log.csv"
aqp_duckdb_wo_stats_breakdown_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_rsj_wo_stats_breakdown_time_log.csv"
aqp_duckdb_whole_plan_exe_time_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_rsj_whole_plan_breakdown_time_log.csv"
aqp_duckdb_wo_stats_whole_plan_exe_time_log = os.getcwd() + f"/{benchmark}_result_luigi/duckdb_rsj_whole_plan_wo_stats_breakdown_time_log.csv"

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

duckdb_log = os.getcwd() + f"/DBShaker_performance.xlsx"
pg_log = os.getcwd() + f"/DB_CompetitorPerformance.xlsx"

strategies = ["Vanilla", "Vanilla with Specified Join Order and Operator",
              "Plan-based AQP", "Plan-based AQP without Updating Cardinality",
              "Plan-based AQP without Splitting Plan"]
bar_colors = [
    '#006400',  # dark green, vanilla duckdb
    '#00FF00',  # bright green, AQP duckdb
    '#165DC7',  # dark blue, vanilla postgres
    '#03BAFC',  # bright blue, AQP postgres
    '#4F3E73',  # dark turquoise, vanilla common duckdb
    '#B4F4F7',  # bright turquoise, AQP common duckdb
]

# bar_colors = [
#     "darkgray",
#     "lightgray",
#     "darkgray",
#     "lightgray"
# ]

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


def analyze_csv_data(csv_file, system, current_benchmark=benchmark):
    data = pd.read_csv(csv_file, skiprows=lambda x: x > 0 and x % 2 == 0)

    # Extract the SQL name from the command column
    if current_benchmark == "JOB":
        data['sql_name'] = data['command'].str.extract(r'/([0-9a-z]+)\.sql')
    else:
        if system == "duckdb":
            data['instance'] = data['command'].str.extract(r'/1_instance_out_wo_multi_block/(\d+)/query')
        else:
            data['instance'] = data['command'].str.extract(r'/1_instance_out_qs_[^/]+/(\d+)/query')
        data['sub_sql_name'] = data['command'].str.extract(r'/query([^/]+)/[^/]+\.sql')
        data['sql_name'] = data['instance'] + '_' + data['sub_sql_name']

    # Convert to float type for calculation
    columns = ['mean', 'stddev', 'median', 'user', 'system', 'min', 'max']
    data[columns] = data[columns].astype(float)
    return data


def analyze_vanilla_breakdown(csv_file, current_benchmark=benchmark):
    results = {}
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip the header

        for row in reader:
            if row and row[0].startswith("execute"):
                command = row[0].strip()
                if current_benchmark == "JOB":
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


def analyze_whole_plan_csv_data(csv_file, current_benchmark=benchmark):
    results = {}
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)

        # 5 warm up + 10 execution
        for row in reader:
            if row and row[0].startswith("execute"):
                command = row[0].strip()
                if current_benchmark == "JOB":
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


# analyze AQP-DuckDB breakdown time: pre-opt, [AQP-pre-process, post-opt, adapt-select, create-plan, execute, AQP-post-process], final-post-opt, final-create-plan, final-exe
def analyze_duckdb_breakdown(csv_file, current_benchmark=benchmark):
    results = {}
    group_columns = ["AQP-pre-process", "post-opt", "adapt-select", "create-plan", "execute", "AQP-post-process"]

    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)

        for row in reader:
            if row and row[0].startswith("execute"):
                command = row[0].strip()
                if current_benchmark == "JOB":
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
                        final_post_opt_time = float(perf_row[-3].strip())
                        final_create_plan_time = float(perf_row[-2].strip())
                        final_exe_time = float(perf_row[-1].strip())
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
                            "final_post_opt": final_post_opt_time,
                            "final_create_plan": final_create_plan_time,
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

                avg_final_post_opt = sum(r["final_post_opt"] for r in valid_rows) / len(valid_rows)
                avg_final_create_plan = sum(r["final_create_plan"] for r in valid_rows) / len(valid_rows)
                avg_final_exe = sum(r["final_exe"] for r in valid_rows) / len(valid_rows)

                averages["pre_opt"] = avg_pre_opt
                averages["AQP-pre-process"] = avg_groups["AQP-pre-process"]
                averages["post-opt"] = avg_groups["post-opt"]
                averages["adapt-select"] = avg_groups["adapt-select"]
                averages["create-plan"] = avg_groups["create-plan"]
                averages["execute"] = avg_groups["execute"]
                averages["AQP-post-process"] = avg_groups["AQP-post-process"]
                averages["final_post_opt"] = avg_final_post_opt
                averages["final_create_plan"] = avg_final_create_plan
                averages["final_exe"] = avg_final_exe
                results[sql_name] = averages

    return results


# analyze AQP-PG breakdown time: pre-opt, [opt, execute, (analyze), AQP-post-process], opt, execute
def analyze_pg_breakdown(csv_file, have_analyze, current_benchmark=benchmark):
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
                if current_benchmark == "JOB":
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


# temp impl. todo: use `analyze_csv_data`
def analyze_duckdb_excel_data(excel_file):
    path = os.getcwd() + f"/DBShaker_performance.xlsx"
    data = pd.read_excel(path, sheet_name="luigi performance", usecols="T,U,AG,AM,AO,AQ",
                         names=['query_id', strategies[0], strategies[2], strategies[3], strategies[4], strategies[1]],
                         skiprows=3, nrows=113, header=None)
    vanilla_duckdb = data[strategies[0]]
    vanilla_duckdb_specify_join = data[strategies[1]]
    aqp_duckdb = data[strategies[2]]
    aqp_duckdb_wo_stat = data[strategies[3]]
    apq_duckdb_wo_split = data[strategies[4]]

    return (vanilla_duckdb, vanilla_duckdb_specify_join,
            aqp_duckdb, aqp_duckdb_wo_stat, apq_duckdb_wo_split)


def plot_end2end(col1_data, col1_color, col1_pattern,
                 col2_data, col2_color, col2_pattern,
                 col3_data, col3_color, col3_pattern,
                 col4_data, col4_color, col4_pattern,
                 legend_names, legend_colors, title,
                 name):
    col1_sum = col1_data
    col2_sum = col2_data
    col3_sum = col3_data
    col4_sum = col4_data

    values = [col1_sum, col2_sum, col3_sum, col4_sum]
    colors = [col1_color, col2_color, col3_color, col4_color]
    patterns = [col1_pattern, col2_pattern, col3_pattern, col4_pattern]

    x_positions = [0, 0.5, 2, 2.5]

    fig, ax = plt.subplots()

    bars = []
    for i in range(len(values)):
        bar = ax.bar(x_positions[i], values[i], width=0.4, color=colors[i], edgecolor='black')
        if patterns[i]:
            bar[0].set_hatch(patterns[i])
        bars.append(bar)

    # Add labels and formatting
    ax.set_xticks([0.25, 2.25])  # Center labels for groups
    ax.set_xticklabels(['DuckDB', 'PostgreSQL'])

    # Add vertical dotted lines, horizontal solid lines, and speedup representation
    for i in range(2):  # Two groups (Group 1 and Group 2)
        x1, x2 = x_positions[i * 2], x_positions[i * 2 + 1]  # Get x positions of bars in the group
        y1, y2 = values[i * 2], values[i * 2 + 1]  # Get heights of bars in the group

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
    ax.legend(handles=legend_patches, loc='upper left', handleheight=4, handlelength=6)

    plt.ylabel(title, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10, fontweight='bold')
    for i, value in enumerate(values):
        plt.text(x_positions[i], value + max(values) * 0.01, f'{value:.2f}', ha='center', fontsize=10)
    plt.ylim(0, max(values) * 2)
    plt.tight_layout()
    # plt.name("Performance Comparison of {benchmark}")
    # plt.show()
    plt.savefig('../figures/' + name + '.pdf')
    plt.clf()
    plt.close()


def plot_end2end_common(col1_data, col1_color, col1_pattern,
                        col2_data, col2_color, col2_pattern,
                        col3_data, col3_color, col3_pattern,
                        col4_data, col4_color, col4_pattern,
                        col5_data, col5_color, col5_pattern,
                        col6_data, col6_color, col6_pattern,
                        legend_names, legend_colors, title,
                        name):
    col1_sum = col1_data
    col2_sum = col2_data
    col3_sum = col3_data
    col4_sum = col4_data
    col5_sum = col5_data
    col6_sum = col6_data

    values = [col1_sum, col2_sum, col3_sum, col4_sum, col5_sum, col6_sum]
    colors = [col1_color, col2_color, col3_color, col4_color, col5_color, col6_color]
    patterns = [col1_pattern, col2_pattern, col3_pattern, col4_pattern, col5_pattern, col6_pattern]

    x_positions = [0, 0.5, 2, 2.5, 4, 4.5]

    fig, ax = plt.subplots()

    bars = []
    for i in range(len(values)):
        bar = ax.bar(x_positions[i], values[i], width=0.4, color=colors[i], edgecolor='black')
        if patterns[i]:
            bar[0].set_hatch(patterns[i])
        bars.append(bar)

    # Add labels and formatting
    ax.set_xticks([0.25, 2.25, 4.25])  # Center labels for groups
    ax.set_xticklabels(['DuckDB', 'DuckDB (common)', 'PostgreSQL (common)'])

    # Add vertical-dotted lines, horizontal solid lines, and speedup representation
    for i in range(3):  # Two groups (Group 1 and Group 2)
        x1, x2 = x_positions[i * 2], x_positions[i * 2 + 1]  # Get x positions of bars in the group
        y1, y2 = values[i * 2], values[i * 2 + 1]  # Get heights of bars in the group

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
            # Compute slowdown
            slowdown = y1 / y2
            # Draw speedup annotation in red
            ax.text((x1 + x2) / 2, y_speedup, f'{slowdown:.2f}x ↓',
                    ha='center', va='bottom', fontsize=20, fontweight='bold', color='red')

    # Add a legend
    legend_patches = [Patch(facecolor=legend_colors[0], edgecolor='black', hatch=patterns[0], label=legend_names[0]),
                      Patch(facecolor=legend_colors[1], edgecolor='black', hatch=patterns[1], label=legend_names[1])]
    ax.legend(handles=legend_patches, loc='upper left', handleheight=4, handlelength=6)

    plt.ylabel(title, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=11, fontweight='bold')
    for i, value in enumerate(values):
        plt.text(x_positions[i], value + max(values) * 0.01, f'{value:.2f}', ha='center', fontsize=10)
    plt.ylim(0, max(values) * 2)
    plt.tight_layout()
    # plt.name("Performance Comparison of {benchmark}")
    # plt.show()
    plt.savefig('../figures/' + name + '.pdf')
    plt.clf()
    plt.close()


def plot_end2end_stacked(col1_data, col1_pattern,
                         col2_data, col2_pattern,
                         col3_data, col3_pattern,
                         col4_data, col4_pattern,
                         legend_names, legend_colors, title,
                         name):
    values = [sum(col1_data.values()), sum(col2_data.values()), sum(col3_data.values()), sum(col4_data.values())]
    colors = ['lightgreen', '#fcdb00', '#e695f5', '#ff954a', 'lightblue']
    patterns = [col1_pattern, col2_pattern, col3_pattern, col4_pattern]
    labels = ["Other", "Analyze", "Optimization", "AQP-Process", "Execution"]

    other_time = {0: col1_data["Other"], 1: col2_data["Other"], 2: col3_data["Other"], 3: col4_data["Other"]}
    analyze_time = {0: col1_data["Analyze"], 1: col2_data["Analyze"], 2: col3_data["Analyze"], 3: col4_data["Analyze"]}
    optimization_time = {0: col1_data["Optimization"], 1: col2_data["Optimization"], 2: col3_data["Optimization"],
                         3: col4_data["Optimization"]}
    aqp_process_time = {0: col1_data["AQP-Process"], 1: col2_data["AQP-Process"], 2: col3_data["AQP-Process"],
                        3: col4_data["AQP-Process"]}
    execution_time = {0: col1_data["Execution"], 1: col2_data["Execution"], 2: col3_data["Execution"],
                      3: col4_data["Execution"]}

    x_positions = [0, 0.5, 2, 2.5]

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
    ax.set_xticks([0.25, 2.25])  # Center labels for groups
    ax.set_xticklabels(['DuckDB', 'PostgreSQL'])

    # Add vertical dotted lines, horizontal solid lines, and speedup representation
    for i in range(2):  # Two groups (Group 1 and Group 2)
        x1, x2 = x_positions[i * 2], x_positions[i * 2 + 1]  # Get x positions of bars in the group
        y1, y2 = values[i * 2], values[i * 2 + 1]  # Get heights of bars in the group
        y1_exe, y2_exe = execution_time[i * 2], execution_time[i * 2 + 1]

        # Ensure y1 is the smaller value and y2 is the larger value
        swap_flag = False
        if y1 > y2:
            y1, y2 = y2, y1
            x1, x2 = x2, x1  # Swap x positions accordingly
            assert y1_exe > y2_exe
            y1_exe, y2_exe = y2_exe, y1_exe
            swap_flag = True

        # Draw a vertical dotted line at the middle of x position of the lower bar
        ax.plot([(x1 + x2) / 2, (x1 + x2) / 2], [y1, y2], color='red', linestyle=':', linewidth=1)

        # Draw a horizontal solid line at the height of the higher bar, centered at the middle
        ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y2, y2], color='red', linestyle='-', linewidth=1)
        # Draw a horizontal solid line at the height of the lower bar, centered at the middle
        ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y1, y1], color='red', linestyle='-', linewidth=1)

        # Adjust the position of the speedup representation slightly above the higher bar
        y_speedup = y2 + 1.2 * ax.get_ylim()[1] / 10  # Slightly above the higher bar

        # Draw a vertical dotted line at the middle of x position of the lower bar
        ax.plot([(x1 + x2) / 2, (x1 + x2) / 2], [y1_exe, y2_exe], color='blue', linestyle=':', linewidth=1)

        # Draw a horizontal solid line at the height of the higher bar, centered at the middle
        ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y2_exe, y2_exe], color='blue', linestyle='-',
                linewidth=1)
        # Draw a horizontal solid line at the height of the lower bar, centered at the middle
        ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y1_exe, y1_exe], color='blue', linestyle='-',
                linewidth=1)

        # Adjust the position of the speedup representation slightly above the higher bar
        y_speedup_exe = y2 + 0.7 * ax.get_ylim()[1] / 10  # Slightly above the higher bar

        if swap_flag:
            # Compute speedup
            speedup = y2 / y1
            # Draw speedup annotation in red
            ax.text((x1 + x2) / 2, y_speedup, f'E2E {speedup:.2f}x ↑',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')

            # Compute speedup
            speedup_exe = y2_exe / y1_exe
            # Draw speedup annotation in red
            ax.text((x1 + x2) / 2, y_speedup_exe, f'EXE {speedup_exe:.2f}x ↑',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='blue')
        else:
            # # Compute slowdown
            # slowdown = y2 / y1 - 1
            # # Draw speedup annotation in red
            # ax.text((x1 + x2) / 2, y_speedup, f'Slowdown: {slowdown * 100:.2f}%',
            #         ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
            # Compute slowdown
            slowdown = y1 / y2
            # Draw speedup annotation in red
            ax.text((x1 + x2) / 2, y_speedup, f'E2E {slowdown:.2f}x ↓',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')

            # Compute slowdown
            slowdown_exe = y1_exe / y2_exe
            # Draw speedup annotation in red
            ax.text((x1 + x2) / 2, y_speedup_exe, f'EXE {slowdown_exe:.2f}x ↓',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='blue')

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


def plot_end2end_stacked_common(col1_data, col1_pattern,
                                col2_data, col2_pattern,
                                col3_data, col3_pattern,
                                col4_data, col4_pattern,
                                col5_data, col5_pattern,
                                col6_data, col6_pattern,
                                legend_names, legend_colors, title,
                                name):
    values = [sum(col1_data.values()), sum(col2_data.values()), sum(col3_data.values()), sum(col4_data.values()),
              sum(col5_data.values()), sum(col6_data.values())]
    colors = ['lightgreen', '#fcdb00', '#e695f5', '#ff954a', 'lightblue']
    patterns = [col1_pattern, col2_pattern, col3_pattern, col4_pattern, col5_pattern, col6_pattern]
    labels = ["Other", "Analyze", "Optimization", "AQP-Process", "Execution"]

    other_time = {0: col1_data["Other"], 1: col2_data["Other"], 2: col3_data["Other"], 3: col4_data["Other"],
                  4: col5_data["Other"], 5: col6_data["Other"]}
    analyze_time = {0: col1_data["Analyze"], 1: col2_data["Analyze"], 2: col3_data["Analyze"], 3: col4_data["Analyze"],
                    4: col5_data["Analyze"], 5: col6_data["Analyze"]}
    optimization_time = {0: col1_data["Optimization"], 1: col2_data["Optimization"], 2: col3_data["Optimization"],
                         3: col4_data["Optimization"], 4: col5_data["Optimization"], 5: col6_data["Optimization"]}
    aqp_process_time = {0: col1_data["AQP-Process"], 1: col2_data["AQP-Process"], 2: col3_data["AQP-Process"],
                        3: col4_data["AQP-Process"], 4: col5_data["AQP-Process"], 5: col6_data["AQP-Process"]}
    execution_time = {0: col1_data["Execution"], 1: col2_data["Execution"], 2: col3_data["Execution"],
                      3: col4_data["Execution"], 4: col5_data["Execution"], 5: col6_data["Execution"]}

    x_positions = [0, 0.5, 2, 2.5, 4, 4.5]

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
    ax.set_xticks([0.25, 2.25, 4.25])  # Center labels for groups
    ax.set_xticklabels(['DuckDB', 'DuckDB (common)', 'PostgreSQL (common)'])

    # Add vertical dotted lines, horizontal solid lines, and speedup representation
    for i in range(3):  # Two groups (Group 1 and Group 2)
        x1, x2 = x_positions[i * 2], x_positions[i * 2 + 1]  # Get x positions of bars in the group
        y1, y2 = values[i * 2], values[i * 2 + 1]  # Get heights of bars in the group
        y1_exe, y2_exe = execution_time[i * 2], execution_time[i * 2 + 1]

        # Ensure y1 is the smaller value and y2 is the larger value
        swap_flag = False
        if y1 > y2:
            y1, y2 = y2, y1
            x1, x2 = x2, x1  # Swap x positions accordingly
            assert y1_exe > y2_exe
            y1_exe, y2_exe = y2_exe, y1_exe
            swap_flag = True

        # Draw a vertical dotted line at the middle of x position of the lower bar
        ax.plot([(x1 + x2) / 2, (x1 + x2) / 2], [y1, y2], color='red', linestyle=':', linewidth=1)

        # Draw a horizontal solid line at the height of the higher bar, centered at the middle
        ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y2, y2], color='red', linestyle='-', linewidth=1)
        # Draw a horizontal solid line at the height of the lower bar, centered at the middle
        ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y1, y1], color='red', linestyle='-', linewidth=1)

        # Adjust the position of the speedup representation slightly above the higher bar
        y_speedup = y2 + 1.2 * ax.get_ylim()[1] / 10  # Slightly above the higher bar

        # Draw a vertical dotted line at the middle of x position of the lower bar
        ax.plot([(x1 + x2) / 2, (x1 + x2) / 2], [y1_exe, y2_exe], color='blue', linestyle=':', linewidth=1)

        # Draw a horizontal solid line at the height of the higher bar, centered at the middle
        ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y2_exe, y2_exe], color='blue', linestyle='-',
                linewidth=1)
        # Draw a horizontal solid line at the height of the lower bar, centered at the middle
        ax.plot([(x1 + x2) / 2 - 0.05, (x1 + x2) / 2 + 0.05], [y1_exe, y1_exe], color='blue', linestyle='-',
                linewidth=1)

        # Adjust the position of the speedup representation slightly above the higher bar
        y_speedup_exe = y2 + 0.7 * ax.get_ylim()[1] / 10  # Slightly above the higher bar

        if swap_flag:
            # Compute speedup
            speedup = y2 / y1
            # Draw speedup annotation in red
            ax.text((x1 + x2) / 2, y_speedup, f'E2E {speedup:.2f}x ↑',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')

            # Compute speedup
            speedup_exe = y2_exe / y1_exe
            # Draw speedup annotation in red
            ax.text((x1 + x2) / 2, y_speedup_exe, f'EXE {speedup_exe:.2f}x ↑',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='blue')
        else:
            # # Compute slowdown
            # slowdown = y2 / y1 - 1
            # # Draw speedup annotation in red
            # ax.text((x1 + x2) / 2, y_speedup, f'Slowdown: {slowdown * 100:.2f}%',
            #         ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
            # Compute slowdown
            slowdown = y1 / y2
            # Draw speedup annotation in red
            ax.text((x1 + x2) / 2, y_speedup, f'E2E {slowdown:.2f}x ↓',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')

            # Compute slowdown
            slowdown_exe = y1_exe / y2_exe
            # Draw speedup annotation in red
            ax.text((x1 + x2) / 2, y_speedup_exe, f'EXE {slowdown_exe:.2f}x ↓',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='blue')

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


def compare_query_by_query(col1_data: dict, col2_data: dict, title):
    x = np.arange(len(col1_data))
    plt.figure(figsize=(12, 6))
    speedups = {key: (col1_data[key] - col2_data[key]) / 1000
    if col1_data[key] > col2_data[key]
    else (col1_data[key] - col2_data[key]) / 1000
                for key in col1_data}
    sorted_speedups = sorted(speedups.items(), key=lambda item: item[1])
    queries = [item[0] for item in sorted_speedups]
    values = [item[1] for item in sorted_speedups]

    plt.bar(queries, values, color='skyblue')
    plt.xticks(x, labels=queries, rotation=90, ha='right')
    plt.xlabel("Query ID")
    plt.ylabel("Query by query execution speedup/slowdown execution time (s)")

    plt.tight_layout()
    plt.savefig("../figures/query-by-query_" + title + ".pdf")
    plt.clf()
    plt.close()


def compare_query_by_query_violin_dict(col1_data: dict, col2_data: dict, title):
    plt.figure(figsize=(12, 6))
    speedups = {key: math.log((col1_data[key] - col2_data[key]) / col2_data[key] * 100)
    if col1_data[key] > col2_data[key]
    else -math.log((col2_data[key] - col1_data[key]) / col1_data[key] * 100)
                for key in col1_data}

    # Sort and identify top/bottom 5 outliers
    sorted_speedups = sorted(speedups.items(), key=lambda x: x[1])
    outliers = sorted_speedups[:1] + sorted_speedups[-1:]

    values = [item[1] for item in speedups.items()]
    sns.violinplot(y=values, inner="box", inner_kws=dict(box_width=25, whis_width=5), color="skyblue")

    # Annotate outliers
    for key, val in outliers:
        plt.text(
            x=0.02, y=val,
            s=f"{key}: {val:.1f}%",
            fontsize=14, fontweight="bold", color="red"
        )

    plt.ylabel("Execution Comparison (%)", fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("../figures/query-by-query_violin_" + title + ".pdf")
    plt.clf()
    plt.close()


def compare_query_by_query_violin_dict_common(col1_data, col2_data, col3_data, col4_data, col5_data, col6_data,
                                              group_labels, title):
    all_data = []

    def add_pairwise_speedups(source_dict, base_dict, group_label):
        for key in source_dict:
            # diff = source_dict[key] - base_dict[key]
            if key in base_dict and base_dict[key] != 0 and source_dict[key] != 0:
                # if diff > 0:
                #     speedup = diff / base_dict[key] * 100
                # else:
                #     speedup = diff / source_dict[key] * 100
                # if -e < speedup < e:
                #     speedup = 0
                # elif speedup >= e:
                #     speedup = math.log(speedup)
                # else:
                #     speedup = -math.log(-speedup)
                speedup = source_dict[key] / base_dict[key]
                all_data.append({
                    "sql_name": key,
                    "speedup": speedup,
                    "group": group_label
                })

    # Calculate speedups
    add_pairwise_speedups(col1_data, col2_data, group_labels[0])
    add_pairwise_speedups(col3_data, col4_data, group_labels[1])
    add_pairwise_speedups(col5_data, col6_data, group_labels[2])

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Plot setup
    plt.figure(figsize=(12, 4))
    ax = sns.violinplot(data=df, x="group", y="speedup", hue="group", density_norm="count",
                        inner="box", inner_kws=dict(box_width=25, whis_width=5),
                        palette=['#00FF00', '#03fc98', '#03BAFC'], legend=False)

    sns.stripplot(data=df, x="group", y="speedup",
                  color='white', size=4, jitter=True, alpha=0.6, edgecolor="auto", linewidth=0.3)

    # Annotate outliers with improved positioning
    for i, group in enumerate(group_labels):
        group_data = df[df["group"] == group]
        if group_data.empty:
            continue

        # Get extreme values
        max_val = group_data["speedup"].max()
        min_val = group_data["speedup"].min()

        # Get corresponding query names
        max_query = group_data.loc[group_data["speedup"].idxmax(), "sql_name"]
        min_query = group_data.loc[group_data["speedup"].idxmin(), "sql_name"]

        # # Annotate maximum
        # plt.annotate(f"{max_query}: {math.exp(max_val):.1f}%",
        #              xy=(i, max_val),
        #              xytext=(i - 0.1, max_val + ax.get_ylim()[1]/10),
        #              fontsize=16, fontweight='bold', color='darkred',
        #              arrowprops=dict(arrowstyle="->", color='darkred', lw=1.5,
        #                              connectionstyle="arc3,rad=-0.3",  # Negative rad for left-curving
        #                              relpos=(0.5, 0.5)),
        #              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=1, alpha=0.9),
        #              horizontalalignment='right')
        #
        # # Annotate minimum
        # plt.annotate(f"{min_query}: {-math.exp(-min_val):.1f}%",
        #              xy=(i, min_val),
        #              xytext=(i - 0.1, min_val - ax.get_ylim()[1]/10),
        #              fontsize=16, fontweight='bold', color='darkblue',
        #              arrowprops=dict(arrowstyle="->", color='darkblue', lw=1.5,
        #                              connectionstyle="arc3,rad=0.3",  # Positive rad for right-curving
        #                              relpos=(0.5, 0.5)),
        #              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkblue", lw=1, alpha=0.9),
        #              horizontalalignment='right')

    # Reference line and scaling
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Labels and titles
    plt.ylabel("Execution Comparison (%)", fontsize=16, fontweight='bold')
    plt.xlabel("")
    plt.xticks(fontsize=16, fontweight='bold')
    yticks = plt.yticks()[0]
    plt.yticks(yticks, [f"$e^{{{int(tick)}}}$" for tick in yticks], fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()
    # plt.show()
    plt.savefig("../figures/query-by-query_violin_" + title + ".pdf")
    plt.clf()
    plt.close()


def compare_query_by_query_violin_dataframe(col1_data: pd.DataFrame, col2_data: pd.DataFrame, title):
    plt.figure(figsize=(12, 6))
    # speedups = np.where(col1_data["mean"] > col2_data["mean"],
    #                     np.log(col1_data["mean"] - col2_data["mean"] / col2_data["mean"] * 100),
    #                     -np.log(col2_data["mean"] - col1_data["mean"]) / col1_data["mean"] * 100)
    speedups = col1_data["mean"] / col2_data["mean"]
    merged = col1_data.copy()
    merged["speedup"] = speedups
    outliers = pd.concat([
        merged.nlargest(1, "speedup"),
        merged.nsmallest(1, "speedup")
    ])
    sns.violinplot(y=merged["speedup"], inner="box", inner_kws=dict(box_width=25, whis_width=5), color="skyblue")

    # Annotate outliers
    for _, row in outliers.iterrows():
        plt.text(
            x=0.05,  # slight x-offset for visibility
            y=row["speedup"],
            s=f'{row["sql_name"]}: {row["speedup"]:.1f}%',
            fontsize=14,
            fontweight='bold',
            color='red'
        )

    plt.ylabel("End2end Comparison (%)", fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("../figures/query-by-query_violin_" + title + ".pdf")
    plt.clf()
    plt.close()


def compare_query_by_query_violin_dataframe_common(col1_data, col2_data, col3_data, col4_data, col5_data, col6_data,
                                                   group_labels, title):
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

    # Calculate speedups
    add_pairwise_speedups(col1_data, col2_data, group_labels[0])
    add_pairwise_speedups(col3_data, col4_data, group_labels[1])
    add_pairwise_speedups(col5_data, col6_data, group_labels[2])

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    plt.figure(figsize=(12, 4))
    ax = sns.violinplot(data=df, x="group", y="speedup", hue="group", density_norm="count",
                        inner="box", inner_kws=dict(box_width=25, whis_width=5),
                        palette=['#00FF00', '#03fc98', '#03BAFC'], legend=False)

    sns.stripplot(data=df, x="group", y="speedup",
                  color='black', size=4, jitter=True, alpha=0.6, edgecolor="auto", linewidth=0.3)

    # voilin_colors = ['#b6f2b6', '#a4edd0', '#75cff0']  # Use these base colors
    # point_colors = ['#085908', '#036940', '#055978']
    # alphas = [0.2, 0.2, 0.2]  # Adjust transparency level here (0.0 to 1.0)
    # for i, group in enumerate(group_labels):
    #     group_data = df[df["group"] == group]
    #     ax = sns.violinplot(data=group_data, x="group", y="speedup",
    #                    inner="box", inner_kws=dict(box_width=25, whis_width=5),
    #                    color=voilin_colors[i], saturation=alphas[i], linewidth=1)
    #
    #     # Plot matching points
    #     sns.stripplot(data=group_data, x="group", y="speedup",
    #                   color=point_colors[i], jitter=0.2, size=4, edgecolor="auto", linewidth=0.5)

    # Annotate outliers with improved positioning
    for i, group in enumerate(group_labels):
        group_data = df[df["group"] == group]
        if group_data.empty:
            continue

        # Get extreme values
        max_val = group_data["speedup"].max()
        min_val = group_data["speedup"].min()

        # Get corresponding query names
        max_query = group_data.loc[group_data["speedup"].idxmax(), "sql_name"]
        min_query = group_data.loc[group_data["speedup"].idxmin(), "sql_name"]

        # Annotate maximum
        plt.annotate(f"{max_query}: {math.exp(max_val):.1f}%",
                     xy=(i, max_val),
                     xytext=(i - 0.1, max_val + ax.get_ylim()[1]/10),
                     fontsize=16, fontweight='bold', color='darkred',
                     arrowprops=dict(arrowstyle="->", color='darkred', lw=1.5,
                                     connectionstyle="arc3,rad=-0.3",  # Negative rad for left-curving
                                     relpos=(0.5, 0.5)),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", lw=1, alpha=0.9),
                     horizontalalignment='right')

        # Annotate minimum
        plt.annotate(f"{min_query}: {-math.exp(-min_val):.1f}%",
                     xy=(i, min_val),
                     xytext=(i - 0.1, min_val - ax.get_ylim()[1]/10),
                     fontsize=16, fontweight='bold', color='darkblue',
                     arrowprops=dict(arrowstyle="->", color='darkblue', lw=1.5,
                                     connectionstyle="arc3,rad=0.3",  # Positive rad for right-curving
                                     relpos=(0.5, 0.5)),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkblue", lw=1, alpha=0.9),
                     horizontalalignment='right')

    # Reference line and scaling
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Labels and titles
    plt.ylabel("End2end Comparison (%)", fontsize=16, fontweight='bold')
    plt.xlabel("")
    plt.xticks(fontsize=16, fontweight='bold')
    yticks = plt.yticks()[0]
    plt.yticks(ticks=yticks, labels=[f"$e^{{{int(tick)}}}$" for tick in yticks], fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()
    # plt.show()
    plt.savefig("../figures/query-by-query_violin_" + title + ".pdf")
    plt.clf()
    plt.close()


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
    ax.set_ylabel('End to end Time (s)', fontsize=14, fontweight='bold')
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


if __name__ == "__main__":
    ### prepare duckdb data ###
    vanilla_duckdb_data = analyze_csv_data(vanilla_duckdb_log, "duckdb")
    aqp_duckdb_data = analyze_csv_data(aqp_duckdb_log, "duckdb")
    aqp_duckdb_wo_stats_data = analyze_csv_data(aqp_duckdb_wo_stats_log, "duckdb")

    vanilla_duckdb_sum = sum(vanilla_duckdb_data['mean'])
    aqp_duckdb_sum = sum(aqp_duckdb_data['mean'])
    aqp_duckdb_wo_stats_sum = sum(aqp_duckdb_wo_stats_data['mean'])

    # analyze vanilla breakdown time
    vanilla_duckdb_breakdown_data = analyze_vanilla_breakdown(vanilla_duckdb_breakdown_log)
    vanilla_duckdb_execute_time = {query_name: metrics[' Execute'] for query_name, metrics in
                                   vanilla_duckdb_breakdown_data.items()}
    vanilla_exe_sum = sum(sub_dict[' Execute'] for sub_dict in vanilla_duckdb_breakdown_data.values()) / 1000
    vanilla_opt_sum = sum(sub_dict['PreOptimize'] for sub_dict in vanilla_duckdb_breakdown_data.values()) / 1000 + \
                      sum(sub_dict[' final-PostOptimize'] for sub_dict in vanilla_duckdb_breakdown_data.values()) / 1000
    # we only need 1. execution time, 2. AQP-process time, 3. optimization time, and 4. other time (including parse, etc.)
    vanilla_duckdb_breakdown = dict()
    vanilla_duckdb_breakdown['Execution'] = vanilla_exe_sum
    vanilla_duckdb_breakdown['AQP-Process'] = 0
    vanilla_duckdb_breakdown['Optimization'] = vanilla_opt_sum
    vanilla_duckdb_breakdown['Analyze'] = 0
    vanilla_duckdb_breakdown['Other'] = vanilla_duckdb_sum - vanilla_exe_sum - vanilla_opt_sum

    # analyze whole plan execution time
    aqp_duckdb_whole_plan_data = analyze_whole_plan_csv_data(aqp_duckdb_whole_plan_exe_time_log)
    aqp_duckdb_wo_stats_whole_plan_data = analyze_whole_plan_csv_data(aqp_duckdb_wo_stats_whole_plan_exe_time_log)

    aqp_duckdb_whole_plan_sum = sum(aqp_duckdb_whole_plan_data.values()) / 1000
    aqp_duckdb_wo_stats_whole_plan_sum = sum(aqp_duckdb_wo_stats_whole_plan_data.values()) / 1000

    # analyze AQP breakdown time
    aqp_duckdb_breakdown_data = analyze_duckdb_breakdown(aqp_duckdb_breakdown_log)
    aqp_duckdb_execute_time = {query_name: metrics['execute'] + metrics['final_exe'] for query_name, metrics in
                               aqp_duckdb_breakdown_data.items()}
    aqp_duckdb_exe_sum = sum(sub_dict['execute'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000 + \
                         sum(sub_dict['final_exe'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000
    aqp_duckdb_opt_sum = sum(sub_dict['pre_opt'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000 + \
                         sum(sub_dict['post-opt'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000 + \
                         sum(sub_dict['final_post_opt'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000
    aqp_duckdb_process_sum = sum(
        sub_dict['AQP-pre-process'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000 + \
                             sum(sub_dict['adapt-select'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000 + \
                             sum(sub_dict['AQP-post-process'] for sub_dict in aqp_duckdb_breakdown_data.values()) / 1000
    aqp_duckdb_breakdown = dict()
    aqp_duckdb_breakdown['Execution'] = aqp_duckdb_exe_sum
    aqp_duckdb_breakdown['AQP-Process'] = aqp_duckdb_process_sum
    aqp_duckdb_breakdown['Optimization'] = aqp_duckdb_opt_sum
    aqp_duckdb_breakdown['Analyze'] = 0
    aqp_duckdb_breakdown['Other'] = aqp_duckdb_sum - aqp_duckdb_exe_sum - aqp_duckdb_process_sum - aqp_duckdb_opt_sum

    aqp_duckdb_wo_stats_breakdown_data = analyze_duckdb_breakdown(aqp_duckdb_wo_stats_breakdown_log)
    aqp_duckdb_wo_stats_execute_time = {query_name: metrics['execute'] + metrics['final_exe'] for query_name, metrics in
                                        aqp_duckdb_wo_stats_breakdown_data.items()}
    aqp_duckdb_wo_stats_exe_sum = sum(
        sub_dict['execute'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data.values()) / 1000 + \
                                  sum(sub_dict['final_exe'] for sub_dict in
                                      aqp_duckdb_wo_stats_breakdown_data.values()) / 1000
    aqp_duckdb_wo_stats_opt_sum = sum(
        sub_dict['pre_opt'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data.values()) / 1000 + \
                                  sum(sub_dict['post-opt'] for sub_dict in
                                      aqp_duckdb_wo_stats_breakdown_data.values()) / 1000 + \
                                  sum(sub_dict['final_post_opt'] for sub_dict in
                                      aqp_duckdb_wo_stats_breakdown_data.values()) / 1000
    aqp_duckdb_wo_stats_process_sum = sum(
        sub_dict['AQP-pre-process'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data.values()) / 1000 + \
                                      sum(sub_dict['adapt-select'] for sub_dict in
                                          aqp_duckdb_wo_stats_breakdown_data.values()) / 1000 + \
                                      sum(sub_dict['AQP-post-process'] for sub_dict in
                                          aqp_duckdb_wo_stats_breakdown_data.values()) / 1000
    aqp_duckdb_wo_stats_breakdown = dict()
    aqp_duckdb_wo_stats_breakdown['Execution'] = aqp_duckdb_wo_stats_exe_sum
    aqp_duckdb_wo_stats_breakdown['AQP-Process'] = aqp_duckdb_wo_stats_process_sum
    aqp_duckdb_wo_stats_breakdown['Optimization'] = aqp_duckdb_wo_stats_opt_sum
    aqp_duckdb_wo_stats_breakdown['Analyze'] = 0
    aqp_duckdb_wo_stats_breakdown[
        'Other'] = aqp_duckdb_wo_stats_sum - aqp_duckdb_wo_stats_exe_sum - aqp_duckdb_wo_stats_process_sum - aqp_duckdb_wo_stats_opt_sum

    ### prepare duckdb data of the common queries ###
    vanilla_duckdb_data_common = vanilla_duckdb_data[vanilla_duckdb_data['sql_name'].isin(common_queries)]
    aqp_duckdb_data_common = aqp_duckdb_data[aqp_duckdb_data['sql_name'].isin(common_queries)]
    aqp_duckdb_wo_stats_data_common = aqp_duckdb_wo_stats_data[
        aqp_duckdb_wo_stats_data['sql_name'].isin(common_queries)]

    vanilla_duckdb_sum_common = sum(vanilla_duckdb_data_common['mean'])
    aqp_duckdb_sum_common = sum(aqp_duckdb_data_common['mean'])
    aqp_duckdb_wo_stats_sum_common = sum(aqp_duckdb_wo_stats_data_common['mean'])

    # analyze vanilla breakdown time
    vanilla_duckdb_breakdown_data_common = {k: v for k, v in vanilla_duckdb_breakdown_data.items() if
                                            k in common_queries}
    vanilla_duckdb_execute_time_common = {query_name: metrics[' Execute'] for query_name, metrics in
                                          vanilla_duckdb_breakdown_data_common.items()}
    vanilla_exe_sum = sum(sub_dict[' Execute'] for sub_dict in vanilla_duckdb_breakdown_data_common.values()) / 1000
    vanilla_opt_sum = sum(
        sub_dict['PreOptimize'] for sub_dict in vanilla_duckdb_breakdown_data_common.values()) / 1000 + sum(
        sub_dict[' final-PostOptimize'] for sub_dict in vanilla_duckdb_breakdown_data_common.values()) / 1000
    # we only need 1. execution time, 2. AQP-process time, 3. optimization time, and 4. other time (including parse, etc.)
    vanilla_duckdb_breakdown_common = dict()
    vanilla_duckdb_breakdown_common['Execution'] = vanilla_exe_sum
    vanilla_duckdb_breakdown_common['AQP-Process'] = 0
    vanilla_duckdb_breakdown_common['Optimization'] = vanilla_opt_sum
    vanilla_duckdb_breakdown_common['Analyze'] = 0
    vanilla_duckdb_breakdown_common['Other'] = vanilla_duckdb_sum_common - vanilla_exe_sum - vanilla_opt_sum

    # analyze whole plan execution time
    aqp_duckdb_whole_plan_data_common = {k: v for k, v in aqp_duckdb_whole_plan_data.items() if
                                         k in common_queries}
    aqp_duckdb_wo_stats_whole_plan_data_common = {k: v for k, v in aqp_duckdb_wo_stats_whole_plan_data.items() if
                                                  k in common_queries}

    aqp_duckdb_whole_plan_sum_common = sum(aqp_duckdb_whole_plan_data_common.values()) / 1000
    aqp_duckdb_wo_stats_whole_plan_sum_common = sum(aqp_duckdb_wo_stats_whole_plan_data_common.values()) / 1000

    # analyze AQP breakdown time
    aqp_duckdb_breakdown_data_common = {k: v for k, v in aqp_duckdb_breakdown_data.items() if
                                        k in common_queries}
    aqp_duckdb_execute_time_common = {query_name: metrics['execute'] + metrics['final_exe'] for query_name, metrics in
                                      aqp_duckdb_breakdown_data_common.items()}
    aqp_duckdb_exe_sum = sum(
        sub_dict['execute'] for sub_dict in aqp_duckdb_breakdown_data_common.values()) / 1000 + sum(
        sub_dict['final_exe'] for sub_dict in aqp_duckdb_breakdown_data_common.values()) / 1000
    aqp_duckdb_opt_sum = sum(
        sub_dict['pre_opt'] for sub_dict in aqp_duckdb_breakdown_data_common.values()) / 1000 + sum(
        sub_dict['post-opt'] for sub_dict in aqp_duckdb_breakdown_data_common.values()) / 1000 + sum(
        sub_dict['final_post_opt'] for sub_dict in aqp_duckdb_breakdown_data_common.values()) / 1000
    aqp_duckdb_process_sum = sum(
        sub_dict['AQP-pre-process'] for sub_dict in aqp_duckdb_breakdown_data_common.values()) / 1000 + sum(
        sub_dict['adapt-select'] for sub_dict in aqp_duckdb_breakdown_data_common.values()) / 1000 + sum(
        sub_dict['AQP-post-process'] for sub_dict in aqp_duckdb_breakdown_data_common.values()) / 1000
    aqp_duckdb_breakdown_common = dict()
    aqp_duckdb_breakdown_common['Execution'] = aqp_duckdb_exe_sum
    aqp_duckdb_breakdown_common['AQP-Process'] = aqp_duckdb_process_sum
    aqp_duckdb_breakdown_common['Optimization'] = aqp_duckdb_opt_sum
    aqp_duckdb_breakdown_common['Analyze'] = 0
    aqp_duckdb_breakdown_common[
        'Other'] = aqp_duckdb_sum_common - aqp_duckdb_exe_sum - aqp_duckdb_process_sum - aqp_duckdb_opt_sum

    aqp_duckdb_wo_stats_breakdown_data_common = {k: v for k, v in aqp_duckdb_wo_stats_breakdown_data.items() if
                                                 k in common_queries}
    aqp_duckdb_wo_stats_exe_sum = sum(
        sub_dict['execute'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data_common.values()) / 1000 + sum(
        sub_dict['final_exe'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data_common.values()) / 1000
    aqp_duckdb_wo_stats_opt_sum = sum(
        sub_dict['pre_opt'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data_common.values()) / 1000 + sum(
        sub_dict['post-opt'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data_common.values()) / 1000 + sum(
        sub_dict['final_post_opt'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data_common.values()) / 1000
    aqp_duckdb_wo_stats_process_sum = sum(
        sub_dict['AQP-pre-process'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data_common.values()) / 1000 + sum(
        sub_dict['adapt-select'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data_common.values()) / 1000 + sum(
        sub_dict['AQP-post-process'] for sub_dict in aqp_duckdb_wo_stats_breakdown_data_common.values()) / 1000
    aqp_duckdb_wo_stats_breakdown_common = dict()
    aqp_duckdb_wo_stats_breakdown_common['Execution'] = aqp_duckdb_wo_stats_exe_sum
    aqp_duckdb_wo_stats_breakdown_common['AQP-Process'] = aqp_duckdb_wo_stats_process_sum
    aqp_duckdb_wo_stats_breakdown_common['Optimization'] = aqp_duckdb_wo_stats_opt_sum
    aqp_duckdb_wo_stats_breakdown_common['Analyze'] = 0
    aqp_duckdb_wo_stats_breakdown_common[
        'Other'] = aqp_duckdb_wo_stats_sum_common - aqp_duckdb_wo_stats_exe_sum - aqp_duckdb_wo_stats_process_sum - aqp_duckdb_wo_stats_opt_sum

    ### prepare postgres data ###
    vanilla_pg_data = analyze_csv_data(vanilla_pg_log, "postgres")
    aqp_pg_data = analyze_csv_data(aqp_pg_log, "postgres")
    aqp_pg_wo_stats_data = analyze_csv_data(aqp_pg_wo_stats_log, "postgres")

    vanilla_pg_sum = sum(vanilla_pg_data['mean'])
    aqp_pg_sum = sum(aqp_pg_data['mean'])
    aqp_pg_wo_stats_sum = sum(aqp_pg_wo_stats_data['mean'])

    # analyze vanilla breakdown time
    vanilla_pg_breakdown_data = analyze_vanilla_breakdown(vanilla_pg_breakdown_log)
    vanilla_pg_execute_time = {query_name: metrics[' Execute'] for query_name, metrics in
                               vanilla_pg_breakdown_data.items()}
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
    aqp_pg_execute_time = {query_name: metrics['execute'] + metrics['final_exe'] for query_name, metrics in
                           aqp_pg_breakdown_data.items()}
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
    aqp_pg_wo_stats_execute_time = {query_name: metrics['execute'] + metrics['final_exe'] for query_name, metrics in
                                    aqp_pg_wo_stats_breakdown_data.items()}
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

    (vanilla_duckdb, vanilla_duckdb_specify_join,
     aqp_duckdb, aqp_duckdb_wo_stat, apq_duckdb_wo_split) = analyze_duckdb_excel_data(duckdb_log)
    # vanilla_duckdb_sum = sum(vanilla_duckdb) / 1000
    vanilla_duckdb_specify_join_sum = sum(vanilla_duckdb_specify_join) / 1000
    # aqp_duckdb_sum = sum(aqp_duckdb) / 1000
    # aqp_duckdb_wo_stats_sum = sum(aqp_duckdb_wo_stat) / 1000
    apq_duckdb_wo_split_sum = sum(apq_duckdb_wo_split) / 1000

    vanilla_pg, aqp_pg_wo_stat, aqp_pg = analyze_pg_excel_data(pg_log)
    vanilla_pg_sum = sum(vanilla_pg) / 1000
    aqp_pg_wo_stat_sum = sum(aqp_pg_wo_stat) / 1000
    aqp_pg_sum = sum(aqp_pg) / 1000

    ###############################################################################################
    # fig for q1: vanilla VS AQP (with updating cardinality)
    plot_end2end(vanilla_duckdb_sum, bar_colors[0], None,
                 aqp_duckdb_sum, bar_colors[1], 'x',
                 vanilla_pg_sum, bar_colors[2], None,
                 aqp_pg_sum, bar_colors[3], 'x',
                 ["Vanilla", "Plan-based AQP"],
                 ["darkgray", "lightgray"], "Total End-to-end Time (s)",
                 f'{benchmark}_q1_fig')

    plot_end2end_common(vanilla_duckdb_sum, bar_colors[0], None,
                        aqp_duckdb_sum, bar_colors[1], 'x',
                        vanilla_duckdb_sum_common, bar_colors[4], None,
                        aqp_duckdb_sum_common, bar_colors[5], 'x',
                        vanilla_pg_sum, bar_colors[2], None,
                        aqp_pg_sum, bar_colors[3], 'x',
                        ["Vanilla", "Plan-based AQP"],
                        ["darkgray", "lightgray"], "Total End-to-end Time (s)",
                        f'{benchmark}_q1_fig_common')

    plot_end2end_stacked(vanilla_duckdb_breakdown, None,
                         aqp_duckdb_breakdown, 'x',
                         vanilla_pg_breakdown, None,
                         aqp_pg_breakdown, 'x',
                         ["Vanilla", "Plan-based AQP"],
                         ["lightgray", "lightgray"], "Total End-to-end Breakdown Time (s)",
                         f'{benchmark}_q1_fig_breakdown')

    plot_end2end_stacked_common(vanilla_duckdb_breakdown, None,
                                aqp_duckdb_breakdown, 'x',
                                vanilla_duckdb_breakdown_common, None,
                                aqp_duckdb_breakdown_common, 'x',
                                vanilla_pg_breakdown, None,
                                aqp_pg_breakdown, 'x',
                                ["Vanilla", "Plan-based AQP"],
                                ["lightgray", "lightgray"], "Total End-to-end Breakdown Time (s)",
                                f'{benchmark}_q1_fig_breakdown_common')

    # fig for q2: w/wo updating cardinality
    plot_end2end(aqp_duckdb_wo_stats_sum, bar_colors[1], '/',
                 aqp_duckdb_sum, bar_colors[1], 'x',
                 aqp_pg_wo_stats_sum, bar_colors[3], '/',
                 aqp_pg_sum, bar_colors[3], 'x',
                 ["Plan-based AQP w/o Updating Cardinality", "Plan-based AQP with Updating Cardinality"],
                 ["lightgray", "lightgray"], "Total End-to-end Time (s)",
                 f'{benchmark}_q2_fig')

    plot_end2end_common(aqp_duckdb_wo_stats_sum, bar_colors[1], '/',
                        aqp_duckdb_sum, bar_colors[1], 'x',
                        aqp_duckdb_wo_stats_sum_common, bar_colors[5], '/',
                        aqp_duckdb_sum_common, bar_colors[5], 'x',
                        aqp_pg_wo_stats_sum, bar_colors[3], '/',
                        aqp_pg_sum, bar_colors[3], 'x',
                        ["Plan-based AQP w/o Updating Cardinality", "Plan-based AQP with Updating Cardinality"],
                        ["lightgray", "lightgray"], "Total End-to-end Time (s)",
                        f'{benchmark}_q2_fig_common')

    plot_end2end_stacked(aqp_duckdb_wo_stats_breakdown, '/',
                         aqp_duckdb_breakdown, 'x',
                         aqp_pg_wo_stats_breakdown, '/',
                         aqp_pg_breakdown, 'x',
                         ["Plan-based AQP w/o Updating Cardinality", "Plan-based AQP with Updating Cardinality"],
                         ["lightgray", "lightgray"], "Total End-to-end Breakdown Time (s)",
                         f'{benchmark}_q2_fig_breakdown')

    plot_end2end_stacked_common(aqp_duckdb_wo_stats_breakdown, '/',
                                aqp_duckdb_breakdown, 'x',
                                aqp_duckdb_wo_stats_breakdown_common, '/',
                                aqp_duckdb_breakdown_common, 'x',
                                aqp_pg_wo_stats_breakdown, '/',
                                aqp_pg_breakdown, 'x',
                                ["Plan-based AQP w/o Updating Cardinality", "Plan-based AQP with Updating Cardinality"],
                                ["lightgray", "lightgray"], "Total End-to-end Breakdown Time (s)",
                                f'{benchmark}_q2_fig_breakdown_common')

    # fig for q3: w/wo specifying join order and operator (wo split and wo updating cardinality), only execution time
    plot_end2end(vanilla_duckdb_breakdown['Execution'], bar_colors[0], None,
                 aqp_duckdb_wo_stats_whole_plan_sum, bar_colors[0], 'o',
                 vanilla_pg_breakdown['Execution'], bar_colors[2], None,
                 aqp_pg_wo_stats_whole_plan_sum, bar_colors[2], 'o',
                 ["Vanilla", "Vanilla with Specified Join Order and Operator"],
                 ["darkgray", "darkgray"], "Total Execution Time (s)",
                 f'{benchmark}_q3_fig')

    plot_end2end_common(vanilla_duckdb_breakdown['Execution'], bar_colors[0], None,
                        aqp_duckdb_wo_stats_whole_plan_sum, bar_colors[0], 'o',
                        vanilla_duckdb_breakdown_common['Execution'], bar_colors[4], None,
                        aqp_duckdb_wo_stats_whole_plan_sum_common, bar_colors[4], 'o',
                        vanilla_pg_breakdown['Execution'], bar_colors[2], None,
                        aqp_pg_wo_stats_whole_plan_sum, bar_colors[2], 'o',
                        ["Vanilla", "Vanilla with Specified Join Order and Operator"],
                        ["darkgray", "darkgray"], "Total Execution Time (s)",
                        f'{benchmark}_q3_fig_common')

    # fig for q4: C5 VS C7: w/wo splitting the plan (with updating Cardinality), only execution time
    plot_end2end(aqp_duckdb_breakdown['Execution'], bar_colors[1], 'x',
                 aqp_duckdb_whole_plan_sum, bar_colors[1], '\\',
                 aqp_pg_breakdown['Execution'], bar_colors[3], 'x',
                 aqp_pg_whole_plan_sum, bar_colors[3], '\\',
                 ["Plan-based AQP with Splitting Plan", "Plan-based AQP w/o Splitting Plan"],
                 ["lightgray", "lightgray"], "Total Execution Time (s)",
                 f'{benchmark}_q4_fig')

    plot_end2end_common(aqp_duckdb_breakdown['Execution'], bar_colors[1], 'x',
                        aqp_duckdb_whole_plan_sum, bar_colors[1], '\\',
                        aqp_duckdb_breakdown_common['Execution'], bar_colors[5], 'x',
                        aqp_duckdb_whole_plan_sum_common, bar_colors[5], '\\',
                        aqp_pg_breakdown['Execution'], bar_colors[3], 'x',
                        aqp_pg_whole_plan_sum, bar_colors[3], '\\',
                        ["Plan-based AQP with Splitting Plan", "Plan-based AQP w/o Splitting Plan"],
                        ["lightgray", "lightgray"], "Total Execution Time (s)",
                        f'{benchmark}_q4_fig_common')

    # vanilla VS best
    plot_end2end_common(vanilla_duckdb_breakdown['Execution'], bar_colors[0], None,
                        aqp_duckdb_whole_plan_sum, bar_colors[1], '\\',
                        vanilla_duckdb_breakdown_common['Execution'], bar_colors[4], None,
                        aqp_duckdb_whole_plan_sum_common, bar_colors[5], '\\',
                        vanilla_pg_breakdown['Execution'], bar_colors[2], None,
                        aqp_pg_whole_plan_sum, bar_colors[3], '\\',
                        ["Vanilla", "Plan-based AQP w/o Splitting Plan"],
                        ["darkgray", "lightgray"], "Total Execution Time (s)",
                        f'{benchmark}_vanilla_vs_best_common')

    ###############################################################################################
    # query by query compare
    compare_query_by_query(vanilla_duckdb_execute_time, aqp_duckdb_execute_time,
                           f'{benchmark}_vanilla_vs_AQP_DuckDB')
    compare_query_by_query(aqp_duckdb_wo_stats_execute_time, aqp_duckdb_execute_time,
                           f'{benchmark}_without_vs_with_Updating_Cardinality_DuckDB')
    compare_query_by_query(vanilla_duckdb_execute_time, aqp_duckdb_wo_stats_whole_plan_data,
                           f'{benchmark}_without_vs_with_Specifying_Join_Order_DuckDB')
    compare_query_by_query(aqp_duckdb_execute_time, aqp_duckdb_whole_plan_data,
                           f'{benchmark}_without_vs_with_Splitting_the_Plan_DuckDB')
    compare_query_by_query(vanilla_pg_execute_time, aqp_pg_execute_time,
                           f'{benchmark}_vanilla_vs_AQP_PostgreSQL')
    compare_query_by_query(aqp_pg_wo_stats_execute_time, aqp_pg_execute_time,
                           f'{benchmark}_without_vs_with_Updating_Cardinality_PostgreSQL')
    compare_query_by_query(vanilla_pg_execute_time, aqp_pg_wo_stats_whole_plan_data,
                           f'{benchmark}_without_vs_with_Specifying_Join_Order_PostgreSQL')
    compare_query_by_query(aqp_pg_execute_time, aqp_pg_whole_plan_data,
                           f'{benchmark}_without_vs_with_Splitting_the_Plan_PostgreSQL')

    # vanilla VS best
    compare_query_by_query(vanilla_duckdb_execute_time, aqp_duckdb_whole_plan_data,
                           f'{benchmark}_vanilla_vs_best_AQP_DuckDB')
    compare_query_by_query(vanilla_pg_execute_time, aqp_pg_whole_plan_data,
                           f'{benchmark}_vanilla_vs_best_AQP_PostgreSQL')

    # compare_query_by_query_violin_dataframe(vanilla_duckdb_data, aqp_duckdb_data, f'{benchmark}_vanilla_vs_AQP_DuckDB')
    # compare_query_by_query_violin_dataframe(aqp_duckdb_wo_stats_data, aqp_duckdb_data,
    #                                         f'{benchmark}_without_vs_with_Updating_Cardinality_DuckDB')
    # compare_query_by_query_violin_dict(vanilla_duckdb_execute_time, aqp_duckdb_wo_stats_whole_plan_data,
    #                                    f'{benchmark}_without_vs_with_Specifying_Join_Order_DuckDB')
    # compare_query_by_query_violin_dict(aqp_duckdb_execute_time, aqp_duckdb_whole_plan_data,
    #                                    f'{benchmark}_without_vs_with_Splitting_the_Plan_DuckDB')
    # compare_query_by_query_violin_dataframe(vanilla_pg_data, aqp_pg_data, f'{benchmark}_vanilla_vs_AQP_PostgreSQL')
    # compare_query_by_query_violin_dataframe(aqp_pg_wo_stats_data, aqp_pg_data,
    #                                         f'{benchmark}_without_vs_with_Updating_Cardinality_PostgreSQL')
    # compare_query_by_query_violin_dict(vanilla_pg_execute_time, aqp_pg_wo_stats_whole_plan_data,
    #                                    f'{benchmark}_without_vs_with_Specifying_Join_Order_PostgreSQL')
    # compare_query_by_query_violin_dict(aqp_pg_execute_time, aqp_pg_whole_plan_data,
    #                                    f'{benchmark}_without_vs_with_Splitting_the_Plan_PostgreSQL')

    compare_query_by_query_violin_dataframe_common(vanilla_duckdb_data, aqp_duckdb_data,
                                                   vanilla_duckdb_data_common, aqp_duckdb_data_common,
                                                   vanilla_pg_data, aqp_pg_data,
                                                   group_labels=["DuckDB", "DuckDB (common)", "PostgreSQL (common)"],
                                                   title=f'{benchmark}_vanilla_vs_AQP')
    compare_query_by_query_violin_dataframe_common(aqp_duckdb_wo_stats_data, aqp_duckdb_data,
                                                   aqp_duckdb_wo_stats_data_common, aqp_duckdb_data_common,
                                                   aqp_pg_wo_stats_data, aqp_pg_data,
                                                   group_labels=["DuckDB", "DuckDB (common)", "PostgreSQL (common)"],
                                                   title=f'{benchmark}_without_vs_with_Updating_Cardinality')
    compare_query_by_query_violin_dict_common(vanilla_duckdb_execute_time, aqp_duckdb_wo_stats_whole_plan_data,
                                              vanilla_duckdb_execute_time_common,
                                              aqp_duckdb_wo_stats_whole_plan_data_common,
                                              vanilla_pg_execute_time, aqp_pg_wo_stats_whole_plan_data,
                                              group_labels=["DuckDB", "DuckDB (common)", "PostgreSQL (common)"],
                                              title=f'{benchmark}_without_vs_with_Specifying_Join_Order')
    compare_query_by_query_violin_dict_common(aqp_duckdb_execute_time, aqp_duckdb_whole_plan_data,
                                              aqp_duckdb_execute_time_common, aqp_duckdb_whole_plan_data_common,
                                              aqp_pg_execute_time, aqp_pg_whole_plan_data,
                                              group_labels=["DuckDB", "DuckDB (common)", "PostgreSQL (common)"],
                                              title=f'{benchmark}_without_vs_with_Splitting_the_Plan')

    ###############################################################################################
    # query by query with deviation
    vanilla_duckdb_data_stats = plot_query_by_query_box_chart_with_deviation(vanilla_duckdb_data,
                                                                             f'{benchmark}_vanilla_DuckDB')
    aqp_duckdb_data_stats = plot_query_by_query_box_chart_with_deviation(aqp_duckdb_data, f'{benchmark}_AQP_DuckDB')
    # plot_query_by_query_box_chart_compare_with_deviation(vanilla_duckdb_data_stats, aqp_duckdb_data_stats)
    aqp_duckdb_wo_stats_data_stats = plot_query_by_query_box_chart_with_deviation(aqp_duckdb_wo_stats_data,
                                                                                  f'{benchmark}_AQP_DuckDB_without_Updating_Cardinality')
    # aqp_duckdb_whole_plan_data_stats = plot_query_by_query_box_chart_with_deviation(aqp_duckdb_whole_plan_data,
    #                                                                                 f'{benchmark}_AQP_DuckDB_whole_plan')
    # aqp_duckdb_wo_stats_whole_plan_data_stats = plot_query_by_query_box_chart_with_deviation(
    #     aqp_duckdb_wo_stats_whole_plan_data,
    #     f'{benchmark}_AQP_DuckDB_whole_plan_without_Updating_Cardinality')

    vanilla_pg_data_stats = plot_query_by_query_box_chart_with_deviation(vanilla_pg_data,
                                                                         f'{benchmark}_vanilla_PostgreSQL')
    aqp_pg_data_stats = plot_query_by_query_box_chart_with_deviation(aqp_pg_data, f'{benchmark}_AQP_PostgreSQL')
    aqp_pg_wo_stats_data_stats = plot_query_by_query_box_chart_with_deviation(aqp_pg_wo_stats_data,
                                                                              f'{benchmark}_AQP_PostgreSQL_without_Updating_Cardinality')
    # aqp_pg_whole_plan_data_stats = plot_query_by_query_box_chart_with_deviation(aqp_pg_whole_plan_data,
    #                                                                             f'{benchmark}_AQP_PostgreSQL_whole_plan')
    # aqp_pg_wo_stats_whole_plan_data_stats = plot_query_by_query_box_chart_with_deviation(
    #     aqp_pg_wo_stats_whole_plan_data,
    #     f'{benchmark}_AQP_PostgreSQL_whole_plan_without_Updating_Cardinality')
