import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.express as px
import numpy as np
import os
import shutil
import re
from collections import OrderedDict
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from matplotlib.font_manager import FontProperties

# -------------------- Load the JSON file --------------------

with open("results.json", "r") as f:
    data = json.load(f)

# -------------------- Create dynamic folder based on inputs --------------------

inputs = data.get("inputs", {})

crop = inputs.get("crop_type", "Unknown")
farm_size = inputs.get("farm_size", "NA")
budget = inputs.get("budget", "NA")
applications = inputs.get("applications", [])

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|/]', '-', name)

# Clean app names: replace spaces with "-" to avoid filesystem issues
apps_str = "_".join([sanitize_filename(app.replace(" ", "-")) for app in applications])

folder_name = f"Plots/Evaluation_of_design_aspects/{crop}_{budget}_{farm_size}_{apps_str}"

os.makedirs(folder_name, exist_ok=True)
shutil.copy("results.json", f"{folder_name}/results_EDA.json")


# -------------------- Flatten Optimizers --------------------

optimizers = ['cost_area', 'payload_cost','area_payload','proposed_approach']
nice = {
    'proposed_approach':  'Proposed Approach',
    'cost_area'        :  'cost+area',
    'payload_cost'     :  'payload+cost',
    'area_payload'     :  'area+payload',
}
flattened = {}

def fully_flatten(nested_list):
    """Fully flattens nested lists of lists of dicts"""
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(fully_flatten(item))
        else:
            flat.append(item)
    return flat

cmap = plt.get_cmap('tab10')
colors = {opt: cmap(i) for i,opt in enumerate(optimizers)}

for opt in optimizers:
    raw = data[opt]
    raw = fully_flatten(raw)
    flattened[opt] = raw

markers = {
    'proposed_approach': 's',
    'area_payload': '+',
    'cost_area': '^',
    'payload_cost': 'D'
}

# -------------------- Prepare Metrics --------------------

metrics = []
for opt in optimizers:
    for config in flattened[opt]:
        per_unit_payload = config['payload']
        payload          = config['payload']
        quantity         = config['quantity']
        edge_cost        = config['edge_server']['cost']

        total_coverage   = config['coverage'] * quantity
        total_payload    = payload * quantity
        per_unit_cost    = config['base_cost'] + config['additional_cost'] + (edge_cost / quantity)

        metrics.append({
            'optimizer'      : opt,
            'total_cost'     : config['total_cost'],
            'per_unit_cost'  : per_unit_cost,
            'coverage'       : total_coverage,
            'per_unit_payload': per_unit_payload,
            'payload'        : payload,
            'total_payload'  : total_payload,
            'quantity'       : quantity,
            'runtime'        : config['runtime_hours']
        })

df = pd.DataFrame(metrics)

# -------------------- Frontier Plot --------------------
left_end    = df[df.optimizer == 'payload_cost']['coverage'].max() * 1.05
gap_pct     = 0.0001
right_start = farm_size * (1 - gap_pct)
right_end   = df['coverage'].max() * 1.00005

y_low_max   = budget * 1.2
MARKER_SIZE = 500
# --- Master 1×3 grid:
#   [0] -> Unit cost vs unit payload
#   [1] -> Total cost vs total payload
#   [2] -> Total cost vs area coverage (broken X inside)
fig = plt.figure(figsize=(20, 9))
master = gridspec.GridSpec(
    1, 3,
    width_ratios=[1, 1, 1.8],
    wspace=0.35
)

# --- (1) Left: UNIT cost vs UNIT payload ---
ax1 = fig.add_subplot(master[0, 0])
for opt in optimizers:
    sub = df[df.optimizer == opt]
    m  = markers[opt]
    fc = colors[opt] if m in ['+', 'x'] else 'none'
    ec = colors[opt]

    ax1.scatter(
        sub['per_unit_payload'], sub['per_unit_cost'],
        label=nice[opt], marker=m,
        facecolors=fc, edgecolors=ec,
        s=MARKER_SIZE, alpha=0.8, linewidths=1.5
    )

ax1.set_xlabel("Unit Payload (kg)", fontsize=22, labelpad=25)
ax1.set_ylabel("Unit Cost in thousands ($)", fontsize=22)

# increase x-axis to 130 kg
ax1.set_xlim(0, 130)

# bottom caption instead of top title
ax1.text(
    0.5, -0.22, "(a) Unit Cost vs Unit Payload",
    transform=ax1.transAxes,
    ha="center", va="top",
    fontsize=20, fontweight="bold"
)

# --- (2) Middle: TOTAL cost vs TOTAL payload ---
ax2 = fig.add_subplot(master[0, 1])
for opt in optimizers:
    sub = df[df.optimizer == opt]
    m  = markers[opt]
    fc = colors[opt] if m in ['+', 'x'] else 'none'
    ec = colors[opt]

    ax2.scatter(
        sub['total_payload'], sub['total_cost'],
        label=nice[opt], marker=m,
        facecolors=fc, edgecolors=ec,
        s=MARKER_SIZE, alpha=0.8, linewidths=1.5
    )

# Budget line in total-cost space
ax2.axhline(budget, color='blue', linestyle='--', label='Budget Constraint')

ax2.set_xlabel("Total Payload (kg)", fontsize=22, labelpad=25)
ax2.set_ylabel("Total Cost in thousands ($)", fontsize=22)

ax2.text(
    0.5, -0.22, "(b) Total Cost vs Total Payload",
    transform=ax2.transAxes,
    ha="center", va="top",
    fontsize=20, fontweight="bold"
)

# --- (3) Right: TOTAL cost vs coverage (broken X, two panels) ---
subspec = gridspec.GridSpecFromSubplotSpec(
    1, 2,
    subplot_spec=master[0, 2],
    width_ratios=[1, 2],
    wspace=0.1
)

ax_bl = fig.add_subplot(subspec[0, 0])                # left coverage (low end)
ax_br = fig.add_subplot(subspec[0, 1], sharey=ax_bl)   # right coverage (high end)

for opt in optimizers:
    sub = df[df.optimizer == opt]
    x, y = sub['coverage'], sub['total_cost']
    m  = markers[opt]
    fc = colors[opt] if m in ['+', 'x'] else 'none'
    ec = colors[opt]

    # left side of coverage
    mask_left = (x <= left_end) & (y <= y_low_max)
    ax_bl.scatter(
        x[mask_left], y[mask_left],
        label=nice[opt], marker=m,
        facecolors=fc, edgecolors=ec,
        s=MARKER_SIZE, alpha=0.8, linewidths=1.5
    )

    # right side of coverage (no optimizer labels here)
    mask_right = (x >= right_start) & (y <= y_low_max)
    ax_br.scatter(
        x[mask_right], y[mask_right],
        marker=m,
        facecolors=fc, edgecolors=ec,
        s=MARKER_SIZE, alpha=0.8, linewidths=1.5
    )

# Constraints for coverage plots
for ax in (ax_bl, ax_br):
    ax.axhline(budget, color='blue', linestyle='--')  # budget line
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

ax_bl.axvline(farm_size, color='black', linestyle='--')
ax_br.axvline(farm_size, color='black', linestyle='--', label='Farm Size Constraint')

# Axis limits
ax_bl.set_xlim(0, left_end)
ax_bl.set_ylim(0, y_low_max)

ax_br.set_xlim(right_start, right_end)
ax_br.set_ylim(0, y_low_max)

# Labels for broken coverage plot
ax_bl.set_xlabel("Area coverage (m²)", fontsize=22, labelpad=25)
# ax_br.set_xlabel("Area coverage (m²)", fontsize=22)
ax_bl.set_ylabel("Total Cost in thousands ($)", fontsize=22)

# bottom caption for the pair (we put it on ax_br)
ax_br.text(
    0.5, -0.22, "(c) Total Cost vs Area Coverage",
    transform=ax_br.transAxes,
    ha="center", va="top",
    fontsize=20, fontweight="bold"
)

# Diagonal break marks between left and right coverage panels
d = 0.01
for ax in (ax_bl, ax_br):
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    if ax is ax_bl:
        # right edge of left panel
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    else:
        # left edge of right panel
        ax.plot((-d, +d), (-d, +d), **kwargs)
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# Hide duplicated Y labels on right coverage panel
ax_br.tick_params(axis='y', labelleft=False)
ax_br.set_ylabel('')

# Make sure bottom X ticks are visible on broken plot
for ax in (ax_bl, ax_br):
    ax.tick_params(labelbottom=True)

# --- Y-axis formatting ---
# More precise formatter for UNIT cost so we don’t see repeated "2K"
unit_thousand_fmt = FuncFormatter(lambda x, pos: f"{x / 1e6:.1f}M")
ax1.yaxis.set_major_formatter(unit_thousand_fmt)

# Keep integer K for total cost axes
thousand_fmt = FuncFormatter(lambda x, pos: f"{x / 1e6:.0f}M")
for ax in (ax2, ax_bl, ax_br):
    ax.yaxis.set_major_formatter(thousand_fmt)

# --- Unified legend ---
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
h3, l3 = ax_br.get_legend_handles_labels()

all_handles = h1 + h2 + h3
all_labels  = l1 + l2 + l3
unique = OrderedDict(zip(all_labels, all_handles))

for ax in (ax1, ax2, ax_bl, ax_br):
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

bold_font = FontProperties(weight='bold', size=22)

fig.legend(
    unique.values(), unique.keys(),
    loc='upper center',
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
    prop=bold_font
)

fig.subplots_adjust(bottom=0.22)

fig.savefig(f"{folder_name}/EDA_unitcost_coverage_frontiers.pdf", dpi=300)
plt.close(fig)



# -------------------- Weighted Scoring --------------------
weights = { 'total_cost': 0.611, 'coverage': 0.278, 'payload': 0.111}
scored_df = df.copy()
scored_df['norm_cost']     = scored_df['total_cost'].min() / scored_df['total_cost']
scored_df['norm_coverage'] = scored_df['coverage'] / scored_df['coverage'].max()
scored_df['norm_payload']  = scored_df['total_payload']  / scored_df['total_payload'].max()
scored_df['score'] = (
      weights['total_cost']  * scored_df['norm_cost']
    + weights['coverage']    * scored_df['norm_coverage']
    + weights['payload']     * scored_df['norm_payload']
)
raw_scores      = scored_df.groupby('optimizer')['score'].max()
friendly_scores = raw_scores.rename(index=nice)

# 2) open a multi‐page PDF
with PdfPages(f"{folder_name}/weighted_scores_two_page.pdf") as pp:

    # Page 1: the bar chart
    fig, ax = plt.subplots(figsize=(8,6))
    friendly_scores.plot(kind='bar', ax=ax, width=0.5)
    ax.margins(x=0)
    ax.set_xticklabels(friendly_scores.index, rotation=20, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel("Weighted Score", fontsize=16)
    ax.set_xlabel("Optimizer", fontsize=16)
    ax.set_title("Optimizer Scores", fontsize=18, pad=8)
    plt.tight_layout()
    pp.savefig(fig)    # saves page 1
    plt.close(fig)

    # Page 2: the table by itself
    fig, ax = plt.subplots(figsize=(8,6))
    ax.axis('off')     # hide the empty axes

    # prepare table data
    table_data = [[opt, f"{score:.3f}"] for opt, score in friendly_scores.items()]

    tbl = ax.table(
        cellText=table_data,
        colLabels=["Optimizer","Score"],
        cellLoc='center',
        colLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(1, 1.5)

    ax.set_title("Weighted Score Table", fontsize=18, pad=20)
    plt.tight_layout()
    pp.savefig(fig)    # saves page 2
    plt.close(fig)
