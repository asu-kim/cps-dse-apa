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
        total_coverage = config['coverage'] * config['quantity']
        per_unit_payload = config['payload']
        payload = config['payload'] * config['quantity']
        quantity = config['quantity']
        edge_cost = config['edge_server']['cost']
        per_unit_cost = config['base_cost'] + config['additional_cost'] + (edge_cost / quantity)
        metrics.append({
            'optimizer': opt,
            'total_cost': config['total_cost'],
            'per_unit_cost': per_unit_cost,
            'coverage': total_coverage,
            'per_unit_payload': per_unit_payload,
            'payload': payload,
            'quantity': quantity,
            'runtime': config['runtime_hours']
        })

df = pd.DataFrame(metrics)

# -------------------- Frontier Plot --------------------
left_end    = df[df.optimizer=='payload_cost']['coverage'].max() * 1.05
gap_pct     = 0.0001
right_start = farm_size * (1 - gap_pct)
right_end   = df['coverage'].max() * 1.00005

y_low_max   = budget * 1.1
y_high_min  = budget * 5
y_high_max  = budget * 10

# 2) master 1×2 grid
fig = plt.figure(figsize=(16,13))
master = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.3)

# 3) Left panel: Cost vs. Payload
ax1 = fig.add_subplot(master[0])
for opt in optimizers:
    sub = df[df.optimizer==opt]
    m, fc, ec = markers[opt], (colors[opt] if markers[opt] in ['+','x'] else 'none'), colors[opt]
    ax1.scatter(sub['payload'], sub['total_cost'],
                label=nice[opt], marker=m,
                facecolors=fc, edgecolors=ec,
                s=200, alpha=0.8, linewidths=1.5)
ax1.axhline(budget, color='blue', linestyle='--', label='Budget Constraint')

ax1.set_xlabel("Payload (kg)", fontsize=22)
ax1.set_ylabel("Cost in millions ($)", fontsize=22)
ax1.set_title("Cost vs. Payload (a)", fontsize=22, fontweight='bold', pad=10)
ax1.set_ylim(0, budget*1.25)

# 4) Right panel: nested 2×2 broken X & Y
subspec = gridspec.GridSpecFromSubplotSpec(2, 2,
    subplot_spec=master[1],
    width_ratios =[1,2],
    height_ratios=[1,2],
    wspace=0.1, hspace=0.1
)

ax_tl = fig.add_subplot(subspec[0,0], sharex=None, sharey=None)
ax_tr = fig.add_subplot(subspec[0,1], sharey=ax_tl)
ax_bl = fig.add_subplot(subspec[1,0], sharex=ax_tl, sharey=None)
ax_br = fig.add_subplot(subspec[1,1], sharex=ax_tr, sharey=ax_bl)

# scatter into each quadrant (only bottom-left carries the optimizer labels)
for opt in optimizers:
    sub = df[df.optimizer==opt]
    x, y = sub['coverage'], sub['total_cost']
    m, fc, ec = markers[opt], (colors[opt] if markers[opt] in ['+','x'] else 'none'), colors[opt]

    # bottom-left
    mask = (x <= left_end) & (y <= y_low_max)
    ax_bl.scatter(x[mask], y[mask], label=nice[opt], marker=m,
                  facecolors=fc, edgecolors=ec, s=200, alpha=0.8, linewidths=1.5)
    # bottom-right (no label)
    mask = (x >= right_start) & (y <= y_low_max)
    ax_br.scatter(x[mask], y[mask], marker=m,
                  facecolors=fc, edgecolors=ec, s=200, alpha=0.8, linewidths=1.5)
    # top-left
    mask = (x <= left_end) & (y >= y_high_min)
    ax_tl.scatter(x[mask], y[mask], marker=m,
                  facecolors=fc, edgecolors=ec, s=200, alpha=0.8, linewidths=1.5)
    # top-right
    mask = (x >= right_start) & (y >= y_high_min)
    ax_tr.scatter(x[mask], y[mask], marker=m,
                  facecolors=fc, edgecolors=ec, s=200, alpha=0.8, linewidths=1.5)

# draw constraints (only label farm-size once on ax_br)
for ax in (ax_tl, ax_tr, ax_bl, ax_br):
    ax.axhline(budget, color='blue', linestyle='--')              # budget
    ax.axvline(farm_size, color='black', linestyle='--', label='Farm Size Constraint')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

# set limits
ax_bl.set_xlim(0, left_end);      ax_bl.set_ylim(0, y_low_max)
ax_br.set_xlim(right_start, right_end); ax_br.set_ylim(0, y_low_max)
ax_tl.set_xlim(0, left_end);      ax_tl.set_ylim(y_high_min, y_high_max)
ax_tr.set_xlim(right_start, right_end); ax_tr.set_ylim(y_high_min, y_high_max)

millions_fmt = FuncFormatter(lambda x, pos: f"{x/1e6:.1f}M")
for ax in (ax1, ax_tl, ax_tr, ax_bl, ax_br):
    ax.yaxis.set_major_formatter(millions_fmt)

thousands = FuncFormatter(lambda x, pos: f"{x/1e3:.0f}")
ax1.xaxis.set_major_locator(MultipleLocator(5000))
ax1.xaxis.set_major_formatter(thousands)

# label only the outer axes of the broken plot
ax_bl.set_xlabel("Area coverage (m²)", fontsize=22)
ax_br.set_xlabel("Area coverage (m²)", fontsize=22)
ax_bl.set_ylabel("Cost in millions ($)", fontsize=22)

ax_tr.set_title("Cost vs. Area Coverage (b)", fontsize=22, fontweight='bold', pad=10)

# little diagonal break marks
d = .01
for ax in (ax_bl,ax_br):
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d,1+d), (-d, +d), **kwargs)
    ax.plot((1-d,1+d), (1-d,1+d), **kwargs)
    ax.plot((-d,+d), (1-d,1+d), **kwargs)
    ax.plot((1-d,+d), (1-d,1+d), **kwargs)

for ax in (ax_bl, ax_br):
    ax.tick_params(labelbottom=True)

for ax in (ax_tr, ax_br):
    ax.tick_params(axis='y', labelleft=False)
    ax.set_ylabel('')

ax_tl.tick_params(labelbottom=False)
ax_tr.tick_params(labelbottom=False)

# 5) build one unified legend (avoid duplicates)
h1, l1 = ax1.get_legend_handles_labels()  # optimizers + budget
h2, l2 = ax_br.get_legend_handles_labels()  # farm-size
all_handles = h1 + h2
all_labels  = l1 + l2
unique = OrderedDict(zip(all_labels, all_handles))

ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)

bold_font = FontProperties(weight='bold', size=22)

fig.legend(
    unique.values(), unique.keys(),
    loc='upper center',
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.01), prop=bold_font
)

#plt.tight_layout(rect=[0,0,1,0.95])
fig.savefig(f"{folder_name}/EDA_combined_broken_frontiers.pdf", dpi=300)
plt.close(fig)

# -------------------- Weighted Scoring --------------------
weights = { 'total_cost': 0.611, 'coverage': 0.278, 'payload': 0.111}
scored_df = df.copy()
scored_df['norm_cost']     = scored_df['total_cost'].min() / scored_df['total_cost']
scored_df['norm_coverage'] = scored_df['coverage'] / scored_df['coverage'].max()
scored_df['norm_payload']  = scored_df['payload']  / scored_df['payload'].max()
scored_df['score'] = (
      weights['total_cost']  * scored_df['norm_cost']
    + weights['coverage']    * scored_df['norm_coverage']
    + weights['payload']     * scored_df['norm_payload']
)
raw_scores     = scored_df.groupby('optimizer')['score'].max()
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