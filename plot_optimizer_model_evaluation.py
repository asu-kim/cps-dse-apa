import json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.express as px
import numpy as np
import os
import shutil
import re
import kaleido
from collections import OrderedDict
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from matplotlib.font_manager import FontProperties

# -------------------- Load the JSON file --------------------

with open("results.json", "r") as f:
    data = json.load(f)

mpl.rcParams['text.usetex'] = False
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

folder_name = f"Plots/Optimizer_OM_evaluation/{crop}_{budget}_{farm_size}_{apps_str}"

os.makedirs(folder_name, exist_ok=True)
shutil.copy("results.json", f"{folder_name}/results_OM.json")


# -------------------- Flatten Optimizers --------------------

nice = {
    'proposed_approach':    'Proposed Approach',
    'simulated_annealing':  'Simulated Annealing (DESTION\'23)',
    'bayesian':             'Bayesian Optimization (DESTION\'23)',
    'random_search':        'Random Search (DESTION\'22)',
    'genetic_algorithm':    'Genetic Algorithm (DESTION\'22)',
    'pg_dse':               'PG-DSE (ASP-DAC\'23)',
    'discrete':             'Discrete Search (DESTION\'22)',
    'lengler':              'Lengler (DESTION\'22)',
    'portfolio':            'Portfolio (DESTION\'22)'
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
colors = {opt: cmap(i) for i,opt in enumerate(nice)}

for opt in nice:
    raw = data[opt]
    raw = fully_flatten(raw)
    flattened[opt] = raw

markers = {
    'proposed_approach': 's',
    'simulated_annealing': '+',
    'bayesian':'p',
    'random_search': '^',
    'genetic_algorithm': 'D',
    'pg_dse':'x',
    'discrete':'v',
    'lengler':'o',
    'portfolio':'*'
}

# -------------------- Prepare Metrics --------------------

metrics = []
for opt in nice:
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
MARKER_SIZE = 750
fig, (ax1, ax2) = plt.subplots(1, 2,
    figsize=(20,12),
    sharey=True,
    gridspec_kw={'width_ratios': [1, 1]}
)
fig.subplots_adjust(wspace=0.1, top=0.75, bottom=0.15,left=0.38, right=0.98)
min_cost = df['total_cost'].min()
y_lower  = min_cost
y_upper  = budget * 1.05

# Cost vs. Payload on ax1
for opt in nice:
    sub = df[df.optimizer==opt]
    m = markers[opt]
    fc = colors[opt] if m in ['+','x'] else 'none'
    ec = colors[opt]
    ax1.scatter(
        sub['payload'], sub['total_cost'],
        label=nice[opt],
        marker=m,
        facecolors=fc,
        edgecolors=ec,
        s=MARKER_SIZE,
        alpha=0.8,
        linewidths=1.5,
        zorder=4
    )
ax1.axhline(budget, color='blue', linestyle='--', label='Budget')
ax1.set_xlabel("Payload in Thousands (Kg)", fontsize=22, labelpad=25)
ax1.set_ylabel("Total Cost in Millions ($)", fontsize=22)
# ax1.set_title("Cost vs. Payload (a)", fontsize=22, pad=5, fontweight='bold')
ax1.text(
    0.5, -0.22, "(a) Total Cost vs. Payload",
    transform=ax1.transAxes,
    ha="center", va="top",
    fontsize=20, fontweight="bold"
)
ax1.set_ylim(0, budget * 1.5)

# Cost vs. Coverage on ax2
for opt in nice:
    sub = df[df.optimizer==opt]
    m = markers[opt]
    fc = colors[opt] if m in ['+','x'] else 'none'
    ec = colors[opt]
    ax2.scatter(
        sub['coverage'], sub['total_cost'],
        label=nice[opt],
        marker=m,
        facecolors=fc,
        edgecolors=ec,
        s=MARKER_SIZE,
        alpha=0.8,
        linewidths=1.5,
        zorder=4
    )
ax2.axhline(budget, color='blue', linestyle='--', label='Budget')
ax2.axvline(farm_size, color='red', linestyle='--', label='Farm Size',linewidth=2, zorder=15)
ax2.set_xlabel("Area coverage in Thousands (m²)", fontsize=22, labelpad=25)
# ax2.set_title("Cost vs. Area Coverage (b)", fontsize=22, pad=80, fontweight='bold')
ax2.text(
    0.5, -0.22, "(b) Total Cost vs Area Coverage",
    transform=ax2.transAxes,
    ha="center", va="top",
    fontsize=22, fontweight="bold"
)
ax2.set_ylim(0, budget * 1.5)

# pull a single legend from ax1 (or ax2—they have the same handles)
h1, l1 = ax1.get_legend_handles_labels()  # optimizers + budget
h2, l2 = ax2.get_legend_handles_labels()  # farm-size
all_handles = h1 + h2
all_labels  = l1 + l2
unique = OrderedDict(zip(all_labels, all_handles))


x1_min, x1_max = ax1.get_xlim()
x2_min, x2_max = ax2.get_xlim()



# ax2.text(farm_size - (x2_max - x2_min) * 0.1, y_upper * 1.035, "(i) Insufficient area\ncoverage",
#          fontsize=22, ha='right')
#
# ax2.text(farm_size + (x2_max - x2_min) * 0.1, y_upper * 1.035, "(ii) Sufficient area\ncoverage",
#          fontsize=22,ha='left')


ax2.annotate(
    '',
    xy=(farm_size, y_upper * 1.015),      # arrow head at farm size line
    xytext=(3000, y_upper * 1.015),        # arrow tail at x=3000
    arrowprops=dict(
        arrowstyle='->',
        linestyle=':',
        linewidth=2,
        color='black'
    ),
    zorder=10
)

# place a single legend at the bottom of the figure
bold_font = FontProperties(weight='bold', size=22)
fig.legend(
    unique.values(), unique.keys(),
    loc='upper center',
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.65, 1.02), prop=bold_font
)

for ax in (ax1, ax2):
    plt.setp(ax.get_xticklabels(), ha='right')
    ax.set_ylim(y_lower, y_upper)

millions_fmt = FuncFormatter(lambda x, pos: f"{x / 1e6:.1f}M")
unit_thousand_fmt = FuncFormatter(lambda x, pos: f"{x / 1e3:.1f}K")
for ax in (ax1, ax2):
    ax.yaxis.set_major_formatter(millions_fmt)

# ----- X-axis for Cost vs. Payload (ax1) -----
# x data is in kg; we show "in thousands (Kg)"
ax1.set_xlim(20000, 50000)  # 0–40k kg  (change as needed)
ax1.xaxis.set_major_locator(MultipleLocator(10000))  # ticks every 4k
ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1e3:.0f}K"))

# ----- X-axis for Cost vs. Area Coverage (ax2) -----
# x data is in m²; we show "in thousands (m²)"
ax2.set_xlim(26000, 69000)  # 0–12k m²  (change as needed)
ax2.xaxis.set_major_locator(MultipleLocator(13000))  # ticks every 2k
ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1e3:.0f}K"))

# for ax in (ax1, ax2):
#     ax2.xaxis.set_major_locator(MultipleLocator(1000)) #500 vs. 10000
#     ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1e3:.0f}"))
#
# thousands = FuncFormatter(lambda x, pos: f"{x/1e3:.0f}")
# ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # fewer, nice integer ticks
# ax1.xaxis.set_major_formatter(thousands)
#
# for ax in (ax1, ax2):
#     _, xmax = ax.get_xlim()
#     ax.set_xlim(3000, xmax)

ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
fig.subplots_adjust(bottom=0.22)

#plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{folder_name}/OM_combined_frontiers_3.pdf", dpi=300)
plt.close(fig)
# -------------------- Frontier Plot --------------------

# -------------------- Weighted Scoring --------------------
full_order = [
  'proposed_approach','simulated_annealing','bayesian','random_search',
  'genetic_algorithm','pg_dse','discrete','lengler','portfolio'
]

weights = {
    'total_cost': 0.611,
    'coverage':   0.278,
    'payload':    0.111,
}

scored_df = df.copy()
scored_df['norm_cost']     = scored_df['total_cost'].min() / scored_df['total_cost']
scored_df['norm_coverage'] = scored_df['coverage'] / scored_df['coverage'].max()
scored_df['norm_payload']  = scored_df['payload']  / scored_df['payload'].max()

scored_df['score'] = (
      weights['total_cost']  * scored_df['norm_cost']
    + weights['coverage']    * scored_df['norm_coverage']
    + weights['payload']     * scored_df['norm_payload']
)


raw_scores = scored_df.groupby('optimizer')['score'].max()
raw_scores = raw_scores.reindex(full_order, fill_value=0.0)
friendly_scores = raw_scores.rename(index=nice)

fig, ax = plt.subplots(figsize=(12, 12))
friendly_scores.plot(kind='bar', ax=ax, width=0.5)

with PdfPages(f"{folder_name}/OM_weighted_score_two_page.pdf") as pp:

    # ── Page 1: bar chart ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    friendly_scores.plot(kind='bar', ax=ax, width=0.5)

    ax.set_xticklabels(friendly_scores.index, rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel("Weighted Score", fontsize=18)
    ax.set_xlabel("Optimizer", fontsize=18)
    ax.set_title("Optimizer Scores", fontsize=20, pad=10)

    plt.tight_layout()
    pp.savefig(fig)   # save bar chart page
    plt.close(fig)

    # ── Page 2: standalone table ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')    # no axes

    table_data = [[opt, f"{score:.3f}"] for opt, score in friendly_scores.items()]
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Optimizer", "Score"],
        cellLoc='center',
        colLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(1, 1.5)

    ax.set_title("Weighted Score Table", fontsize=20, pad=20)
    plt.tight_layout()
    pp.savefig(fig)   # save table page
    plt.close(fig)