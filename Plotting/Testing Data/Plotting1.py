"""
In this file I am generating the box plots for test results
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator  # <- add near your imports
# ----------------------------
# AAAI/Matplotlib figure setup
# ----------------------------
# Single-column width ≈ 3.25 in; double-column ≈ 6.75 in.
FIG_WIDTH_SINGLE = 3.25
FIG_HEIGHT = 2.1  


# Enforce Arial 10 pt across the figure
mpl.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],  # will fall back if Arial not installed
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    # Embed TrueType fonts in PDF/PS
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Thin lines for print
    "lines.linewidth": 1.5,
})

# ----------------------------
# Load data
# ----------------------------
files = {
    "PHCAPAM": r"PHCAPAM_TestSample1_BestModel_results.xlsx",
    "CAPAM": r"CAPAM_TestSample1_BestModel_results.xlsx"
}

# Your colors (kept as-is). Consider using colorblind-safe palettes if needed.
model_colors = {
    "CAPAM":"#0ecef0",
    "PHCAPAM":  "#bf55a8"
}


#------------------------ REWARD TREND -------------------------------------
rewards_data = {}
for model, file_path in files.items():
    df = pd.read_excel(file_path)
    reward_cols = [c for c in df.columns if "reward" in c.lower()]
    if not reward_cols:
        raise ValueError(f"No reward column found in {file_path}")
    s = df[reward_cols[0]].astype(float)
    rewards_data[model] = s.reset_index(drop=True)

df = pd.DataFrame(rewards_data)
df.index.name = "Scenario"

# Optional: a lightly smoothed version for readability (small rolling window)
roll_win = max(3, len(df)//40)  # adapt window to number of scenarios
df_smooth = df.rolling(window=roll_win, center=True, min_periods=1).mean()

# ----------------------------
# Plot
# ----------------------------
fig_w = FIG_WIDTH_SINGLE
fig, ax = plt.subplots(figsize=(fig_w, FIG_HEIGHT))

# Plot raw lines (faint) + smoothed lines (solid) for readability in print
for model in df.columns:
    x = df.index.values
    # Raw dotted line
    ax.plot(
        x, df[model].values,
        linestyle=":",
        linewidth=1.0,
        alpha=0.4,
        color=model_colors.get(model, "gray"),
        zorder=1
    )
    
    # Smoothed solid line
    ax.plot(
        x, df_smooth[model].values,
        linestyle="-",
        linewidth=1.6,
        color=model_colors.get(model, "gray"),
        label=model,
        zorder=2
    )

# Axis labels and title (keep concise for AAAI)
ax.set_xlabel("Scenario")
ax.set_ylabel("Reward")

# # Ticks: reduce clutter; show ~8–10 ticks max
ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(8))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))

# Thin grid; helps the eye without overpowering
ax.grid(True, linewidth=0.4, alpha=0.4)
# Clean up spines
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Legend: compact, two columns if needed
handles, labels = ax.get_legend_handles_labels()

# Keep only smoothed legend entries (no "(raw)")
keep = [i for i, lab in enumerate(labels) if "(raw)" not in lab]

# Place legend at top center, above the plot
ax.legend(
    [handles[i] for i in keep],
    [labels[i] for i in keep],
    frameon=False,
    ncol=len(keep),                 # horizontal layout
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),     # move above plot area
    columnspacing=1.0,
    handlelength=2.0
)
plt.show()
fig.tight_layout(pad=0.1)

# # ----------------------------
# # Save
# # ----------------------------
fig.savefig("Reward_trend_Test1.pdf", bbox_inches="tight")



# # df should be shape (n_scenarios, n_models) from your earlier step
# # rows = scenarios, cols = ["CAPAM","PHCAPAM"]
# # We want rows=Models, cols=Scenarios -> transpose
# # H = df.T.values
# # models = list(df.columns)           # row labels
# # scenarios = np.arange(df.shape[0])  # column labels

# # # Consistent color scale across models
# # vmin, vmax = np.nanmin(H), np.nanmax(H)

# # fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE, FIG_HEIGHT))

# # im = ax.imshow(H, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)

# # # Axes labels & ticks (sparse x ticks for readability)
# # ax.set_title("Reward Heatmap (Test Sample 1)")
# # ax.set_xlabel("Scenario")
# # ax.set_ylabel("Model")

# # max_xticks = 8
# # tick_idx = np.linspace(0, len(scenarios)-1, num=min(max_xticks, len(scenarios))).astype(int)
# # ax.set_xticks(tick_idx)
# # ax.set_xticklabels([str(int(scenarios[i])) for i in tick_idx])
# # ax.set_yticks(np.arange(len(models)))
# # ax.set_yticklabels(models)

# # # Thin gridlines to separate columns (optional but nice in print)
# # ax.set_xticks(np.arange(-0.5, len(scenarios), 1), minor=True)
# # ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
# # ax.grid(which="minor", color="white", linewidth=0.4, alpha=0.6)
# # ax.tick_params(which="minor", bottom=False, left=False)

# # # Colorbar (compact)
# # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# # cbar.set_label("Reward", rotation=90)

# # fig.tight_layout(pad=0.1)
# # plt.show()
# # # Save vector + optional PNG
# # fig.savefig("Reward_heatmap_test1.pdf", bbox_inches="tight")

#------------------------ MODEL WINRATE ------------------------------------- 
# Calculate win counts: the number of scenarios in which each model had the highest reward
# Find model with highest reward per scenario
best_model_per_scenario = df.idxmax(axis=1)

# Count how many times each model was best
best_model_counts = best_model_per_scenario.value_counts().sort_values(ascending=False)

# Convert to DataFrame for plotting
win_counts_df = best_model_counts.to_frame(name="Win Count")

# --- Plotting (per-bar colors + horizontal labels) ---
fig, ax = plt.subplots(figsize=(fig_w, FIG_HEIGHT))

models = win_counts_df.index.tolist()
counts = win_counts_df["Win Count"].to_numpy()
bar_colors = [model_colors.get(m, "gray") for m in models]

bars = ax.bar(models, counts, color=bar_colors, edgecolor="black", linewidth=0.3)

ax.set_ylabel("Count")
ax.set_xlabel("Model")

# Horizontal x labels
ax.tick_params(axis="x", labelrotation=0)

# Light y-grid
ax.grid(axis="y", linewidth=0.4, alpha=0.4)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# Optional: annotate counts + %
total = counts.sum()
for rect, c in zip(bars, counts):
    ax.text(rect.get_x() + rect.get_width()/2,
            rect.get_height() + 0.02*max(counts.max(), 1),
            f"{int(c)}",
            ha="center", va="bottom", fontsize=10)

plt.tight_layout()
fig.savefig("Modelwinrate_test1.pdf", bbox_inches='tight')

plt.show()



# Violin Plot
# --- Build df_melted for the violin plot ---
df_plot = df.copy()

# If earlier you used "PH" as the column name, normalize it:
if "PH" in df_plot.columns and "PHCAPAM" not in df_plot.columns:
    df_plot = df_plot.rename(columns={"PH": "PHCAPAM"})

# Ensure we have a Scenario column when melting
if df_plot.index.name != "Scenario":
    df_plot.index.name = "Scenario"

df_melted = df_plot.reset_index().melt(
    id_vars="Scenario", var_name="Model", value_name="Reward"
)

order = [m for m in ["PHCAPAM", "CAPAM"] if m in df_melted["Model"].unique()]
if not order:
    order = list(df_melted["Model"].unique())

palette = {m: model_colors.get(m, "gray") for m in order}

fig, ax = plt.subplots(figsize=(fig_w, FIG_HEIGHT))  # single-column

sns.violinplot(
    x="Model",
    y="Reward",
    data=df_melted,
    order=order,
    palette=palette,
    inner="quartile",   # show quartiles
    cut=0,
    linewidth=0.8,
    ax=ax
)

# Overlay mean as white diamond
for j, m in enumerate(order):
    mean_val = df_melted.loc[df_melted["Model"] == m, "Reward"].mean()
    ax.scatter(j, mean_val, color="white", edgecolor="black",
               marker="D", s=15, zorder=3)

# Labels and grid
ax.set_xlabel("Model")
ax.set_ylabel("Reward")
ax.tick_params(axis="x", labelrotation=0)
ax.grid(True, axis="y", linewidth=0.8, alpha=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout(pad=0.1)
fig.savefig("Violin_plot_test1.pdf", bbox_inches="tight")
plt.show()