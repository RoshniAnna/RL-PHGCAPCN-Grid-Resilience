import pandas as pd
import matplotlib.pyplot as plt

# --- Load the reward data from CSV ---
file_path = "PHCAPAM_reward_curve.csv"  # Update if path differs
df = pd.read_csv(file_path)

# --- Compute moving average ---
window_size = 100  # You can adjust this for smoother or more detailed curves
df["moving_avg"] = df["reward"].rolling(window=window_size).mean()


# --- Set font to Arial size 10 ---
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10
})

# --- Plot ---
fig_width = 3  # inches
fig_height = 2.5  

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.plot(df["step"], df["reward"], color="pink", label="Raw reward", linewidth=1)
ax.plot(df["step"], df["moving_avg"], color="darkmagenta", label=f"Moving average", linewidth=2.5)

ax.set_xlabel("Step")
ax.set_ylabel("Mean episode reward")
ax.legend()
ax.grid(True)
ax.set_xlim(left=0)  # Start X-axis at 0
ax.set_ylim(-0.5, 0.6)
plt.tight_layout()

# # --- Save as PDF ---
plt.savefig("phcapam_reward_curve.pdf", format="pdf", dpi=300, bbox_inches='tight')
plt.show()
