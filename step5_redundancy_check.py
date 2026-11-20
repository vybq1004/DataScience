"""
5024CMD Data Science - Step 5: Dimensionality Reduction - Redundancy Check
Confirms that the four selected inputs do not repeat each other's information.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw charts straight to image files (no pop-up window)
import matplotlib.pyplot as plt

# pd.read_csv() loads the cleaned dataset produced by Step 3
df = pd.read_csv("bank_clean.csv")
inputs = ["age", "balance", "housing_num", "campaign"]

# .corr() calculates the correlation between every pair of columns
# method="spearman" uses ranks, so it is less affected by outliers such as very high balances
pearson = df[inputs].corr(method="pearson").round(2)
spearman = df[inputs].corr(method="spearman").round(2)
print("=== 1. Pearson correlation between the four inputs ===")
print(pearson)
print("\n=== 2. Spearman correlation between the four inputs ===")
print(spearman)

# Find the strongest correlation between two different inputs
# .abs() makes all values positive; .where() hides the diagonal (a column with itself is always 1)
mask = ~pd.DataFrame(
    [[i == j for j in inputs] for i in inputs], index=inputs, columns=inputs)
strongest = spearman.abs().where(mask).stack().max()
print("\nStrongest correlation between two different inputs (absolute Spearman):", strongest)
print("Rule used: |r| >= 0.7 means two inputs are redundant and one could be removed.")

# plt.imshow() draws the correlation table as a coloured grid (heatmap)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(spearman, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(inputs)), labels=inputs, rotation=45)
ax.set_yticks(range(len(inputs)), labels=inputs)
# ax.text() writes each correlation value inside its cell
for i in range(len(inputs)):
    for j in range(len(inputs)):
        ax.text(j, i, spearman.iloc[i, j], ha="center", va="center")
# plt.colorbar() adds the colour scale legend
plt.colorbar(im)
ax.set_title("Spearman correlation between the four inputs")
plt.tight_layout()
plt.savefig("figures/redundancy_heatmap.png", dpi=150)
plt.close()
print("\nSaved figures/redundancy_heatmap.png")
