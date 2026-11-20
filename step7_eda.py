"""
5024CMD Data Science - Step 7: Exploratory Data Analysis (EDA)
"""
import os
# limit OpenBLAS to one thread to avoid "Memory allocation failed" errors on some PCs
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr
import matplotlib
matplotlib.use("Agg")  # draw charts straight to image files (no pop-up window)
import matplotlib.pyplot as plt

# pd.set_option() shows every column when printing tables
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
# os.makedirs() creates the "figures" folder if it does not exist yet
os.makedirs("figures", exist_ok=True)

# pd.read_csv() loads the cleaned data (Step 3) and the binned data (Step 4)
df = pd.read_csv("bank_clean.csv")
binned = pd.read_csv("bank_binned.csv")
numeric = ["age", "balance", "campaign"]

# ---------- Part 1: summary statistics ----------
print("=== 1a. Summary statistics of numeric attributes ===")
# .describe() gives count, mean, std, min, quartiles and max; .round(2) rounds to 2 decimals
print(df[numeric].describe().round(2))
# .skew() measures how lopsided a distribution is (0 = symmetrical, > 1 = strongly right-skewed)
print("\nSkewness:")
print(df[numeric].skew().round(2))
print("\n=== 1b. Frequency of categorical attributes ===")
# .value_counts() counts each category
print(df["housing"].value_counts())
print(df["deposit"].value_counts())

# ---------- Part 2: univariate charts ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, numeric):
    # .hist() draws a histogram showing how often values fall in each range
    ax.hist(df[col], bins=10, edgecolor="black")
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Number of customers")
plt.tight_layout()
plt.savefig("figures/histograms_numeric.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, col in zip(axes, ["housing", "deposit"]):
    counts = df[col].value_counts().sort_index()
    # .bar() draws a bar chart of the category counts
    ax.bar(counts.index, counts.values, edgecolor="black")
    ax.set_title(f"Customers by {col}")
    ax.set_ylabel("Number of customers")
plt.tight_layout()
plt.savefig("figures/bars_categorical.png", dpi=150)
plt.close()

# ---------- Part 3: subscription rate by each input ----------
binned["deposit_num"] = (binned["deposit"] == "yes").astype(int)
orders = {"age_group": ["young", "adult", "middle_aged", "senior"],
          "balance_band": ["negative", "low", "medium", "high"],
          "housing": ["no", "yes"],
          "campaign_band": ["single", "few", "many"]}
print("\n=== 3. Subscription rate (%) by input ===")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (col, order) in zip(axes, orders.items()):
    # .groupby() splits customers by category; .mean() of deposit_num gives the share who subscribed
    rate = binned.groupby(col)["deposit_num"].agg(["count", "mean"]).reindex(order)
    rate["rate_percent"] = (rate["mean"] * 100).round(1)
    print(f"\n{col}:")
    print(rate[["count", "rate_percent"]])
    ax.bar(order, rate["rate_percent"], edgecolor="black")
    ax.set_title(f"Subscription rate by {col}")
    ax.set_ylabel("% subscribed")
    ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("figures/subscription_rate_by_input.png", dpi=150)
plt.close()

# ---------- Part 4: numeric inputs split by deposit ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, numeric):
    # one box for customers who did not subscribe and one for those who did
    ax.boxplot([df.loc[df["deposit"] == "no", col], df.loc[df["deposit"] == "yes", col]])
    ax.set_xticks([1, 2], labels=["no", "yes"])
    ax.set_title(f"{col} by deposit")
    ax.set_xlabel("deposit")
plt.tight_layout()
plt.savefig("figures/boxplots_by_deposit.png", dpi=150)
plt.close()

# .scatter() plots each customer as a point; colour shows whether they subscribed
fig, ax = plt.subplots(figsize=(6, 5))
for label, colour in [("no", "tab:red"), ("yes", "tab:blue")]:
    part = df[df["deposit"] == label]
    ax.scatter(part["age"], part["balance"], c=colour, label=f"deposit = {label}", alpha=0.7)
ax.set_xlabel("age")
ax.set_ylabel("balance (€)")
ax.set_title("Age vs balance, coloured by deposit")
ax.legend()
plt.tight_layout()
plt.savefig("figures/scatter_age_balance.png", dpi=150)
plt.close()

# ---------- Part 5: statistical relationships with deposit ----------
# p_values stores the p-value of every test so they can be charted in Part 6
p_values = {}
print("\n=== 5a. Correlation of numeric inputs with deposit (point-biserial) ===")
for col in numeric:
    # pointbiserialr() measures the correlation between a yes/no variable and a numeric one
    r, p = pointbiserialr(df["deposit_num"], df[col])
    print(f"{col}: r = {r:.2f}, p-value = {p:.3f}")
    p_values[col] = p

print("\n=== 5b. Chi-square test of each categorical input against deposit ===")
for col in orders:
    # pd.crosstab() counts customers in each combination of the two categories
    table = pd.crosstab(binned[col], binned["deposit"])
    # chi2_contingency() tests whether the two categories are related (p < 0.05 = significant)
    chi2, p, dof, _ = chi2_contingency(table)
    print(f"{col}: chi-square = {chi2:.2f}, df = {dof}, p-value = {p:.3f}")
    p_values[col] = p

# ---------- Part 6: chart of the p-values ----------
# sorted() puts the smallest p-value at the top of the chart
names = sorted(p_values, key=p_values.get, reverse=True)
values = [p_values[n] for n in names]
fig, ax = plt.subplots(figsize=(7, 4))
# ax.barh() draws one horizontal bar per test
ax.barh(names, values, color="#4C72B0", edgecolor="black")
# ax.axvline() draws the 0.05 significance line
ax.axvline(0.05, color="red", linestyle="--", label="0.05 significance level")
for i, v in enumerate(values):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center")
ax.set_xlim(0, 1.05)
ax.set_xlabel("p-value (left of the red line = significant)")
ax.set_title("p-value of each input tested against deposit")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("figures/p_values.png", dpi=150)
plt.close()

print("\nSaved charts in the figures folder")