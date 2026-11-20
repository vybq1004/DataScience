"""
5024CMD Data Science - Step 3: Duplicates, Data Types and Outliers
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw charts straight to image files (no pop-up window)
import matplotlib.pyplot as plt

# pd.set_option() shows every column when printing tables
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# os.makedirs() creates the "figures" folder for saved charts; exist_ok=True avoids an error if it already exists
os.makedirs("figures", exist_ok=True)

# pd.read_csv() loads the 5-attribute dataset produced by Step 2
df = pd.read_csv("bank_selected.csv")

# ---------- Part 1: duplicate records ----------
print("=== 1. Duplicate records ===")
# .duplicated() marks rows that are exact copies of an earlier row; .sum() counts them
print("Number of duplicate rows:", df.duplicated().sum())

# ---------- Part 2: data types ----------
print("\n=== 2a. Data types before conversion ===")
# .dtypes shows the data type of each column
print(df.dtypes)

# .astype("category") stores yes/no text columns as categories (a fixed set of labels)
df["housing"] = df["housing"].astype("category")
df["deposit"] = df["deposit"].astype("category")
# .map() creates numeric copies (yes = 1, no = 0) for correlation and modelling
df["housing_num"] = df["housing"].map({"yes": 1, "no": 0}).astype(int)
df["deposit_num"] = df["deposit"].map({"yes": 1, "no": 0}).astype(int)

print("\n=== 2b. Data types after conversion ===")
print(df.dtypes)
print("\nFirst 5 records after conversion:")
# .head() shows the first 5 rows
print(df.head())

# ---------- Part 3: outliers (IQR method) ----------
print("\n=== 3. Outliers using the IQR method ===")
numeric = ["age", "balance", "campaign"]
rows = []
for col in numeric:
    # .quantile() returns the value below which a given share of the data falls (25% and 75%)
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    # count the values that fall outside the lower and upper limits
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    rows.append({"attribute": col, "Q1": q1, "Q3": q3, "IQR": iqr,
                 "lower_limit": lower, "upper_limit": upper, "outliers": n_out})
# pd.DataFrame() builds a results table from the list of dictionaries
print(pd.DataFrame(rows).round(2).to_string(index=False))

# plt.subplots() creates one figure with three side-by-side charts
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col in zip(axes, numeric):
    # .boxplot() draws a box plot; points beyond the whiskers are potential outliers
    ax.boxplot(df[col])
    ax.set_title(col)
fig.suptitle("Box plots of the numeric attributes (50-record sample)")
# plt.tight_layout() stops labels overlapping; plt.savefig() saves the chart as an image
plt.tight_layout()
plt.savefig("figures/boxplots_numeric.png", dpi=150)
plt.close()
print("\nSaved figures/boxplots_numeric.png")

# All outliers are kept (they are genuine values, see report Section 2.5)
# .to_csv() saves the cleaned dataset for the next steps
df.to_csv("bank_clean.csv", index=False)
print("Saved bank_clean.csv with", df.shape[0], "rows and", df.shape[1], "columns")
