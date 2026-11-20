"""
5024CMD Data Science - Step 2: Handling Missing Values
"""
import pandas as pd

# pd.set_option() changes how pandas displays output: show every column on one line
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# pd.read_csv() loads the 50-record sample created by Step 0 into a DataFrame (a table of rows and columns)
df = pd.read_csv("bank_sample.csv")

# ---------- Part 1: drop the attributes excluded in Section 1.6 ----------
print("=== 1. Columns before dropping ===")
# .shape[1] returns the number of columns
print("Number of columns:", df.shape[1])

excluded = ["job", "marital", "education", "default", "loan", "contact",
            "day", "month", "duration", "pdays", "previous", "poutcome"]
# .drop(columns=...) removes the listed columns and returns a new DataFrame
df_selected = df.drop(columns=excluded)

print("\nColumns after dropping:", df_selected.shape[1])
# .columns.tolist() returns the remaining column names as a list
print("Remaining attributes:", df_selected.columns.tolist())

# ---------- Part 2: check the selected attributes for missing values ----------
print("\n=== 2a. Empty (NaN) values in selected attributes ===")
# .isna() marks empty cells as True; .sum() counts them per column
print(df_selected.isna().sum())

print("\n=== 2b. 'unknown' values in selected attributes ===")
# (df_selected == "unknown") marks every cell equal to "unknown"; .sum() counts them per column
print((df_selected == "unknown").sum())

print("\n=== 2c. Categories in the categorical attributes ===")
# .unique() returns the distinct values in a column, to confirm no unexpected codes
print("housing:", df_selected["housing"].unique().tolist())
print("deposit:", df_selected["deposit"].unique().tolist())

# ---------- Part 3: check numeric attributes for impossible or special values ----------
print("\n=== 3. Range of numeric attributes ===")
# .min() and .max() return the smallest and largest value in each column
print(df_selected[["age", "balance", "campaign"]].agg(["min", "max"]))
# (df_selected["balance"] < 0) marks negative balances; .sum() counts them
print("\nCustomers with a negative balance (overdrawn accounts):", (df_selected["balance"] < 0).sum())

# ---------- Save the result for the next step ----------
# .to_csv() writes the DataFrame to a new CSV file; index=False stops pandas adding a row-number column
df_selected.to_csv("bank_selected.csv", index=False)
print("\nSaved bank_selected.csv with", df_selected.shape[0], "rows and", df_selected.shape[1], "columns")
