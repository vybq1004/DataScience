"""
5024CMD Data Science - Step 1: Initial Data Quality Check
"""
import pandas as pd

# pd.set_option() changes how pandas displays output:
# show every column and allow wide lines so no attribute is hidden
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# pd.read_csv() loads the 50-record sample created by Step 0 into a DataFrame (a table of rows and columns)
df = pd.read_csv("bank_sample.csv")

# ---------- Part 1: dataset size and first rows ----------
print("=== 1. Dataset shape (rows, columns) ===")
# .shape returns the number of rows and columns as (rows, columns)
print(df.shape)
print("\n=== First 5 records ===")
# .head() returns the first 5 rows of the DataFrame
print(df.head())

# ---------- Part 2: data types ----------
print("\n=== 2. Data types of each attribute ===")
# .dtypes returns the data type of each column (e.g. int64 = whole number, str/object = text)
print(df.dtypes)

# ---------- Part 3: missing values and duplicates ----------
print("\n=== 3a. Missing values (NaN) per attribute ===")
# .isna() marks each empty cell as True; .sum() then counts the True values in each column
print(df.isna().sum())
# a second .sum() adds the column counts together to give one overall total
print("\nTotal missing values:", df.isna().sum().sum())
print("\n=== 3b. Duplicate records ===")
# .duplicated() marks each row that is an exact copy of an earlier row; .sum() counts them
print("Number of duplicate rows:", df.duplicated().sum())

# ---------- Part 4: hidden missing values and special codes ----------
print("\n=== 4a. 'unknown' values per categorical attribute ===")
# .select_dtypes(exclude="number") keeps only the non-numeric (text) columns
categorical = df.select_dtypes(exclude="number").columns
# (df[c] == "unknown") marks cells equal to "unknown"; .sum() counts them and .mean() gives the proportion
# round() rounds the percentage to 1 decimal place
# pd.DataFrame() builds a small results table from these counts
unknown = pd.DataFrame({
    "unknown_count": [(df[c] == "unknown").sum() for c in categorical],
    "unknown_percent": [round((df[c] == "unknown").mean() * 100, 1) for c in categorical],
}, index=categorical)
# keep only the columns that actually contain "unknown"
print(unknown[unknown["unknown_count"] > 0])

print("\n=== 4b. Special code pdays = -1 (never contacted before) ===")
# (df["pdays"] == -1) marks customers never contacted before; .sum() counts them
count = (df["pdays"] == -1).sum()
# len(df) returns the total number of rows, used to work out the percentage
print(f"pdays = -1: {count} records ({count / len(df) * 100:.1f}%)")

# ---------- Part 5: summary statistics of numeric attributes ----------
print("\n=== 5. Summary statistics of numeric attributes ===")
# .describe() returns count, mean, standard deviation, min, quartiles and max for numeric columns
# .round(2) rounds the results to 2 decimal places
print(df.describe().round(2))
