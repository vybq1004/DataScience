"""
5024CMD Data Science - Step 4: Numeric to Categorical (Binning) and CSV to ARFF
"""
import pandas as pd
import arff  # liac-arff library, install with: pip install liac-arff

# pd.set_option() shows every column when printing tables
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# pd.read_csv() loads the cleaned dataset produced by Step 3
df = pd.read_csv("bank_clean.csv")

# ---------- Part 1: binning numeric attributes into categories ----------
# pd.cut() puts each value into a range (bin); right=False means each range includes its lower limit only
df["age_group"] = pd.cut(df["age"], bins=[0, 30, 40, 50, 200], right=False,
                         labels=["young", "adult", "middle_aged", "senior"])
df["balance_band"] = pd.cut(df["balance"], bins=[-100000, 0, 500, 2000, 1000000], right=False,
                            labels=["negative", "low", "medium", "high"])
df["campaign_band"] = pd.cut(df["campaign"], bins=[1, 2, 4, 100], right=False,
                             labels=["single", "few", "many"])

print("=== 1. Binning rules and record counts ===")
for col in ["age_group", "balance_band", "campaign_band"]:
    # .value_counts(sort=False) counts records in each category, kept in bin order
    print(f"\n{col}:")
    print(df[col].value_counts(sort=False))

binned = df[["age_group", "balance_band", "housing", "campaign_band", "deposit"]]
print("\nFirst 5 records after binning:")
# .head() shows the first 5 rows
print(binned.head())
# .to_csv() saves the categorical dataset
binned.to_csv("bank_binned.csv", index=False)
print("\nSaved bank_binned.csv")

# ---------- Part 2: CSV to ARFF (WEKA format) ----------
# An ARFF file needs a relation name, a list of attributes with their allowed values, and the data rows
arff_dict = {
    "relation": "bank_term_deposit",
    # .cat.categories / .unique() give the allowed values of each attribute
    "attributes": [
        ("age_group", list(df["age_group"].cat.categories)),
        ("balance_band", list(df["balance_band"].cat.categories)),
        ("housing", ["no", "yes"]),
        ("campaign_band", list(df["campaign_band"].cat.categories)),
        ("deposit", ["no", "yes"]),
    ],
    # .astype(str).values.tolist() turns the table into a list of rows
    "data": binned.astype(str).values.tolist(),
}
# open(..., "w") creates the file; arff.dump() writes the dictionary in ARFF format
with open("bank_binned.arff", "w") as f:
    arff.dump(arff_dict, f)

print("\n=== 2. ARFF file excerpt (first 15 lines) ===")
# readlines() reads the file line by line so the first lines can be printed
with open("bank_binned.arff") as f:
    for line in f.readlines()[:15]:
        print(line.rstrip())
print("\nSaved bank_binned.arff")
