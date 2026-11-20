"""
5024CMD Data Science - Step 8: Association Rule Mining (Apriori) on 10 records
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# pd.set_option() shows every column when printing tables
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

# pd.read_csv() loads the binned (categorical) data produced by Step 4
binned = pd.read_csv("bank_binned.csv")

# .head(10) selects the first 10 records (the sample was already randomly ordered in Step 0)
ten = binned.head(10)
print("=== 1. The 10 records used ===")
print(ten)

# pd.get_dummies() turns each category into its own True/False column, e.g. "housing=no"
items = pd.get_dummies(ten, prefix_sep="=")

# apriori() finds item combinations that appear in at least min_support of the records (30% = 3 of 10)
frequent = apriori(items, min_support=0.3, use_colnames=True)
print("\nNumber of frequent itemsets:", len(frequent))

# association_rules() builds "if X then Y" rules and keeps those with confidence of at least 0.8
rules = association_rules(frequent, metric="confidence", min_threshold=0.8)

# keep only rules whose result (consequent) is the subscription decision
target = rules[rules["consequents"].apply(lambda c: len(c) == 1 and next(iter(c)).startswith("deposit="))]
# .sort_values() orders the rules by lift and then confidence, highest first
target = target.sort_values(["lift", "confidence", "support"], ascending=False)

# frozenset values are converted to readable text such as "housing=no AND campaign_band=single"
target = target.assign(
    rule=target["antecedents"].apply(lambda a: " AND ".join(sorted(a))) + "  =>  "
         + target["consequents"].apply(lambda c: next(iter(c))))
print("\n=== 2. Rules that predict deposit (min support 0.3, min confidence 0.8) ===")
print(target[["rule", "support", "confidence", "lift"]].round(2).to_string(index=False))
