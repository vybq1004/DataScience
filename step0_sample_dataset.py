"""
5024CMD Data Science - Step 0: Select a 50-record sample
Takes a random sample of 50 records from bank.csv, keeping the same yes/no ratio
as the full dataset, and saves it as bank_sample.csv for all later steps.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# pd.set_option() shows every column when printing tables
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# pd.read_csv() loads the full dataset into a DataFrame
df = pd.read_csv("bank.csv")
print("Full dataset:", df.shape[0], "records")
# .value_counts() counts how many records have each value of deposit
print(df["deposit"].value_counts())

# train_test_split() with train_size=50 picks 50 records at random
# stratify=df["deposit"] keeps the same yes/no proportion as the full dataset
# random_state=42 fixes the random choice so the same 50 records are selected every time
sample, _ = train_test_split(df, train_size=50, stratify=df["deposit"], random_state=42)

# .reset_index(drop=True) renumbers the rows from 0 to 49
sample = sample.reset_index(drop=True)

print("\nSample:", sample.shape[0], "records")
print(sample["deposit"].value_counts())

# .to_csv() saves the sample; index=False stops pandas adding a row-number column
sample.to_csv("bank_sample.csv", index=False)
print("\nSaved bank_sample.csv")
