"""
5024CMD Data Science - Step 6: Store the Data in a SQLite Database
"""
import sqlite3
import pandas as pd

# pd.set_option() shows every column when printing tables
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# pd.read_csv() loads the raw sample and the binned/cleaned data
raw = pd.read_csv("bank_sample.csv")
clean = pd.read_csv("bank_clean.csv")
binned = pd.read_csv("bank_binned.csv")

# Give every customer an ID (1 to 50) so the tables can be linked
raw.insert(0, "customer_id", range(1, len(raw) + 1))
clean.insert(0, "customer_id", range(1, len(clean) + 1))

# sqlite3.connect() opens (or creates) the database file bank.db
conn = sqlite3.connect("bank.db")
# conn.cursor() creates an object used to run SQL commands
cur = conn.cursor()

# executescript() runs several SQL statements: drop old tables, then create the new ones
cur.executescript("""
DROP TABLE IF EXISTS raw_sample;
DROP TABLE IF EXISTS campaign_contact;
DROP TABLE IF EXISTS customer;

CREATE TABLE customer (
    customer_id   INTEGER PRIMARY KEY,
    age           INTEGER NOT NULL,
    balance       INTEGER NOT NULL,
    housing       TEXT    NOT NULL CHECK (housing IN ('yes', 'no')),
    age_group     TEXT,
    balance_band  TEXT
);

CREATE TABLE campaign_contact (
    contact_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id    INTEGER NOT NULL,
    campaign       INTEGER NOT NULL,
    campaign_band  TEXT,
    deposit        TEXT    NOT NULL CHECK (deposit IN ('yes', 'no')),
    FOREIGN KEY (customer_id) REFERENCES customer (customer_id)
);
""")

# Build the customer and campaign_contact tables from the cleaned and binned data
customer = pd.DataFrame({
    "customer_id": clean["customer_id"], "age": clean["age"], "balance": clean["balance"],
    "housing": clean["housing"], "age_group": binned["age_group"], "balance_band": binned["balance_band"]})
contact = pd.DataFrame({
    "customer_id": clean["customer_id"], "campaign": clean["campaign"],
    "campaign_band": binned["campaign_band"], "deposit": clean["deposit"]})

# .to_sql() inserts a DataFrame into a database table; if_exists="append" keeps the table design above
customer.to_sql("customer", conn, if_exists="append", index=False)
contact.to_sql("campaign_contact", conn, if_exists="append", index=False)
# The untouched 50-record sample is stored as well, so every cleaning step can be repeated
raw.to_sql("raw_sample", conn, if_exists="replace", index=False)
conn.commit()  # conn.commit() saves the changes to the file

print("=== 1. Tables and record counts ===")
for table in ["raw_sample", "customer", "campaign_contact"]:
    # pd.read_sql() runs a SQL query and returns the result as a DataFrame
    n = pd.read_sql(f"SELECT COUNT(*) AS records FROM {table}", conn)["records"][0]
    print(f"{table}: {n} records")

print("\n=== 2. Example query: subscription rate by housing loan ===")
query = """
SELECT c.housing,
       COUNT(*) AS customers,
       ROUND(AVG(CASE WHEN k.deposit = 'yes' THEN 1.0 ELSE 0 END) * 100, 1) AS subscription_rate_percent
FROM customer c
JOIN campaign_contact k ON c.customer_id = k.customer_id
GROUP BY c.housing;
"""
print(pd.read_sql(query, conn))

# conn.close() closes the connection to the database
conn.close()
print("\nSaved bank.db")
