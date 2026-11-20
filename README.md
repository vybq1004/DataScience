# DataScience: Predicting Term Deposit Subscription

5024CMD Data Science Individual Assignment, Valerina Yeap Bei Qian.

This project predicts whether a bank customer will subscribe to a term deposit, using four input attributes: **age**, **balance**, **housing** and **campaign**. It includes a simple web interface where a user enters these values and receives a prediction from the trained model.

## Dataset

- `bank.csv`: full Bank Marketing dataset (11,162 records, 17 attributes), provided by the 5024CMD lecturer.
- `bank_sample.csv`: the 50-record sample used for this study, selected at random with the same yes/no ratio as the full dataset (created by `step0_sample_dataset.py`, random seed 42).

 The original data is from the UCI Machine Learning Repository:

Moro, S., Rita, P., & Cortez, P. (2012). *Bank marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

## Files

| File | Purpose |
|---|---|
| `step0_sample_dataset.py` | Selects a stratified random sample of 50 records and saves `bank_sample.csv` |
| `step1_data_quality_check.py` | Checks dataset size, data types, missing values, duplicates and hidden missing values |
| `step2_handle_missing_values.py` | Removes excluded attributes and saves `bank_selected.csv` |
| `step3_clean_data.py` | Checks duplicates, converts data types, detects outliers (IQR) and saves `bank_clean.csv` |
| `step4_transform_and_arff.py` | Bins age, balance and campaign into categories and exports `bank_binned.arff` |
| `step5_redundancy_check.py` | Checks correlation between the four inputs (dimensionality reduction) |
| `step6_database.py` | Stores the data in a SQLite database (`bank.db`) |
| `step7_eda.py` | Summary statistics, charts and statistical tests |
| `step8_association_rules.py` | Apriori association rules on 10 records |
| `train_model.py` | Compares five algorithms with 5-fold cross-validation and saves the best model as `model.pkl` |
| `app.py` | Web interface that takes user input and shows the model's prediction |

## How to run

1. Install Python 3.10 or later.
2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
3. Run the scripts in order:
   ```
   python step0_sample_dataset.py
   python step1_data_quality_check.py
   python step2_handle_missing_values.py
   python step3_clean_data.py
   python step4_transform_and_arff.py
   python step5_redundancy_check.py
   python step6_database.py
   python step7_eda.py
   python step8_association_rules.py
   python train_model.py
   ```
   Charts are saved in the `figures` folder.
4. Start the web interface:
   ```
   streamlit run app.py
   ```
   The app opens in your browser at http://localhost:8501.
