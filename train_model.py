"""
5024CMD Data Science - Model Training
Compares several classification algorithms on the 50-record sample for predicting term deposit subscription
from age, balance, housing and campaign, then saves the best model for the web app.
"""
import os
# limit OpenBLAS to one thread to avoid "Memory allocation failed" errors on some PCs
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # draw charts straight to image files (no pop-up window)
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# pd.set_option() shows every column when printing tables
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# pd.read_csv() loads the dataset produced by Step 2 (5 selected attributes)
df = pd.read_csv("bank_selected.csv")

# X holds the four input attributes; y holds the output attribute
X = df[["age", "balance", "housing", "campaign"]]
# .map() converts the text labels "yes"/"no" into 1/0 so the models can learn from them
y = df["deposit"].map({"yes": 1, "no": 0})

# ColumnTransformer() prepares each type of column differently:
# StandardScaler() rescales numeric columns to mean 0 and standard deviation 1 (needed by kNN and Logistic Regression)
# OneHotEncoder() turns the text column "housing" into a 0/1 column
preprocess = ColumnTransformer([
    ("numeric", StandardScaler(), ["age", "balance", "campaign"]),
    ("categorical", OneHotEncoder(drop="if_binary"), ["housing"]),
])

# The algorithms to compare; random_state fixes the randomness so results can be repeated
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
    "k-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
}

# StratifiedKFold() splits the data into 5 parts, keeping the yes/no ratio the same in each part
# 5 folds (not 10) are used because the sample has only 50 records, so each test part still has 10 records
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# make_scorer() wraps precision_score so a fold with no "yes" predictions scores 0 instead of raising a warning
scoring = {"accuracy": "accuracy",
           "precision": make_scorer(precision_score, zero_division=0),
           "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}

results = []
for name, model in models.items():
    # Pipeline() joins the preparation step and the model, so new inputs are prepared the same way
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    # cross_validate() trains and tests the pipeline 5 times and returns each score
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
    # .mean() averages the 5 scores for each metric
    results.append({
        "Algorithm": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1-score": scores["test_f1"].mean(),
        "ROC AUC": scores["test_roc_auc"].mean(),
    })

# pd.DataFrame() turns the list of results into a table; .sort_values() orders it by F1-score
results_table = pd.DataFrame(results).sort_values("F1-score", ascending=False).round(3)
print("=== 5-fold cross-validation results ===")
print(results_table.to_string(index=False))

# The best algorithm is the one with the highest F1-score (balances precision and recall)
best_name = results_table.iloc[0]["Algorithm"]
print("\nBest algorithm:", best_name)

# os.makedirs() creates the "figures" folder if it does not exist yet
os.makedirs("figures", exist_ok=True)

# Bar chart comparing the five algorithms on each metric
# .plot(kind="bar") draws grouped bars from the results table
results_table.set_index("Algorithm")[["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]].plot(
    kind="bar", figsize=(10, 5), edgecolor="black")
plt.title("Model comparison (5-fold cross-validation)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/model_comparison.png", dpi=150)
plt.close()

# Confusion matrix of the best algorithm
best_pipeline = Pipeline([("preprocess", preprocess), ("model", models[best_name])])
# cross_val_predict() gives each record a prediction from the fold where it was in the test part
predicted = cross_val_predict(best_pipeline, X, y, cv=cv)
# confusion_matrix() counts correct and wrong predictions for each class (rows = actual, columns = predicted)
cm = confusion_matrix(y, predicted)
print("\n=== Confusion matrix of", best_name, "(rows = actual, columns = predicted; 0 = no, 1 = yes) ===")
print(pd.DataFrame(cm, index=["actual no", "actual yes"], columns=["predicted no", "predicted yes"]))

fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], labels=["no", "yes"])
ax.set_yticks([0, 1], labels=["no", "yes"])
ax.set_xlabel("Predicted deposit")
ax.set_ylabel("Actual deposit")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
ax.set_title(f"Confusion matrix: {best_name}")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

# Train the best pipeline on all 50 records with .fit(), then save it with joblib.dump()
best_pipeline.fit(X, y)
joblib.dump({"model": best_pipeline, "name": best_name}, "model.pkl")
# .to_csv() saves the comparison table so it can be used in the report
results_table.to_csv("model_results.csv", index=False)
print("Saved model.pkl, model_results.csv and charts in the figures folder")