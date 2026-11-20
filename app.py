"""
5024CMD Data Science - Web Interface
Asks the user for the four input variables and shows the prediction from the trained model.
"""
import pandas as pd
import joblib
import streamlit as st

# st.set_page_config() sets the browser tab title and layout of the page
st.set_page_config(page_title="Term Deposit Predictor", layout="centered")


# @st.cache_resource keeps the model in memory so it is only loaded once
@st.cache_resource
def load_model():
    # joblib.load() reads the saved model file created by train_model.py
    return joblib.load("model.pkl")


saved = load_model()
model = saved["model"]

# st.title() and st.write() display text on the page
st.title("Term Deposit Subscription Predictor")
st.write("Enter a customer's details to predict whether they are likely to subscribe "
         "to a term deposit. Model used: **" + saved["name"] + "**.")

# st.number_input() creates a box for entering a number, with a minimum, maximum and default value
age = st.number_input("Age (years)", min_value=18, max_value=100, value=40, step=1)
balance = st.number_input("Average yearly account balance (€)", min_value=-10000,
                          max_value=100000, value=1000, step=100)
# st.selectbox() creates a drop-down list of choices
housing = st.selectbox("Has a housing loan?", ["no", "yes"])
campaign = st.number_input("Number of contacts during this campaign", min_value=1,
                           max_value=63, value=1, step=1)

# st.button() shows a button and returns True when it is clicked
if st.button("Predict"):
    # pd.DataFrame() puts the inputs into one row with the same column names used in training
    customer = pd.DataFrame([{"age": age, "balance": balance,
                              "housing": housing, "campaign": campaign}])
    # .predict() returns 1 (will subscribe) or 0 (will not subscribe)
    prediction = model.predict(customer)[0]
    # .predict_proba() returns the probability of each class; [0][1] is the probability of "yes"
    probability = model.predict_proba(customer)[0][1]

    # st.success() and st.error() show a coloured result box
    if prediction == 1:
        st.success(f"Likely to SUBSCRIBE (probability {probability:.0%}). Recommended: call this customer.")
    else:
        st.error(f"Unlikely to subscribe (probability {probability:.0%}). Recommended: lower call priority.")
