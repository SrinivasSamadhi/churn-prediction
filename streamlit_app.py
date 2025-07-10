
import streamlit as st
import numpy as np
#import joblib

# Load model and scaler
from tensorflow.keras.models import load_model
import joblib

model = load_model("model.h5")         # ✅ Now fully trained model
scaler = joblib.load("scaler.pkl")        # ✅ Your StandardScaler



st.title("Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of exit.")

# Input fields
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
credit_score = st.slider("Credit Score", 300, 1000, 600)
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (years)", 0, 10, 3)
balance = st.number_input("Balance", value=10000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)

# Encode inputs
geo_map = {"France": [1, 0, 0], "Germany": [0, 0, 1], "Spain": [0, 1, 0]}
gender_encoded = 1 if gender == "Male" else 0
geo_encoded = geo_map[geography]

# Construct input array
input_data = np.array(geo_encoded + [gender_encoded, credit_score, age, tenure, balance,
                                     num_of_products, has_credit_card,
                                     is_active_member, estimated_salary]).reshape(1, -1)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    print(prediction[0][0])
    prediction = (prediction[0][0] > 0.5).astype(int)
    print(prediction)
    if prediction == 1:
        st.error("⚠️ This customer is likely to exit.")
    else:
        st.success("✅ This customer is likely to stay.")
