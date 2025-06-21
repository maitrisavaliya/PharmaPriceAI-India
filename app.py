import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and features
model = joblib.load('rf_model.pkl')
features = joblib.load('model_features.pkl')

st.title("💊 Pharma Price Predictor (India)")

# User inputs
dosage_form = st.selectbox("Dosage Form", ["tablet", "capsule", "syrup", "injection"])
pack_unit = st.selectbox("Pack Unit", ["strip", "bottle", "vial"])
pack_size = st.number_input("Pack Size", min_value=1.0, value=10.0)
primary_strength = st.number_input("Primary Strength (mg)", min_value=1.0, value=500.0)
num_active_ingredients = st.number_input("Number of Active Ingredients", min_value=1, max_value=5, value=1)
therapeutic_class = st.selectbox("Therapeutic Class", ["antibiotic", "antihistamine", "analgesic", "other"])

# Construct input row
input_data = pd.DataFrame(columns=features)
input_data.loc[0] = 0  # fill with 0s first

# Fill numerical features
input_data.at[0, 'pack_size'] = float(pack_size)
input_data.at[0, 'primary_strength_mg'] = float(primary_strength)
input_data.at[0, 'num_active_ingredients'] = int(num_active_ingredients)


# One-hot encode inputs
col_name_map = {
    f'dosage_form_{dosage_form}': dosage_form,
    f'pack_unit_{pack_unit}': pack_unit,
    f'therapeutic_class_{therapeutic_class}': therapeutic_class
}

for col in input_data.columns:
    if col in col_name_map:
        input_data.at[0, col] = 1

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: ₹{prediction:.2f}")
