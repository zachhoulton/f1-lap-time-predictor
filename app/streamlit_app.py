import streamlit as st
import pandas as pd
import joblib

model = joblib.load('notebooks/data/lap_time_model_xgb.joblib')
drivers = ['VER', 'GAS', 'PER', 'ALO', 'LEC', 'STR', 'SAR', 'MAG', 'ALB', 'ZHO', 'HUL', 'OCO', 'NOR', 'LAW', 'HAM', 'SAI', 'RUS', 'BOT', 'PIA']
teams = ['Red Bull', 'Mercedes', 'Ferrari', 'Alpine', 'Aston Martin', 'Williams', 'Haas', 'Alfa Romeo', 'McLaren', 'AlphaTauri']
compounds = ['SOFT', 'MEDIUM', 'HARD']
default_inputs = {
    'LapNumber': 10,
    'TyreLife': 3,
    'Stint': 1
}

st.title("F1 Lap Time Predictor")

driver = st.selectbox("Driver", drivers)
team = st.selectbox("Team", teams)
compound = st.selectbox("Tire Compound", compounds)

lap_number = st.number_input("Lap Number", min_value=1, max_value=70, value=default_inputs['LapNumber'])
tyre_life = st.number_input("Tyre Life", min_value=1, max_value=30, value=default_inputs['TyreLife'])
stint = st.number_input("Stint", min_value=1, max_value=5, value=default_inputs['Stint'])

input_df = pd.DataFrame([{
    'LapNumber': lap_number,
    'TyreLife': tyre_life,
    'Stint': stint,
    'Driver_' + driver: 1,
    'Team_' + team: 1,
    'Compound_' + compound: 1,
}])

for col in model.get_booster().feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.get_booster().feature_names]

if st.button("Predict Lap Time"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Lap Time: {prediction:.3f} seconds")