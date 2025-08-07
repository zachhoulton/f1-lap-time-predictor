import streamlit as st
import os
import joblib
import pandas as pd

st.set_page_config(page_title="F1 Lap Time Predictor", layout="centered")
st.title("F1 Lap Time Predictor")

# RacingNews365 circuit outlines
track_images = {
    "Melbourne": "https://cdn.racingnews365.com/Circuits/Australia/_503xAUTO_crop_center-center_none/f1_2024_aus_outline.png?v=1708703549",
    "Shanghai": "https://cdn.racingnews365.com/Circuits/China/_503xAUTO_crop_center-center_none/f1_2024_chn_outline.png?v=1708703688",
    "Suzuka": "https://cdn.racingnews365.com/Circuits/Japan/_503xAUTO_crop_center-center_none/f1_2024_jap_outline.png?v=1708703688",
    "Bahrain": "https://cdn.racingnews365.com/Circuits/Bahrain/_503xAUTO_crop_center-center_none/f1_2024_bhr_outline.png?v=1708703548",
    "Jeddah": "https://cdn.racingnews365.com/Circuits/Saudi-Arabia/_503xAUTO_crop_center-center_none/f1_2024_sau_outline.png?v=1708703549",
    "Miami": "https://cdn.racingnews365.com/Circuits/Miami/_503xAUTO_crop_center-center_none/f1_2024_mia_outline.png?v=1708703688",
    "Imola": "https://cdn.racingnews365.com/Circuits/Imola/_503xAUTO_crop_center-center_none/f1_2024_ero_outline.png?v=1708704457",
    "Monaco": "https://cdn.racingnews365.com/Circuits/Monaco/_503xAUTO_crop_center-center_none/f1_2024_mco_outline.png?v=1708704457",
    "Barcelona": "https://cdn.racingnews365.com/Circuits/Spain/_503xAUTO_crop_center-center_none/f1_2024_spn_outline.png?v=1708704458",
    "Montreal": "https://cdn.racingnews365.com/Circuits/Canada/_503xAUTO_crop_center-center_none/f1_2024_can_outline.png?v=1708704457",
    "Spielburg": "https://cdn.racingnews365.com/Circuits/Austria/_503xAUTO_crop_center-center_none/f1_2024_aut_outline.png?v=1708704458",
    "Silverstone": "https://cdn.racingnews365.com/Circuits/Great-Britain/_503xAUTO_crop_center-center_none/f1_2024_gbr_outline.png?v=1708704458",
    "Spa": "https://cdn.racingnews365.com/Circuits/Belgium/_503xAUTO_crop_center-center_none/f1_2024_bel_outline.png?v=1708704458",
    "Budapest": "https://cdn.racingnews365.com/Circuits/Hungary/_503xAUTO_crop_center-center_none/f1_2024_hun_outline.png?v=1708704458",
    "Zandvoort": "https://cdn.racingnews365.com/Circuits/The-Netherlands/_503xAUTO_crop_center-center_none/f1_2024_nld_outline.png?v=1708704459",
    "Monza": "https://cdn.racingnews365.com/Circuits/Italy/_503xAUTO_crop_center-center_none/f1_2024_ita_outline.png?v=1708704459",
    "Baku": "https://cdn.racingnews365.com/Circuits/Azerbaijan/_503xAUTO_crop_center-center_none/f1_2024_aze_outline.png?v=1708704459",
    "Singapore": "https://cdn.racingnews365.com/Circuits/Singapore/_503xAUTO_crop_center-center_none/f1_2024_sgp_outline.png?v=1708704459",
    "Austin": "https://cdn.racingnews365.com/Circuits/United-States/_503xAUTO_crop_center-center_none/f1_2024_usa_outline.png?v=1708704579",
    "Mexico": "https://cdn.racingnews365.com/Circuits/Mexico/_503xAUTO_crop_center-center_none/f1_2024_mex_outline.png?v=1708704579", 
    "Brazil": "https://cdn.racingnews365.com/Circuits/Brazil/_503xAUTO_crop_center-center_none/f1_2024_bra_outline.png?v=1708705480",
    "Las Vegas": "https://cdn.racingnews365.com/Circuits/Las-Vegas/_503xAUTO_crop_center-center_none/f1_2024_lve_outline.png?v=1708705481", 
    "Qatar": "https://cdn.racingnews365.com/Circuits/Qatar/_503xAUTO_crop_center-center_none/f1_2024_qat_outline.png?v=1708705481",
    "Abu Dhabi": "https://cdn.racingnews365.com/Circuits/Abu-Dhabi/_503xAUTO_crop_center-center_none/f1_2024_abu_outline.png?v=1708705548",
}

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]

if not model_files:
    st.error("No trained models found in 'models/' directory.")
    st.stop()

track_options = {
    f.replace("lap_time_model_", "").replace(".joblib", "").replace("_", " ").title(): f
    for f in model_files
}
selected_track = st.selectbox("Select Track", list(track_options.keys()))
model_file = os.path.join(model_dir, track_options[selected_track])

# Show track outline below the track selection
if selected_track in track_images:
    st.image(track_images[selected_track], caption=f"{selected_track} Circuit Outline")

loaded = joblib.load(model_file)
model = loaded["model"]
feature_names = loaded["features"]

training_years = loaded.get("training_years", [])
valid_drivers = loaded.get("valid_drivers", [])
valid_teams = loaded.get("valid_teams", [])
max_lap_number = loaded.get("max_lap_number", 80)
max_life_per_compound = loaded.get("max_tyre_life_per_compound", {})

# Sidebar inputs
st.sidebar.header("Lap Prediction Inputs")
year = st.sidebar.selectbox("Year", options=training_years, index=len(training_years) - 1)
driver = st.sidebar.selectbox("Driver", options=valid_drivers)
team = st.sidebar.selectbox("Team", options=valid_teams)
compound = st.sidebar.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD"])

compound_max_life = int(max_life_per_compound.get(compound.upper(), 40))

lap_number = st.sidebar.number_input("Lap Number", 1, max_lap_number, 1)
tyre_life = st.sidebar.number_input("Tyre Life", 0, compound_max_life, 0)

if st.sidebar.button("Predict Lap Time"):
    tyre_life_relative = tyre_life / compound_max_life if compound_max_life > 0 else 0

    input_data = {
        "LapNumber": lap_number,
        "TyreLife_Relative": tyre_life_relative,
        "Year": year,
        f"Driver_{driver}": 1,
        f"Team_{team}": 1,
        f"Compound_{compound.upper()}": 1,
    }

    input_df = pd.DataFrame([input_data])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    try:
        predicted_time = model.predict(input_df)[0]
        minutes = int(predicted_time // 60)
        seconds = predicted_time % 60
        st.success(f"Predicted Lap Time: **{minutes}:{seconds:05.2f}** (mm:ss.ss)")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
