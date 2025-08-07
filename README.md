## F1 Lap Time Predictor

This project uses the FastF1 dataset to predict Formula 1 lap times based on things like track, driver, tyre compound, and race year. A streamlit dashboard is used to visualize the circuit layout and explore prediction times.

### Model Features
- Lap time is done using an XGBoost regression model per track
- Inputs are one-hot encoded
- Tyre Life is normalized by compound so that it reflects relative wear
- Track-specific data like max lap number is stored alongside each model for input validation

### How to Run
1. Install dependencies
```pip install -r requirements.txt```

2. Train models
```python trainscript.py```

4. Launch the app
```streamlit run app/streamlit_app.py```

### Future improvements
- Add stint strategy simulations
- Compare predictions vs actual fastest laps per race
- Add more visuals (feature importance)
