import fastf1
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(track_name, aliases=None, years=range(2018, 2024), save_dir="models"):
    fastf1.Cache.enable_cache("cache")
    os.makedirs(save_dir, exist_ok=True)

    all_laps = []
    search_names = aliases or [track_name]

    for year in years:
        found = False
        for name in search_names:
            try:
                schedule = fastf1.get_event_schedule(year)
                if name not in schedule["EventName"].values:
                    print(f"Skipping {name} {year} (event did not exist)")
                    continue

                session = fastf1.get_session(year, name, "R")
                session.load()

                laps = session.laps.pick_quicklaps().copy()
                laps = laps[laps["TrackStatus"] == "1"].copy()
                laps["Year"] = year

                laps = laps[
                    ["Driver", "Team", "Compound", "LapNumber", "TyreLife", "LapTime", "Year"]
                ].dropna()
                laps["LapTime"] = laps["LapTime"].dt.total_seconds()

                all_laps.append(laps)
                print(f"Loaded {name} {year} ({len(laps)} laps)")
                found = True
                break
            except Exception as e:
                print(f"Failed {name} {year}: {e}")
        if not found:
            print(f"Skipped {track_name} {year} (no usable session)")

    if not all_laps:
        print(f"No data collected for {track_name}")
        return

    df = pd.concat(all_laps, ignore_index=True)

    # Compute relative tyre life per compound
    compound_max_life = df.groupby("Compound")["TyreLife"].transform("max")
    df["TyreLife_Relative"] = df["TyreLife"] / compound_max_life

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=["Driver", "Team", "Compound"])

    # Drop raw TyreLife
    X = df_encoded.drop(["LapTime", "TyreLife"], axis=1)
    y = df_encoded["LapTime"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Compute max tyre life per compound for app input validation
    max_tyre_life_per_compound = df.groupby("Compound")["TyreLife"].max().to_dict()

    metadata = {
        "model": model,
        "features": X.columns.tolist(),
        "training_years": sorted(df["Year"].unique().tolist()),
        "valid_drivers": sorted(df["Driver"].unique().tolist()),
        "valid_teams": sorted(df["Team"].unique().tolist()),
        "max_lap_number": int(df["LapNumber"].max()),
        "max_tyre_life_per_compound": max_tyre_life_per_compound
    }

    model_filename = f"{save_dir}/lap_time_model_{track_name.lower().replace(' ', '_')}.joblib"
    joblib.dump(metadata, model_filename)

    print(f"Model saved: {model_filename}")

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model for {track_name}:")
    print(f"MAE = {mae:.2f} s")
    print(f"R²  = {r2:.3f}")
