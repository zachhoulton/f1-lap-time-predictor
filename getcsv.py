import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache('cache')

years = range(2018, 2025)
results = []

for year in years:
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"Failed to load schedule for {year}: {e}")
        continue

    for _, row in schedule.iterrows():
        event_name = row['EventName']
        try:
            session = fastf1.get_session(year, event_name, 'R')
            session.load()
            num_laps = session.laps.shape[0]
            num_drivers = len(session.laps['Driver'].unique())
            print(f"{year} - {event_name} | Laps: {num_laps} | Drivers: {num_drivers}")
            results.append({
                "Year": year,
                "Event": event_name,
                "Laps": num_laps,
                "Drivers": num_drivers
            })
        except Exception as e:
            print(f"Failed: {year} - {event_name} | {e}")

# Save as csv
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("data/race_sessions.csv", index=False)
print("\nSaved results to data/available_race_sessions.csv")
