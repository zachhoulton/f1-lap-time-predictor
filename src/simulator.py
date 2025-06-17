import pandas as pd

COMPOUND_MAP = {
    'SOFT': 0,
    'MEDIUM': 1,
    'HARD': 2,
}

def simulate_strategy(compounds, stint_lengths, model, driver_id, team_id, pit_loss = 22.0):
    assert len(compounds) == len(stint_lengths), "Mismatch in compounds and stint lengths"

    total_time = 0.0
    lap_number = 1
    for stint_index, (compound, laps) in enumerate(zip(compounds, stint_lengths)):
        for tyre_life in range(1, laps+1):
            features = pd.DataFrame([[
                COMPOUND_MAP[compound],
                tyre_life,
                stint_index + 1,
                team_id,
                driver_id
            ]], columns=['Compound', 'TyreLife', 'Stint', 'Team', 'Driver'])

            lap_time = model.predict(features)[0]
            total_time += lap_time
            lap_number += 1

        if stint_index < len(compounds) - 1:
            total_time += pit_loss

    return total_time