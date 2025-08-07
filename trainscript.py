from src.models import train_model

training_years = range(2018, 2025)

training_targets = {
    "Melbourne": ["Australian Grand Prix"],
    "Shanghai": ["Chinese Grand Prix"],
    "Suzuka": ["Japanese Grand Prix"],
    "Bahrain": ["Bahrain Grand Prix"],
    "Jeddah": ["Saudi Arabian Grand Prix"],
    "Miami": ["Miami Grand Prix"],
    "Imola": ["Emilia Romagna Grand Prix"],
    "Monaco": ["Monaco Grand Prix"],
    "Barcelona": ["Spanish Grand Prix"],
    "Montreal": ["Canadian Grand Prix"],
    "Spielburg": ["Austrian Grand Prix"],
    "Silverstone": ["British Grand Prix", "70th Anniversary Grand Prix"],
    "Spa": ["Belgian Grand Prix"],
    "Budapest": ["Hungarian Grand Prix"],
    "Zandvoort": ["Dutch Grand Prix"],
    "Monza": ["Italian Grand Prix"],
    "Baku": ["Azerbaijan Grand Prix"],
    "Singapore": ["Singapore Grand Prix"],
    "Austin": ["United States Grand Prix"],
    "Mexico": ["Mexico City Grand Prix", "Mexican Grand Prix"], 
    "Brazil": ["Brazilian Grand Prix", "São Paulo Grand Prix"],
    "Las Vegas": ["Las Vegas Grand Prix"], 
    "Qatar": ["Qatar Grand Prix"],
    "Abu Dhabi": ["Abu Dhabi Grand Prix"],
}

for track_name, aliases in training_targets.items():
    print(f"\nTraining model for: {track_name}")
    train_model(track_name, aliases=aliases, years=training_years)