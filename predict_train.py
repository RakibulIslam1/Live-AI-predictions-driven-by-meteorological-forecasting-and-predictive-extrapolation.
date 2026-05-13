import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Fetching Google Sheet Data...")
sheet_url = "https://docs.google.com/spreadsheets/d/1Ytp6d1vDej0XGYj1ngof50isUxZI7mDjwfkKCnG9TOQ/export?format=csv"
df = pd.read_csv(sheet_url)

# Clean data: Keep only rows with usage > 0
df = df[df['Usage(KWh)'] > 0].copy()
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')

# Find start and end dates for the historical weather API
start_date = df['Date'].min().strftime('%Y-%m-%d')
end_date = df['Date'].max().strftime('%Y-%m-%d')

print(f"Fetching Historical Weather (Dhaka) from {start_date} to {end_date}...")
# Open-Meteo API for historical data (Dhaka coordinates: 23.81, 90.41)
url = f"https://archive-api.open-meteo.com/v1/archive?latitude=23.8103&longitude=90.4125&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean,relative_humidity_2m_mean&timezone=Asia%2FDhaka"

response = requests.get(url).json()
weather_df = pd.DataFrame({
    'Date': pd.to_datetime(response['daily']['time']),
    'Temp': response['daily']['temperature_2m_mean'],
    'Humidity': response['daily']['relative_humidity_2m_mean']
})

# Merge your usage data with the real historical weather data
final_df = pd.merge(df, weather_df, on='Date', how='inner')

# Add calendar features
final_df['Month'] = final_df['Date'].dt.month
final_df['DayOfWeek'] = final_df['Date'].dt.dayofweek

# Define inputs (X) and output (y)
X = final_df[['Temp', 'Humidity', 'Month', 'DayOfWeek']]
y = final_df['Usage(KWh)']

print("Training Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the brain!
joblib.dump(model, 'electricity_model.pkl')
print("Model saved as electricity_model.pkl. You are ready for Phase 2!")