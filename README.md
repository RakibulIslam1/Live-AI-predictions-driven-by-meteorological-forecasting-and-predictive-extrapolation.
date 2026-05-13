# Live-AI-predictions-driven-by-meteorological-forecasting-and-predictive-extrapolation.

Ever wondered what your electricity bill is going to look like at the end of the month before it actually arrives? That is exactly what this project solves. 

This repository contains a full-stack machine learning dashboard built to predict daily electricity usage (KWh) and estimated billing costs up to 30 days in advance. It uses a trained AI model that looks at historical energy consumption alongside live and forecasted weather data to generate highly accurate predictions.

All wrapped in a sleek, neon-styled web interface.

## ✨ Features

* **🔮 30-Day Predictive Forecasting:** Uses a Random Forest machine learning model to project electricity usage for the next month.
* **☁️ Live Weather Integration:** Automatically fetches 30-day hourly weather forecasts (temperature and humidity) via the Open-Meteo API to build a complete 30-day climate outlook.
* **📊 Interactive Data Visualization:** Features a custom, cyberpunk-styled interactive area chart built with Plotly. Hover over any day to see the exact forecasted temperature, KWh usage, and estimated cost in Taka (৳).
* **📱 Responsive Web UI:** Built with Streamlit to be perfectly viewable on desktop monitors or mobile phones.
* **📂 Raw Data Access:** Includes a collapsible data table for clients who want to export or verify the exact daily math.

## 🛠️ Tech Stack

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** `scikit-learn` (Random Forest Regressor)
* **Data Processing:** `pandas`, `numpy`
* **Visualization:** `plotly`
* **External APIs:** [Open-Meteo](https://open-meteo.com/) (Historical & Live Weather Data)

## 🧠 How it Works (Under the Hood)

The project is split into two main phases:

1. **The Brain (Training):** A separate script downloads historical electricity usage from a Google Sheet, matches those dates with historical weather data, and trains a Random Forest model to understand how temperature and humidity drive up electricity usage. The learned patterns are saved into an `electricity_model.pkl` file.
2. **The Face (Dashboard):** The `app.py` script acts as the live dashboard. It loads the pre-trained `.pkl` model, fetches today's weather forecast, runs the math in milliseconds, and displays the beautiful UI to the user. No heavy training happens on the live site, making it incredibly fast.

## 🚀 Running the Project Locally

If you want to run this dashboard on your own machine, follow these steps:

### 1. Clone the repository
git clone [https://github.com/RakibulIslam1/Live-AI-predictions-driven-by-meteorological-forecasting-and-predictive-extrapolation..git](https://github.com/RakibulIslam1/Live-AI-predictions-driven-by-meteorological-forecasting-and-predictive-extrapolation..git)
cd Live-AI-predictions-driven-by-meteorological-forecasting-and-predictive-extrapolation
2. Install the dependencies
Make sure you have Python installed, then run:
pip install -r requirements.txt
3. Run the application
Fire up the Streamlit server:

streamlit run app.py
The dashboard will automatically open in your default web browser at http://localhost:8501.

🌍 Live Deployment
This app is designed to be easily deployed to Streamlit Community Cloud. Because it relies entirely on free, open-source libraries and APIs (no API keys required for Open-Meteo), it can be hosted completely for free.

Built with ❤️ for real-world utility.
