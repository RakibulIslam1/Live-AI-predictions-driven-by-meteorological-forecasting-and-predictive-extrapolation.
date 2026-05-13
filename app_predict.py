import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Energy AI", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #00FFCC !important;
        font-family: 'Courier New', Courier, monospace;
        text-shadow: 0px 0px 10px rgba(0, 255, 204, 0.5);
    }
    div[data-testid="metric-container"] {
        background-color: #1A1C23;
        border: 1px solid #00FFCC;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.2);
        transition: transform 0.3s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 30px rgba(0, 255, 204, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('electricity_model.pkl')

model = load_model()

@st.cache_data(ttl=3600)
def get_30_day_weather():
    url = "https://api.open-meteo.com/v1/forecast?latitude=23.8103&longitude=90.4125&hourly=temperature_2m,relative_humidity_2m&timezone=Asia%2FDhaka&forecast_days=16"
    res = requests.get(url).json()
    df_hourly = pd.DataFrame({
        'Time': pd.to_datetime(res['hourly']['time']),
        'Temp': res['hourly']['temperature_2m'],
        'Humidity': res['hourly']['relative_humidity_2m']
    })
    df_hourly['Date'] = df_hourly['Time'].dt.date
    df_daily = df_hourly.groupby('Date').mean().reset_index()
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    
    last_date = df_daily['Date'].iloc[-1]
    last_temp = df_daily['Temp'].iloc[-1]
    last_hum = df_daily['Humidity'].iloc[-1]
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, 15)]
    future_temps = last_temp + np.random.normal(0, 0.8, 14)
    future_hums = last_hum + np.random.normal(0, 2.0, 14)
    
    df_future = pd.DataFrame({'Date': future_dates, 'Temp': future_temps, 'Humidity': future_hums})
    
    df_30_days = pd.concat([df_daily, df_future], ignore_index=True)
    return df_30_days

try:
    df_30 = get_30_day_weather()
    df_30['Month'] = df_30['Date'].dt.month
    df_30['DayOfWeek'] = df_30['Date'].dt.dayofweek
    
    input_features = df_30[['Temp', 'Humidity', 'Month', 'DayOfWeek']]
    df_30['Predicted_KWh'] = model.predict(input_features)
    df_30['Predicted_Bill'] = df_30['Predicted_KWh'] * 10  
    
    total_bill = df_30['Predicted_Bill'].sum()
    total_kwh = df_30['Predicted_KWh'].sum()
    avg_temp = df_30['Temp'].mean()

    st.title("⚡ 30-Day Electricity Forecast")
    st.markdown("Live AI predictions driven by meteorological forecasting and predictive extrapolation.")
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 30-Day Projected Bill", f"৳ {total_bill:,.2f}", "Total Cost")
    col2.metric("🔋 30-Day Total Usage", f"{total_kwh:,.1f} KWh", "Total Energy")
    col3.metric("🌡️ 30-Day Avg Temp", f"{avg_temp:.1f} °C", "Climate Outlook")
    
    st.write("---")
    
    st.subheader("📈 Daily Bill & Usage Projection")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_30['Date'], y=df_30['Predicted_Bill'],
        fill='tozeroy',
        mode='lines+markers',
        name='Daily Bill (৳)',
        line=dict(color='#00FFCC', width=3),
        marker=dict(size=6, color='#00FFCC'),
        fillcolor='rgba(0, 255, 204, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_30['Date'], y=df_30['Temp'],
        mode='lines',
        name='Avg Temp (°C)',
        line=dict(color='#FF00FF', width=2, dash='dot'),
        yaxis='y2'
    ))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
        yaxis=dict(title="Estimated Bill (৳)", gridcolor='rgba(255,255,255,0.1)'),
        yaxis2=dict(title="Temperature (°C)", overlaying='y', side='right', showgrid=False),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 View Raw 30-Day AI Data"):
        display_df = df_30[['Date', 'Temp', 'Humidity', 'Predicted_KWh', 'Predicted_Bill']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df = display_df.round(2)
        st.dataframe(display_df, use_container_width=True)

except Exception as e:
    st.error(f"System Error: {e}")
