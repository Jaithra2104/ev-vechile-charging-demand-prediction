import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle  # updated from joblib to fix ModuleNotFoundError
from datetime import datetime
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="EV Adoption Forecast", layout="wide")

# Load model using cloudpickle
with open("forecasting_ev_model.pkl", "rb") as f:
    model = cloudpickle.load(f)

# Custom CSS styling
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #283e51, #485563);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4, h5, h6, .css-10trblm, .css-1v0mbdj, .css-1d391kg {
            color: #FFFFFF !important;
        }
        .css-1cpxqw2, .css-1kyxreq, .css-q8sbsg, .css-ffhzg2 {
            background-color: #1e1e1e !important;
        }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown("""
    <div style='text-align: center; font-size: 36px; font-weight: bold; color: #FFFFFF; margin-top: 20px;'>
        🔮 Electric Vehicle (EV) Adoption Forecast Tool
    </div>
    <div style='text-align: center; font-size: 20px; margin-bottom: 20px; color: #FFFFFF;'>
        Predict EV adoption trends across Washington counties over the next 3 years
    </div>
""", unsafe_allow_html=True)

st.image("ev-car-factory.jpg", use_container_width=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# County selection
county_list = sorted(df['County'].dropna().unique().tolist())
county = st.selectbox("Choose a County", county_list)

# Extract county-specific data
county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]
latest_date = county_df['Date'].max()
months_since_start = county_df['months_since_start'].max()

historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
forecast_horizon = 36
future_rows = []

# Forecasting loop
for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    ev_growth_slope = np.polyfit(range(6), cumulative_ev[-6:], 1)[0] if len(cumulative_ev) == 6 else 0

    row = pd.DataFrame([{
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }])

    pred = model.predict(row)[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
    historical_ev.append(pred)
    historical_ev = historical_ev[-6:]
    cumulative_ev.append(cumulative_ev[-1] + pred)
    cumulative_ev = cumulative_ev[-6:]

# Combine historical and forecast data
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# Plot single-county forecast
st.subheader(f"📊 Cumulative EV Adoption Forecast – {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby('Source'):
    ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o')
ax.set_title(f"Cumulative EV Trend – {county} (Next 3 Years)", fontsize=16, color='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Cumulative EV Count", color='white')
ax.grid(True, alpha=0.3)
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor('#1c1c1c')
ax.tick_params(colors='white')
ax.legend()
st.pyplot(fig)

# Growth summary
historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]

if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 🔼" if forecast_growth_pct > 0 else "decrease 🔽"
    st.success(f"EV adoption in **{county}** is projected to show a **{trend} of {forecast_growth_pct:.2f}%** over the next 3 years.")
else:
    st.warning("Insufficient historical data to compute growth percentage.")

# Multi-county comparison section
st.markdown("---")
st.header("📍 Multi-County EV Trend Comparison")

multi_counties = st.multiselect("Select up to 3 Counties", county_list, max_selections=3)

if multi_counties:
    comparison = []

    for cty in multi_counties:
        cty_df = df[df['County'] == cty].sort_values("Date")
        cty_code = cty_df['county_encoded'].iloc[0]
        hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
        cum_ev = list(np.cumsum(hist_ev))
        months_since = cty_df['months_since_start'].max()
        last_date = cty_df['Date'].max()
        future_rows_cty = []

        for i in range(1, forecast_horizon + 1):
            forecast_date = last_date + pd.DateOffset(months=i)
            months_since += 1
            lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            ev_slope = np.polyfit(range(6), cum_ev[-6:], 1)[0] if len(cum_ev) == 6 else 0

            row = pd.DataFrame([{
                'months_since_start': months_since,
                'county_encoded': cty_code,
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct_change_1,
                'ev_total_pct_change_3': pct_change_3,
                'ev_growth_slope': ev_slope
            }])

            pred = model.predict(row)[0]
            future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
            hist_ev.append(pred)
            hist_ev = hist_ev[-6:]
            cum_ev.append(cum_ev[-1] + pred)
            cum_ev = cum_ev[-6:]

        hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()
        fc_df = pd.DataFrame(future_rows_cty)
        fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]

        combined_cty = pd.concat([hist_cum[['Date', 'Cumulative EV']], fc_df[['Date', 'Cumulative EV']]], ignore_index=True)
        combined_cty['County'] = cty
        comparison.append(combined_cty)

    comp_df = pd.concat(comparison, ignore_index=True)

    st.subheader("📈 Forecast Comparison: Cumulative EV Count")
    fig, ax = plt.subplots(figsize=(14, 7))
    for cty, group in comp_df.groupby('County'):
        ax.plot(group['Date'], group['Cumulative EV'], marker='o', label=cty)
    ax.set_title("EV Adoption Forecast – Multi-County Comparison", fontsize=16, color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Cumulative EV Count", color='white')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1c1c1c")
    fig.patch.set_facecolor('#1c1c1c')
    ax.tick_params(colors='white')
    ax.legend(title="County")
    st.pyplot(fig)

    # Growth summary for multi-county
    growth_report = []
    for cty in multi_counties:
        cty_data = comp_df[comp_df['County'] == cty].reset_index(drop=True)
        base = cty_data['Cumulative EV'].iloc[-forecast_horizon - 1]
        final = cty_data['Cumulative EV'].iloc[-1]
        if base > 0:
            growth = ((final - base) / base) * 100
            growth_report.append(f"{cty}: {growth:.2f}%")
        else:
            growth_report.append(f"{cty}: N/A")

    st.success(f"3-Year Forecasted Growth — {' | '.join(growth_report)}")

# Footer
st.success("EV forecasting complete.")
st.markdown("🔩 Developed as part of the **AICTE Internship Cycle 2 – Skills4Future (S4F)** initiative.")
