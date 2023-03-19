import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime
from statsmodels.tsa.seasonal import STL
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.graphics.tsaplots import plot_acf
import warnings

warnings.simplefilter(action='ignore', category=ValueWarning)

DATA_PATH = "data/rm-yields-data.csv"

def load_data():
    return pd.read_csv(DATA_PATH)

def impute_missing_values(df):
    columns_with_missing_values = ['Winter Wheat', 'Canola', 'Spring Wheat', 'Mustard', 'Durum', 'Sunflowers', 'Oats', 'Lentils', 'Peas', 'Barley', 'Fall Rye', 'Canary Seed', 'Spring Rye', 'Tame Hay', 'Flax', 'Chickpeas']
    for rm in df['RM'].unique():
        for column_name in columns_with_missing_values:
            rm_df = df[(df['RM'] == rm) & df[column_name].notnull()]
            mean_value = rm_df[column_name].mean()
            df.loc[(df['RM'] == rm) & df[column_name].isnull(), column_name] = mean_value
    df.fillna(0, inplace=True)
    return df.round(1)

def calculate_rolling_mean(df, unique_rms):
    time_series_results = {}
    for rm in unique_rms:
        rm_data = df[df['RM'] == rm].sort_values(by='Year')
        rm_data.set_index('Year', inplace=True)
        rm_data.drop(columns=['RM'], inplace=True)
        rolling_mean = rm_data.rolling(window=3).mean()
        time_series_results[rm] = rolling_mean
    return time_series_results

def stl_decomposition_fillna(series, period, seasonal=7):
    series_temp = series.dropna()
    stl = STL(series_temp, period=period, seasonal=seasonal)
    result = stl.fit()
    fitted_values = result.trend + result.seasonal
    filled_series = series.copy()
    filled_series[series.isna()] = fitted_values[series.isna()]
    return filled_series

def prepare_time_series_filled_results(df, unique_rms):
    time_series_filled_results = {}
    for rm in unique_rms:
        rm_data = df[df['RM'] == rm].sort_values(by='Year')
        rm_data.set_index('Year', inplace=True)
        rm_data.drop(columns=['RM'], inplace=True)
        filled_rm_data = pd.DataFrame()
        for crop in rm_data.columns:
            filled_series = stl_decomposition_fillna(rm_data[crop], period=3)
            filled_rm_data[crop] = filled_series
        time_series_filled_results[rm] = filled_rm_data
    return time_series_filled_results

def forecast_prophet(rm, crop, time_series_filled_results, start_year=2022, end_year=2027):
    series = time_series_filled_results[rm][crop].reset_index()
    series.columns = ['ds', 'y']
    series['ds'] = series['ds'].apply(lambda x: datetime(x, 1, 1))
    model = Prophet()
    model.fit(series)
    future = model.make_future_dataframe(periods=(end_year - start_year + 1), freq='Y')
    forecast = model.predict(future)
    return forecast.iloc[-1]['yhat']

def display_forecast(rm, crop, time_series_filled_results):
    forecast_value = forecast_prophet(rm, crop, time_series_filled_results)
    st.write(f"Forecasted yield for {crop} in RM {rm}: {forecast_value:.2f}")

def plot_forecast(rm, crop, time_series_filled_results):
    series = time_series_filled_results[rm][crop].reset_index()
    series.columns = ['ds', 'y']
    series['ds'] = series['ds'].apply(lambda x: datetime(x, 1, 1))
    model = Prophet()
    model.fit(series)
    future = model.make_future_dataframe(periods=(2027 - 2022 + 1), freq='Y')
    forecast = model.predict(future)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(series['ds'], series['y'], label='Observed')
    ax.plot(future['ds'], forecast['yhat'], label='Forecasted', linestyle='--')
    ax.set_xlabel('Year')
    ax.set_ylabel('Yield')
    ax.set_title(f'Prophet Forecast for {crop} in RM {rm}')
    plt.legend()
    return fig

    

def main():
    
    st.title("Saskatchewan Crop Yield Forecast")

    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", "70 °F", "1.2 °F")
    col2.metric("Wind", "9 mph", "-8%")
    col3.metric("Humidity", "86%", "4%")

    df = load_data()
    df = impute_missing_values(df)
    unique_rms = df['RM'].unique()
    time_series_results = calculate_rolling_mean(df, unique_rms)
    filtered_df = df[(df['Year'] >= 1938) & (df['Year'] <= 2021)]
    time_series_filled_results = prepare_time_series_filled_results(filtered_df, unique_rms)
    
    crops = ['Winter Wheat', 'Canola', 'Spring Wheat', 'Oats', 'Barley', 'Fall Rye', 'Flax']
    rm = st.selectbox("Select RM:", unique_rms)
    crop = st.selectbox("Select crop:", crops)

    display_forecast(rm, crop, time_series_filled_results)
    forecast_plot = plot_forecast(rm, crop, time_series_filled_results)
    st.pyplot(forecast_plot)

if __name__ == "__main__":
    main()


