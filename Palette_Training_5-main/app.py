import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from prophet import Prophet
from datetime import datetime
from statsmodels.tsa.seasonal import STL
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.graphics.tsaplots import plot_acf
import warnings

warnings.simplefilter(action='ignore', category=ValueWarning)


DATA_PATH = "https://raw.githubusercontent.com/Metaluv/Palette_Training/main/Palette_Training_5-main/data/rm-yields-data.csv"


def load_data():
    return pd.read_csv(DATA_PATH)

def load_merged_data():
    # Read in geo data
    shapefile_path = os.path.join('Rural_Municipality', 'Rural_Municipality.shp')
    geo_df = gpd.read_file(shapefile_path)
    geo_df.rename(columns={'RMNO': 'RM',}, inplace=True)
    geo_df['RM'] = geo_df['RM'].astype('int64')

    data_df = pd.read_csv(DATA_PATH)

    merged = geo_df.merge(data_df, on='RM', how='left')
    return merged


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

def plot_choropleth_map(merged, crop, year):
    # Filter the merged DataFrame based on the specified year
    merged = merged[merged['Year'] == year]
    
    vmin, vmax = merged[crop].min(), merged[crop].max()
    fig, ax = plt.subplots(1, figsize=(10, 6))
    merged.plot(column=crop, cmap='Oranges', linewidth=0.8, ax=ax, edgecolor='0.8')
    ax.axis('off')
    ax.set_title(f'{crop} Yield in Saskatchewan Rural Municipalities ({year})\n', fontdict={'fontsize': '15', 'fontweight' : '3'})
    ax.annotate('Source: Saskatchewan Ministry of Agriculture', xy=(0.1, .08), xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')
    sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm)
    return fig

def load_crop_data(crop):
    base_url = 'https://raw.githubusercontent.com/Metaluv/Palette_Training/main/Palette_Training_5-main/data/'
    crop_filename = {
        'Winter Wheat': 'wheat.csv',
        'Canola': 'canola.csv',
        'Spring Wheat': 'wheat.csv',
        'Oats': 'oats.csv',
        'Barley': 'barley.csv',
        'Fall Rye': 'mustard.csv',
        'Flax': 'flax.csv'
    }
    file_url = base_url + crop_filename[crop]
    data = pd.read_csv(file_url)
    return data

def plot_forecast(crop, window=3):
    data = load_crop_data(crop)
    series = data[['Date', f'{crop} 1CAN ($ per tonne)']]
    series.columns = ['ds', 'y']
    series['ds'] = pd.to_datetime(series['ds'])

    rolling_mean = series['y'].rolling(window=window).mean()
    rolling_std = series['y'].rolling(window=window).std()

    model = Prophet()
    model.fit(series)
    future = model.make_future_dataframe(periods=(2027 - 2022 + 1), freq='Y')
    forecast = model.predict(future)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(series['ds'], series['y'], label='Observed')
    ax.plot(series['ds'], rolling_mean, label='Rolling Mean', color='red')
    ax.plot(series['ds'], rolling_std, label='Rolling Std', color='green')
    ax.plot(future['ds'], forecast['yhat'], label='Forecasted', linestyle='--')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Price per tonne')
    ax.set_title(f'Prophet Forecast for {crop} with Rolling Mean and Std')
    plt.legend()
    return fig
    

def main():
    # add title and image
    st.set_page_config(page_title="Saskatchewan Crop Yield Forecast", page_icon=":seedling:")

    #st.image("PaletteSkills_Banner.png")
    st.title("Saskatchewan Crop Yield Forecast")
    st.write("This app predicts the yield of a crop in a given RM for the next 5 years. The objective of this assignment is to provide hands-on experience in data science, including data cleaning, exploratory data analysis, and time series forecasting. The data used in this app is from the [Saskatchewan Crop Production Service](https://dashboard.saskatchewan.ca/agriculture/rm-yields/rm-yields-data). Crop yields by Rural Municipality (R.M.) are produced annually with data from the Ministry of Saskatchewan Crop Report and Saskatchewan Crop Insurance Corporation. Yields are available for each RM as long as there are no confidentiality concerns. The data is available from 1938 to 2021.")

    
    df = load_data()
    df = impute_missing_values(df)
    unique_rms = df['RM'].unique()
    time_series_results = calculate_rolling_mean(df, unique_rms)
    filtered_df = df[(df['Year'] >= 1938) & (df['Year'] <= 2021)]
    time_series_filled_results = prepare_time_series_filled_results(filtered_df, unique_rms)
    
    crops = ['Winter Wheat', 'Canola', 'Spring Wheat', 'Mustard', 'Durum', 'Sunflowers', 'Oats', 'Lentils', 'Peas', 'Barley', 'Fall Rye', 'Canary Seed', 'Spring Rye', 'Tame Hay', 'Flax', 'Chickpeas']

    crop = st.radio("Select crop:", crops)
    rm = st.selectbox("Select RM:", unique_rms)
    year = st.selectbox("Select year:", df['Year'].unique())

    display_forecast(rm, crop, time_series_filled_results)
    forecast_plot = plot_forecast(rm, crop, time_series_filled_results)
    st.pyplot(forecast_plot)

    # Plot the choropleth map
    merged = load_merged_data()  # Load the merged DataFrame (you need to create the load_merged_data function to load the merged data)
    choropleth_map = plot_choropleth_map(merged, crop, year)
    st.pyplot(choropleth_map)

    # Call the plot_forecast function with the crop argument
    rolling_mean_std_plot = plot_forecast(crop)
    st.pyplot(rolling_mean_std_plot)

    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", "70 °F", "1.2 °F")
    col2.metric("Wind", "9 mph", "-8%")
    col3.metric("Humidity", "86%", "4%")

if __name__ == "__main__":
    main()


