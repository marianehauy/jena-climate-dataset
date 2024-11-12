from datetime import date, time

import numpy as np
import pandas as pd

FLOAT_COLS = [
    "internal_pressure",
    "temperature_celsius",
    "dew_point",
    "relative_humidity",
    "saturation_vapor_pressure",
    "vapor_pressure",
    "vapor_pressure_deficit",
    "specific_humidity",
    "water_vapor_concentration",
    "airtight",
    "wind_speed",
    "maximum_wind_speed",
    "wind_direction_degree",
]


def prepare_data(df, period="1h"):
    """
    Prepare and process the input dataframe for climate data analysis.

    This function renames columns, drops unnecessary columns, converts data types,
    handles missing values, and resamples the data to a specified time period.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing climate data.
    period (str, optional): The time period for resampling the data. Defaults to "1h" (1 hour).

    Returns:
    pandas.DataFrame: A processed dataframe with resampled climate data, where numeric columns
                      are averaged over the specified time period.
    """
    df.columns = [
        "date",
        "internal_pressure",
        "temperature_celsius",
        "temperature_kelvin",
        "dew_point",
        "relative_humidity",
        "saturation_vapor_pressure",
        "vapor_pressure",
        "vapor_pressure_deficit",
        "specific_humidity",
        "water_vapor_concentration",
        "airtight",
        "wind_speed",
        "maximum_wind_speed",
        "wind_direction_degree",
    ]
    df = df.drop("temperature_kelvin", axis=1)
    # convert float
    df[FLOAT_COLS] = df[FLOAT_COLS].astype(float)

    # convert date to datetime
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M:%S")

    # converting -9999 to None
    df.loc[df["wind_speed"] == -9999.000000, "wind_speed"] = None
    df.loc[df["wind_speed"] == -9999.000000, "maximum_wind_speed"] = None

    df_resample = df.set_index("date").resample(period)[FLOAT_COLS].mean().reset_index()
    # calculate day since beginning of the dataset
    df_resample["days_since_beginning"] = (
        df_resample["date"].dt.date - df_resample["date"].dt.date.min()
    ).apply(lambda x: x.days)
    df_resample["year"] = df_resample["date"].dt.year
    return df_resample


def _get_seasonality(date_series, period=365, shift=0, amplitude=1):
    # degree significa o shift inicial
    date_series = np.arange(0, len(date_series))
    df_seas = pd.DataFrame()
    df_seas["date_cos"] = np.cos(2 * np.pi * date_series / period + shift) * amplitude
    df_seas["date_sin"] = np.sin(2 * np.pi * date_series / period + shift) * amplitude
    return df_seas


def add_seasonal_features(df, date_col="date"):
    """
    Add temporal features to the input dataframe based on the date column.

    This function calculates and adds yearly, monthly, and daily seasonality components
    to the input dataframe. It uses cosine and sine transformations to represent
    cyclical time features.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing a 'date' column.

    Returns:
    pandas.DataFrame: A new dataframe with additional columns for temporal features:
                        'year_cos', 'year_sin', 'month_cos', 'month_sin', 'day_cos', 'day_sin'.
                        The original columns from the input dataframe are preserved.
    """
    daily_period = 24
    monthly_period = daily_period * 30
    yearly_period = daily_period * 365

    sazonality_year = _get_seasonality(df[date_col], period=yearly_period)
    sazonality_month = _get_seasonality(df[date_col], period=monthly_period)
    sazonality_day = _get_seasonality(df[date_col], period=daily_period)

    sazonality_year.columns = ["year_cos", "year_sin"]
    sazonality_month.columns = ["month_cos", "month_sin"]
    sazonality_day.columns = ["day_cos", "day_sin"]

    sazonality_components = pd.concat(
        [sazonality_year, sazonality_month, sazonality_day], axis=1
    )

    df_feat = pd.concat([df, sazonality_components], axis=1)

    return df_feat


def add_lag_features(df, lag_features, lags=[24, 48, 72]):
    """
    Add lagged features to the input dataframe for time series analysis.

    This function creates new columns in the dataframe that represent lagged values
    of specified features and the target variable. Lagged features can be useful
    for capturing time-dependent patterns in the data.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing time series data.
    features (list): A list of column names for which to create lagged features.
    target (str): The name of the target variable column.
    lags (list, optional): A list of lag values in hours. Defaults to [24, 48, 72].

    Returns:
    pandas.DataFrame: The input dataframe with additional columns for lagged features.
                      New columns are named as '{original_column_name}_lag_{lag_value}h'.
    """
    # Criando lag features
    for lag in lags:
        for col in lag_features:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    return df


def add_moving_average_features(df, features, window=24):
    """
    Add moving average features to the input dataframe.

    This function calculates the 24-hour moving average for specified features
    and adds them as new columns to the input dataframe.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing the time series data.
    features (list): A list of column names for which to calculate moving averages.

    Returns:
    pandas.DataFrame: A new dataframe with additional columns for 24-hour moving
                      averages of the specified features. The new column names
                      are formatted as '{original_column_name}_mean_24h'.
    """
    # calcular média móvel das últimas 24h
    df_mean_features = df[features].rolling(window=window, min_periods=1).mean()
    df_mean_features.columns = [f"{col}_mean_{str(window)}h" for col in features]
    df = pd.concat([df, df_mean_features], axis=1)

    return df


def remove_null(df, FEATURES, TARGET):
    df = df.dropna(subset=FEATURES + [TARGET], axis=0, how="any")
    return df
