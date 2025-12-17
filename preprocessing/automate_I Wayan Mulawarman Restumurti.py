import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def eda_summary(df):
    summary = {
        "shape": df.shape,
        "missing_values": df.isnull().sum(),
        "skewness": df.select_dtypes(include=['int64', 'float64']).skew()
    }
    return summary


def clean_data(df):
    df_clean = df.drop(columns=['CUST_ID'])

    # Handle missing values
    df_clean['MINIMUM_PAYMENTS'].fillna(
        df_clean['MINIMUM_PAYMENTS'].median(), inplace=True
    )
    df_clean['CREDIT_LIMIT'].fillna(
        df_clean['CREDIT_LIMIT'].median(), inplace=True
    )

    # Drop duplicate rows
    df_clean = df_clean.drop_duplicates()

    return df_clean


def log_transform(df, skew_threshold=1):
    df_log = df.copy()

    skewed_cols = df_log.skew()[df_log.skew() > skew_threshold].index
    df_log[skewed_cols] = np.log1p(df_log[skewed_cols])

    return df_log


def scale_data(df):
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(scaled_array, columns=df.columns)

    return df_scaled, scaler


def preprocess_pipeline(path, save_csv=True):
    # Load
    df = load_data(path)

    # Cleaning
    df_clean = clean_data(df)

    # Log transform
    df_log = log_transform(df_clean)

    # Scaling
    df_scaled, scaler = scale_data(df_log)

    # Save output
    if save_csv:
        df_scaled.to_csv('credit_card_scaled.csv', index=False)

    return df_scaled
