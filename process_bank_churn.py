import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List, Dict, Optional

def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Видаляє вказані колонки з DataFrame."""
    return df.drop(columns=columns)

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Розбиває DataFrame на тренувальну та валідну частину, stratify за цільовою."""
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])

def select_columns(df: pd.DataFrame, include_cols: List[str]) -> pd.DataFrame:
    """Вибирає лише вказані колонки з DataFrame."""
    return df[include_cols].copy()

def get_numeric_and_categorical_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Повертає списки числових та об'єктних колонок у DataFrame."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols

def fit_scaler(df: pd.DataFrame, scale_cols: List[str]) -> StandardScaler:
    """Навчає масштабувальник на зазначених числових колонках."""
    scaler = StandardScaler()
    scaler.fit(df[scale_cols])
    return scaler

def transform_scaler(df: pd.DataFrame, scaler: StandardScaler, scale_cols: List[str]) -> pd.DataFrame:
    """Застосовує масштабування до даних."""
    df_scaled = df.copy()
    df_scaled[scale_cols] = scaler.transform(df_scaled[scale_cols])
    return df_scaled

def fit_onehot_encoder(train_df: pd.DataFrame, col: str) -> OneHotEncoder:
    """Навчає one-hot encoder на вказаній колонці."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[[col]])
    return encoder

def encode_column(train_df: pd.DataFrame, val_df: pd.DataFrame, col: str, encoder: OneHotEncoder) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Кодує вказану колонку через один hot encoding."""
    train_encoded = encoder.transform(train_df[[col]])
    val_encoded = encoder.transform(val_df[[col]])
    encoded_cols = encoder.get_feature_names_out([col])
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=train_df.index)
    val_encoded_df = pd.DataFrame(val_encoded, columns=encoded_cols, index=val_df.index)
    train_df = pd.concat([train_df.drop(columns=[col]), train_encoded_df], axis=1)
    val_df = pd.concat([val_df.drop(columns=[col]), val_encoded_df], axis=1)
    return train_df, val_df

def encode_binary_column(df: pd.DataFrame, col: str, true_value: str) -> pd.DataFrame:
    """Змінює бінарну колонку на 0 і 1, де true_value відповідає 1, а все інше 0."""
    df[col] = (df[col] == true_value).astype(int)
    return df

def encode_ordered_category(
    df: pd.DataFrame,
    col: str,
    mapping: Dict[str, int]
) -> pd.DataFrame:
    """Кодує категоріальну колонку за заданим порядком через словник."""
    df[col] = df[col].map(mapping)
    return df


def process_new_data(
    new_df: pd.DataFrame,
    scaler: Optional[StandardScaler],
    scale_cols: List[str],
    encoders: Dict[str, OneHotEncoder],
    binary_columns: Dict[str, str] = None,
    ordered_columns: Dict[str, Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Обробляє нові дані: масштабування, one-hot, бінарне та порядкове кодування.
    """
    df_processed = new_df.copy()

    # Масштабування
    if scaler and scale_cols:
        df_processed = transform_scaler(df_processed, scaler, scale_cols)

    # One-hot кодування
    for col, encoder in encoders.items():
        encoded = encoder.transform(df_processed[[col]])
        encoded_cols = encoder.get_feature_names_out([col])
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df_processed.index)
        df_processed = pd.concat([df_processed.drop(columns=[col]), encoded_df], axis=1)

    # Бінарне кодування
    if binary_columns:
        for col, true_val in binary_columns.items():
            df_processed = encode_binary_column(df_processed, col, true_val)

    # Порядкове кодування
    if ordered_columns:
        for col, mapping in ordered_columns.items():
            df_processed = encode_ordered_category(df_processed, col, mapping)

    return df_processed
