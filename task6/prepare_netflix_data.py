import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(df):
    """
    Подготавливает данные для моделирования
    
    Parameters:
    df - исходный DataFrame
    
    Returns:
    X_train, X_test, y_train, y_test, scaler
    """
    df_clean = df.copy()
    
    # удаляем customer_id (не нужен для обучения)
    if 'customer_id' in df_clean.columns:
        df_clean = df_clean.drop('customer_id', axis=1)
    
    # gender -> бинарный
    if 'gender' in df_clean.columns:
        df_clean['gender_male'] = (df_clean['gender'] == 'Male').astype(int)
        df_clean = df_clean.drop('gender', axis=1)
        print("Закодирован gender")
    
    # subscription_type - порядковый
    if 'subscription_type' in df_clean.columns:
        subscription_map = {'Basic': 0, 'Standard': 1, 'Premium': 2}
        df_clean['subscription_type_encoded'] = df_clean['subscription_type'].map(subscription_map)
        df_clean = df_clean.drop('subscription_type', axis=1)
        print("Закодирован subscription_type")
    
    # region - One-Hot
    if 'region' in df_clean.columns:
        region_dummies = pd.get_dummies(df_clean['region'], prefix='region')
        df_clean = pd.concat([df_clean, region_dummies], axis=1)
        df_clean = df_clean.drop('region', axis=1)
        print("Закодирован region")
    
    # device - One-Hot
    if 'device' in df_clean.columns:
        device_dummies = pd.get_dummies(df_clean['device'], prefix='device')
        df_clean = pd.concat([df_clean, device_dummies], axis=1)
        df_clean = df_clean.drop('device', axis=1)
        print("Закодирован device")
    
    # payment_method - One-Hot
    if 'payment_method' in df_clean.columns:
        payment_dummies = pd.get_dummies(df_clean['payment_method'], prefix='payment')
        df_clean = pd.concat([df_clean, payment_dummies], axis=1)
        df_clean = df_clean.drop('payment_method', axis=1)
        print("Закодирован payment_method")
    
    # favorite_genre - One-Hot
    if 'favorite_genre' in df_clean.columns:
        genre_dummies = pd.get_dummies(df_clean['favorite_genre'], prefix='genre')
        df_clean = pd.concat([df_clean, genre_dummies], axis=1)
        df_clean = df_clean.drop('favorite_genre', axis=1)
        print("Закодирован favorite_genre")
    
    # обработка выбросов в avg_watch_time_per_day
    upper_limit = df_clean['avg_watch_time_per_day'].quantile(0.99)
    df_clean['avg_watch_time_per_day'] = df_clean['avg_watch_time_per_day'].clip(upper=upper_limit)
    print(f"Avg_watch_time_per_day на 99-й перцентиль: {upper_limit:.2f}")
    
    # обработка выбросов в watch_hours
    upper_limit = df_clean['watch_hours'].quantile(0.99)
    df_clean['watch_hours'] = df_clean['watch_hours'].clip(upper=upper_limit)
    print(f"Watch_hours на 99-й перцентиль: {upper_limit:.2f}")
    
    #  monthly_fee - One-Hot
    if 'monthly_fee' in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['monthly_fee'], prefix='fee')
        print("Закодирован monthly_fee")
    
    # признак: отношение watch_hours к числу профилей
    df_clean['watch_hours_per_profile'] = df_clean['watch_hours'] / df_clean['number_of_profiles']
    
    # признак: активность (по факту обратная к last_login_days)
    df_clean['activity_score'] = 1 / (df_clean['last_login_days'] + 1)
    
    # признак: возраст по группам
    df_clean['age_group'] = pd.cut(df_clean['age'], 
                                   bins=[0, 30, 45, 60, 100], 
                                   labels=['young', 'middle', 'senior', 'very_elderly'])
    df_clean = pd.get_dummies(df_clean, columns=['age_group'], prefix='age')
    df_clean = df_clean.drop('age', axis=1)
    
    print(f"\nФорма после подготовки: {df_clean.shape}")
    print(f"Количество признаков: {df_clean.shape[1] - 1}")  # минус целевая переменная
    
    # разделяем на признаки и целевую переменную
    X = df_clean.drop('churned', axis=1)
    y = df_clean['churned']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # масштабирование (для будущей моей логистической регрессии)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nРазмер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой: {X_test.shape}")
    print(f"Баланс классов в обучении: {y_train.mean():.2%}")
    print(f"Баланс классов в тесте: {y_test.mean():.2%}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }