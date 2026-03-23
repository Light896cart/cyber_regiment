# =============================================================================
# utils/feature_selector.py
# Отбор признаков на основе важности из CatBoost
# =============================================================================

import polars as pl
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import gc


# Импортируем CatBoostManager локально, чтобы избежать циклических импортов
def select_features_catboost(
        X: pl.DataFrame,
        y: pl.DataFrame,
        cat_features: List[str],
        n_select: int = 500,
        config_path: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = True
) -> Tuple[List[str], pd.DataFrame]:
    """
    Отбирает топ-N признаков на основе важности из CatBoost.

    Args:
        X: Признаки (Polars DataFrame)
        y: Таргеты (Polars DataFrame, берём первый для оценки)
        cat_features: Список категориальных колонок
        n_select: Сколько признаков оставить
        config_path: Путь к конфигу CatBoost (опционально)
        random_state: Seed для воспроизводимости
        verbose: Выводить ли лог

    Returns:
        selected_features: Список отобранных признаков
        importance_df: DataFrame с важностью всех признаков
    """
    if verbose:
        print(f"🔍 Отбор признаков CatBoost (топ-{n_select})...")

    # Конвертируем в Pandas для CatBoost
    X_pd = X.to_pandas()
    y_pd = y.to_pandas()

    # Берём первый таргет для оценки важности (достаточно для отбора)
    y_multi = y_pd.values  # Shape: (n_samples, 41)

    if verbose:
        print(f"   📊 Таргетов для обучения: {y_multi.shape[1]}")

    # Быстрая оценка на подвыборке (ускоряет в 10-20 раз)
    if len(X_pd) > 5000000:
        sample_idx = np.random.choice(len(X_pd), 5000000, replace=False)
        X_sample = X_pd.iloc[sample_idx]
        y_sample = y_multi[sample_idx]
        if verbose:
            print(f"   📊 Используем подвыборку: {len(X_sample)} строк")
    else:
        X_sample = X_pd
        y_sample = y_multi

    # Импортируем CatBoost внутри функции
    from catboost import CatBoostClassifier, Pool


    # Параметры для быстрой оценки важности
    params = {
        'iterations': 1000,  # Мало итераций для скорости
        'depth': 6,
        'learning_rate': 0.1,
        'loss_function': 'MultiLogloss',
        'eval_metric': 'MultiLogloss',
        'random_seed': random_state,
        'task_type': 'GPU',  # CPU быстрее для малых итераций
        'verbose': False,
        'cat_features': cat_features,
        'early_stopping_rounds': 200
    }

    # Создаём Pool
    train_pool = Pool(
        data=X_sample,
        label=y_sample,  # ✅ Shape: (n_samples, 41)
        cat_features=cat_features
    )

    # Обучаем и получаем важность
    model = CatBoostClassifier(**params)
    model.fit(train_pool)

    # Получаем важность признаков
    importance = model.get_feature_importance()
    feature_names = model.feature_names_

    # Создаём DataFrame с важностью
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    # Отбираем топ-N
    selected_features = importance_df.head(n_select)['feature'].tolist()

    if verbose:
        print(f"   ✅ Отобрано {len(selected_features)} признаков из {len(feature_names)}")
        print(f"   📊 Топ-5 признаков:")
        for i, row in importance_df.head(5).iterrows():
            print(f"      {i + 1}. {row['feature']}: {row['importance']:.4f}")

    # Очистка памяти
    del X_pd, y_pd, X_sample, train_pool, model
    gc.collect()

    return selected_features, importance_df


def filter_dataframe_by_features(
        df: pl.DataFrame,
        selected_features: List[str],
        keep_cols: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Фильтрует DataFrame, оставляя только выбранные признаки.

    Args:
        df: Исходный DataFrame
        selected_features: Список признаков для сохранения
        keep_cols: Дополнительные колонки для сохранения (например, customer_id)

    Returns:
        Отфильтрованный DataFrame
    """
    cols_to_keep = [c for c in selected_features if c in df.columns]
    if keep_cols:
        cols_to_keep = [c for c in keep_cols if c in df.columns] + cols_to_keep

    return df.select(cols_to_keep)