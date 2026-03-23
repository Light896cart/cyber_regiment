
import sys
import gc
import os
import argparse
from pathlib import Path
import json
from typing import List

ROOT_DIR = Path(r"D:\Code\hackaton_cyberpolka_CV")
sys.path.append(str(ROOT_DIR))

from src.data.loader import DataLoader
from models.catboost_model import CatBoostManager
from utils.meta_features import MetaFeaturesGenerator

import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import json
import datetime
import lightgbm as lgb
from models.lgbm_model import LGBMManager

# =============================================================================
# ✅ ФУНКЦИИ ДЛЯ ПРОВЕРКИ МОДЕЛЕЙ
# =============================================================================

def model_exists(model_name: str, fold_folder: str) -> bool:
    """Проверяет, существует ли уже обученная модель."""
    model_path = ROOT_DIR / "models_weight" / fold_folder / model_name / "metadata.json"
    exists = model_path.exists()
    if exists:
        print(f"   ✅ Модель уже существует: {model_name}")
    else:
        print(f"   ❌ Модель не найдена: {model_name}")
    return exists


def load_existing_model(model_name: str, fold_folder: str, model_class):
    """Загружает существующую модель."""
    manager = model_class()
    manager.load_model(model_name, fold_folder=fold_folder)
    return manager


# =============================================================================
# 🔥 НОВАЯ ФУНКЦИЯ: Извлечение OOF для одной модели из объединённого файла
# =============================================================================

def extract_model_oof_from_stacked(
        df_stacked: pd.DataFrame,
        model_name: str,
        target_cols: List[str]
) -> pd.DataFrame:
    """
    Извлекает предсказания одной модели из объединённого STACKED файла.
    Колонки: catboost_target_0, catboost_target_1, ... → target_0, target_1, ...
    """
    df_model = pd.DataFrame()

    for target in target_cols:
        col_name = f"{model_name}_{target}"
        if col_name in df_stacked.columns:
            df_model[target] = df_stacked[col_name].values
        else:
            print(f"   ⚠️  Колонка {col_name} не найдена!")
            return None

    print(f"   ✅ Извлечено {len(df_model.columns)} таргетов для {model_name}")
    return df_model


# =============================================================================
# ОСНОВНОЙ КОД
# =============================================================================

def main():
    # =========================================================
    # 0. Парсинг аргументов командной строки
    # =========================================================
    parser = argparse.ArgumentParser(description='Stage 2 Validation с выбором моделей')
    parser.add_argument('--models', type=str, default='all',
                        help='Модели для обучения: all, catboost, nn, lgbm, catboost+nn, etc.')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Переобучить даже если модель существует')
    parser.add_argument('--test-size', type=float, default=0.5,
                        help='Размер валидации (по умолчанию 0.05 = 5%)')

    args = parser.parse_args()

    if args.models == 'all':
        train_models = ['catboost', 'nn', 'lgbm']
    else:
        train_models = [m.strip().lower() for m in args.models.split(',')]

    print(f"\n{'=' * 60}")
    print(f"🔍 STAGE 2: ВАЛИДАЦИЯ")
    print(f"{'=' * 60}")
    print(f"   📋 Модели для обучения: {train_models}")
    print(f"   🔄 Переобучение: {'Да' if args.force_retrain else 'Нет'}")
    print(f"   📊 Test size: {args.test_size * 100:.0f}%")
    print(f"{'=' * 60}")

    # =========================================================
    # 1. Инициализация
    # =========================================================
    loader = DataLoader(cat_strategy="int")
    meta_generator = MetaFeaturesGenerator(
        artifacts_dir=str(ROOT_DIR / "artifacts")
    )
    loader.load_full_data()
    meta_generator.load_correlation_matrix()
    target_cols = loader.target_cols  # ← Сохраняем для извлечения OOF

    # =========================================================
    # 2. Загрузка данных
    # =========================================================
    print(f"\n📂 Загрузка данных...")

    X_full, y_full = loader.get_full_data()
    print(f"   📊 Всего данных: {X_full.shape[0]} строк, {X_full.shape[1]} признаков")

    # =========================================================
    # 3. Разделение на train/val (80/20)
    # =========================================================
    print(f"\n✂️  Разделение {100 * (1 - args.test_size):.0f}/{args.test_size * 100:.0f}...")

    X_full_pd = X_full.to_pandas()
    y_full_pd = y_full.to_pandas()
    del y_full

    y_sum = y_full_pd.sum(axis=1)

    X_train_pd, X_val_pd, y_train_pd, y_val_pd, _, val_sum = train_test_split(
        X_full_pd, y_full_pd, y_sum,
        test_size=args.test_size,
        random_state=42,
        stratify=y_sum
    )

    del X_full_pd, y_full_pd
    print(f"   📊 Train: {X_train_pd.shape[0]} строк")
    print(f"   📊 Val: {X_val_pd.shape[0]} строк")

    # =========================================================
    # 🔥 4. Генерация предсказаний Stage 1 (ИСПРАВЛЕНО!)
    # =========================================================
    print(f"\n🔮 Генерация Stage 1 предсказаний для мета-признаков...")

    # 🔥 ИЗМЕНЕНИЕ 1: Правильный путь к объединённому файлу
    oof_path = ROOT_DIR / "artifacts" / "oof_predictions_STACKED_stage1.parquet"

    if not oof_path.exists():
        raise FileNotFoundError(f"OOF файл не найден: {oof_path}")

    df_oof_stacked = pd.read_parquet(oof_path)
    print(f"   📊 Загружен STACKED OOF: {df_oof_stacked.shape}")

    # 🔥 ИЗМЕНЕНИЕ 2: Извлекаем CatBoost OOF для мета-признаков
    # (CatBoost обычно лучший для генерации мета-признаков)
    df_oof = extract_model_oof_from_stacked(
        df_oof_stacked,
        model_name='catboost',  # ← Используем CatBoost для мета-признаков
        target_cols=target_cols
    )

    if df_oof is None:
        raise ValueError("Не удалось извлечь CatBoost OOF из STACKED файла!")

    # Разделяем OOF предсказания так же как данные
    oof_indices = np.arange(len(df_oof))
    train_idx, val_idx, _, _ = train_test_split(
        oof_indices, np.zeros(len(df_oof)),
        test_size=args.test_size,
        random_state=42,
        stratify=y_sum
    )

    df_oof_train = df_oof.iloc[train_idx].reset_index(drop=True)
    df_oof_val = df_oof.iloc[val_idx].reset_index(drop=True)

    print(f"   📊 OOF Train: {df_oof_train.shape}")
    print(f"   📊 OOF Val: {df_oof_val.shape}")

    # =========================================================
    # 5. Генерация мета-признаков
    # =========================================================
    print(f"\n🔧 Генерация мета-признаков...")

    df_meta_train = meta_generator.generate_from_dataframe(df_oof_train, n_best_corr=15)
    df_meta_val = meta_generator.generate_from_dataframe(df_oof_val, n_best_corr=15)

    print(f"   📊 Мета-признаки Train: {df_meta_train.shape}")
    print(f"   📊 Мета-признаки Val: {df_meta_val.shape}")

    # =========================================================
    # 6. Объединение признаков
    # =========================================================
    print(f"\n🔗 Объединение признаков...")

    X_train_extended = pd.concat([
        X_train_pd.reset_index(drop=True),
        df_meta_train.reset_index(drop=True)
    ], axis=1)

    X_val_extended = pd.concat([
        X_val_pd.reset_index(drop=True),
        df_meta_val.reset_index(drop=True)
    ], axis=1)

    print(f"   📊 Train extended: {X_train_extended.shape[1]} признаков")
    print(f"   📊 Val extended: {X_val_extended.shape[1]} признаков")

    len_X_train_pd = len(X_train_pd)
    len_X_val_pd = len(X_val_pd)

    del X_train_pd

    X_train_extended.to_parquet(ROOT_DIR / "artifacts" / "X_train_extended_val.parquet", index=False)
    X_val_extended.to_parquet(ROOT_DIR / "artifacts" / "X_val_extended_val.parquet", index=False)
    y_train_pd.to_parquet(ROOT_DIR / "artifacts" / "y_train_val.parquet", index=False)
    y_val_pd.to_parquet(ROOT_DIR / "artifacts" / "y_val_val.parquet", index=False)

    X_train_pl = pl.from_pandas(X_train_extended)
    y_train_pl = pl.from_pandas(y_train_pd)
    X_val_pl = pl.from_pandas(X_val_extended)
    y_val_pl = pl.from_pandas(y_val_pd)

    print(f"\n{'=' * 60}")
    print(f"🔍 ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ ВАЖНОСТИ ПРИЗНАКОВ")
    print(f"{'=' * 60}")

    importance_path = ROOT_DIR / "artifacts" / "feature_importance_stage2.csv"

    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)
        print(f"   ✅ Загружено: {len(importance_df)} признаков (всего в файле)")

        # 🔥 БЕРЁМ ТОП-1500 ВМЕСТО ТОП-1000
        n_select = 1500  # ← ИЗМЕНЕНИЕ
        selected_features = importance_df.head(n_select)['feature'].tolist()

        print(f"   ✅ Отобрано топ-{n_select} признаков")
        print(f"   📊 Топ-5 признаков:")
        for i, row in importance_df.head(5).iterrows():
            print(f"      {i + 1}. {row['feature']}: {row['importance']:.4f}")

        # Проверка сколько ещё осталось
        print(f"   📊 Ещё доступно: {len(importance_df) - n_select} признаков")
    else:
        raise FileNotFoundError(f"Файл важности не найден: {importance_path}")

    # Дальше используем selected_features как обычно
    X_train_pl = X_train_pl.select(selected_features)
    X_val_pl = X_val_pl.select(selected_features)

    cat_features_filtered = [f for f in loader.cat_features if f in selected_features]

    print(f"   ✅ Признаков после отбора: {len(selected_features)}")
    print(f"   🏷️  Категориальных признаков: {len(cat_features_filtered)}")

    # =========================================================
    # 7. Обучение Stage 2 с валидацией
    # =========================================================
    print(f"\n{'=' * 60}")
    print(f"🎯 ОБУЧЕНИЕ STAGE 2 (с валидацией)")
    print(f"{'=' * 60}")

    results_dict = {}

    # =============================================================================
    # 🔥 LIGHTGBM TUNING & TRAINING (Встроено в Stage 2)
    # =============================================================================

    # --- Настройки путей для истории тюнинга (как в tune_lgbm_single.py) ---
    OPTUNA_RESULTS_DIR = ROOT_DIR / "artifacts" / "optuna_results_lgbm"
    OPTUNA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH = OPTUNA_RESULTS_DIR / "best_params.json"
    TRIALS_HISTORY_PATH = OPTUNA_RESULTS_DIR / "trials_history.json"

    # --- Вспомогательные функции (адаптированы для вставки) ---
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def load_best_params():
        trial_count = 0
        history = []
        if TRIALS_HISTORY_PATH.exists():
            try:
                with open(TRIALS_HISTORY_PATH, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    trial_count = len(history)
            except:
                pass

        best_params = None
        best_score = 0
        if BEST_PARAMS_PATH.exists():
            with open(BEST_PARAMS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            best_params = data.get('best_params', None)
            best_score = data.get('best_score', 0)
        return best_params, best_score, trial_count, history

    def suggest_param_from_history(history, param_name, default_range, best_params=None):
        # Упрощенная логика подбора (как в оригинале)
        if len(history) < 5 or best_params is None:
            if isinstance(default_range, tuple):
                if param_name in ['learning_rate', 'lambda_l1', 'lambda_l2']:
                    return 10 ** np.random.uniform(np.log10(default_range[0]), np.log10(default_range[1]))
                else:
                    return np.random.uniform(default_range[0], default_range[1])
            else:
                return np.random.randint(default_range[0], default_range[1])

        base_value = best_params.get(param_name, np.mean(default_range))
        if param_name in ['num_leaves', 'max_depth']:
            noise = np.random.randint(-3, 4)
            return int(np.clip(base_value + noise, default_range[0], default_range[1]))
        elif param_name in ['learning_rate', 'lambda_l1', 'lambda_l2']:
            noise = 10 ** np.random.uniform(-0.5, 0.5)
            return float(np.clip(base_value * noise, default_range[0], default_range[1]))
        elif param_name in ['feature_fraction', 'bagging_fraction']:
            noise = np.random.uniform(-0.2, 0.2)
            return float(np.clip(base_value + noise, default_range[0], default_range[1]))
        else:
            return base_value

    def generate_params(best_params=None, trial_number=0, history=None):
        history = history or []
        # Логика: если есть история и прошло > 5 триалов -> эксплуатация, иначе исследование
        if best_params and trial_number > 5:
            params = {
                'device': 'gpu', 'verbose': 10, 'seed': 42 + trial_number,
                'n_estimators': 450, 'early_stopping_round': 100,
                'num_leaves': suggest_param_from_history(history, 'num_leaves', (31, 127), best_params),
                'max_depth': suggest_param_from_history(history, 'max_depth', (6, 12), best_params),
                'learning_rate': suggest_param_from_history(history, 'learning_rate', (0.01, 0.3), best_params),
                'min_data_in_leaf': suggest_param_from_history(history, 'min_data_in_leaf', (5, 100), best_params),
                'feature_fraction': suggest_param_from_history(history, 'feature_fraction', (0.6, 1.0), best_params),
                'bagging_fraction': suggest_param_from_history(history, 'bagging_fraction', (0.6, 1.0), best_params),
                'bagging_freq': 1,
                'lambda_l1': suggest_param_from_history(history, 'lambda_l1', (0.0, 10.0), best_params),
                'lambda_l2': suggest_param_from_history(history, 'lambda_l2', (0.0, 10.0), best_params),
                'max_bin': 255,
            }
        else:
            params = {
                'device': 'gpu', 'verbose': 10, 'seed': 42 + trial_number,
                'n_estimators': 450, 'early_stopping_round': 100,
                'num_leaves': np.random.randint(31, 127),
                'max_depth': np.random.randint(6, 12),
                'learning_rate': 10 ** np.random.uniform(-2, -0.5),
                'min_data_in_leaf': np.random.randint(5, 100),
                'feature_fraction': np.random.uniform(0.6, 1.0),
                'bagging_fraction': np.random.uniform(0.6, 1.0),
                'bagging_freq': 1,
                'lambda_l1': 10 ** np.random.uniform(-3, 1),
                'lambda_l2': 10 ** np.random.uniform(-3, 1),
                'max_bin': 255,
            }
        return params

    # --- Основной блок обучения LightGBM ---
    print(f"\n{'=' * 60}")
    print(f"🚀 LIGHTGBM STAGE 2 (Tuning Trial)")
    print(f"{'=' * 60}")

    # 1. Загрузка истории параметров
    best_params, best_score, last_trial, history = load_best_params()
    trial_number = last_trial + 1
    print(f"   🔢 Trial: {trial_number} (Лучший AUC ранее: {best_score:.4f})")

    # 2. Генерация параметров
    params = generate_params(best_params, trial_number, history)

    # Проверка GPU
    use_gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            use_gpu = True
            print(f"   ✅ GPU доступен")
        else:
            print(f"   ⚠️  GPU не найден, переключаем на CPU")
            params['device'] = 'cpu'
            params['n_jobs'] = -1
    except:
        params['device'] = 'cpu'
        params['n_jobs'] = -1

    if use_gpu:
        params['gpu_use_dp'] = True
        params['force_col_wise'] = True

    print(f"   ⚙️  Параметры: num_leaves={params['num_leaves']}, lr={params['learning_rate']:.4f}")

    # 3. Обучение
    score = 0.0
    try:
        manager = LGBMManager(
            config_path=str(ROOT_DIR / "configs" / "lightgbm" / "lgbm_config.yaml"),
            save_dir=str(ROOT_DIR / "models_weight"),
            fold_folder="optuna_trials_lgbm"
        )
        manager.model_params.update(params)

        # 🔥 ИСПОЛЬЗУЕМ ДАННЫЕ ИЗ ПАМЯТИ (X_train_pl и т.д.), а не грузим заново
        _, auc = manager.train(
            X_train=X_train_pl,
            y_train=y_train_pl,
            X_val=X_val_pl,
            y_val=y_val_pl,
            cat_features=cat_features_filtered,
            version_name=f"trial_{trial_number}",
            save_model=False,
            verbose=10,
        )
        score = auc
        manager.clear()
        del manager
        gc.collect()

        print(f"   ✅ AUC получен: {score:.4f}")
    except Exception as e:
        print(f"   ❌ Ошибка обучения: {e}")
        score = 0.0

    # 4. Сохранение истории и лучших параметров
    # Сохранение в историю trials
    history_entry = {
        'trial_number': int(trial_number),
        'score': float(score),
        'params': convert_to_serializable(params),
        'datetime': datetime.datetime.now().isoformat()
    }

    # Читаем старую историю, добавляем новую, пишем обратно
    current_history = []
    if TRIALS_HISTORY_PATH.exists():
        try:
            with open(TRIALS_HISTORY_PATH, 'r', encoding='utf-8') as f:
                current_history = json.load(f)
        except:
            pass

    current_history.append(history_entry)
    with open(TRIALS_HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(current_history, f, indent=2, ensure_ascii=False)

    # Обновление best_params.json если есть улучшение
    if score > best_score:
        data = {
            'best_params': convert_to_serializable(params),
            'best_score': float(score),
            'trial_number': int(trial_number),
            'datetime': datetime.datetime.now().isoformat()
        }
        with open(BEST_PARAMS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"   💾 НОВЫЙ РЕКОРД! Сохранено в best_params.json")
    else:
        print(f"   ⏭️  Результат не улучшил рекорд ({best_score:.4f})")

    results_dict['lgbm'] = {
        'score': score,
        'trial': trial_number,
        'params': params
    }

    print(f"{'=' * 60}")
    print(f"✅ LIGHTGBM TRIAL ЗАВЕРШЕН")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()