# =============================================================================
# scripts/03_train_stage2_validate.py
# Stage 2: Валидация на hold-out (80/20) перед финальным обучением
# 🔥 ИЗМЕНЕНО: Поддержка объединённого OOF файла из Stage 1
# 🔧 ИСПРАВЛЕНО: Универсальные пути для команды и GitHub
# =============================================================================

import sys
import gc
import os
import argparse
from pathlib import Path
import json
from typing import List, Dict


# =============================================================================
# 🔧 АВТО-ОПРЕДЕЛЕНИЕ КОРНЯ ПРОЕКТА (как в loader.py)
# =============================================================================

def get_project_root() -> Path:
    """
    🔍 Автоматически находит корень проекта.
    Ищет по наличию .git, pyproject.toml или README.md
    """
    current = Path(__file__).resolve()

    # Поднимаемся вверх по директориям (максимум 10 уровней)
    for _ in range(10):
        if (current / ".git").exists() or \
                (current / "pyproject.toml").exists() or \
                (current / "README.md").exists():
            return current
        current = current.parent

    # Fallback: родительская директория от этого файла
    return Path(__file__).resolve().parent.parent


# 🔥 КОРЕНЬ ПРОЕКТА (авто-определение)
PROJECT_ROOT = get_project_root()

# 🔥 Пути относительно корня проекта (работают везде!)
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models_weight"
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_FOLDS_ROOT = PROJECT_ROOT / "folds"

# 🔥 Можно переопределить через env variables
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
MODELS_DIR = Path(os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR))
CONFIGS_DIR = Path(os.getenv("CONFIGS_DIR", DEFAULT_CONFIGS_DIR))
FOLDS_ROOT = Path(os.getenv("FOLDS_ROOT", DEFAULT_FOLDS_ROOT))

# Добавляем корень проекта в путь
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from models.catboost_model import CatBoostManager
from utils.meta_features import MetaFeaturesGenerator

import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# =============================================================================
# ✅ ФУНКЦИИ ДЛЯ ПРОВЕРКИ МОДЕЛЕЙ
# =============================================================================

def model_exists(model_name: str, fold_folder: str) -> bool:
    """Проверяет, существует ли уже обученная модель."""
    model_path = MODELS_DIR / fold_folder / model_name / "metadata.json"
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


def calculate_weighted_average_ensemble(
        predictions_dict: Dict[str, Dict[str, np.ndarray]],
        target_cols: List[str],
        weights: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """Создаёт взвешенный ансамбль предсказаний."""
    ensemble_predictions = {}

    for target in target_cols:
        pred_sum = None
        for model_name, model_preds in predictions_dict.items():
            if target in model_preds:
                weight = weights.get(model_name, 0.0)
                if pred_sum is None:
                    pred_sum = weight * model_preds[target]
                else:
                    pred_sum += weight * model_preds[target]

        if pred_sum is not None:
            # Нормализуем веса
            total_weight = sum(weights.get(m, 0.0) for m in predictions_dict.keys() if target in predictions_dict[m])
            if total_weight > 0:
                ensemble_predictions[target] = pred_sum / total_weight

    return ensemble_predictions


def create_best_model_per_target_ensemble(
        y_val_np: np.ndarray,
        predictions_dict: Dict[str, Dict[str, np.ndarray]],
        target_cols: List[str],
        model_names: List[str],
        verbose: bool = True
) -> tuple:
    """
    Для каждого таргета выбирает модель с лучшим ROC AUC.

    Returns:
        best_predictions: Dict[target_name, predictions]
        best_model_map: Dict[target_name, model_name]
        per_target_auc: Dict[target_name, Dict[model_name, auc]]
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🏆 BEST MODEL PER TARGET — АНАЛИЗ")
        print(f"{'=' * 60}")

    best_predictions = {}
    best_model_map = {}
    per_target_auc = {}

    model_wins = {model: 0 for model in model_names}

    for i, target in enumerate(target_cols):
        y_true = y_val_np[:, i]

        # Проверка на валидность таргета
        if len(np.unique(y_true)) < 2:
            continue

        # Считаем AUC для каждой модели
        model_aucs = {}
        for model_name in model_names:
            if model_name in predictions_dict and target in predictions_dict[model_name]:
                y_pred = predictions_dict[model_name][target]
                try:
                    auc = roc_auc_score(y_true, y_pred)
                    model_aucs[model_name] = auc
                except:
                    model_aucs[model_name] = 0.5
            else:
                model_aucs[model_name] = 0.5

        per_target_auc[target] = model_aucs

        # Выбираем лучшую модель
        best_model = max(model_aucs, key=model_aucs.get)
        best_model_map[target] = best_model
        model_wins[best_model] += 1

        # Берём предсказания лучшей модели
        best_predictions[target] = predictions_dict[best_model][target]

        if verbose and i < 10:  # Показываем первые 10 для примера
            print(f"   {target}: {best_model} (AUC={model_aucs[best_model]:.4f})")

    if verbose:
        print(f"\n   📊 Распределение 'побед' по моделям:")
        for model, wins in model_wins.items():
            print(f"      {model}: {wins} из {len(target_cols)} таргетов ({wins / len(target_cols) * 100:.1f}%)")

    return best_predictions, best_model_map, per_target_auc


def calculate_ensemble_auc(
        y_val_np: np.ndarray,
        predictions: Dict[str, np.ndarray],
        target_cols: List[str]
) -> float:
    """Считает macro ROC AUC для ансамбля."""
    aucs = []
    for i, target in enumerate(target_cols):
        y_true = y_val_np[:, i]
        if target in predictions and len(np.unique(y_true)) >= 2:
            try:
                auc = roc_auc_score(y_true, predictions[target])
                aucs.append(auc)
            except:
                pass
    return np.mean(aucs) if aucs else 0.0


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
    parser.add_argument('--test-size', type=float, default=0.05,
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
    print(f"   📁 Проект: {PROJECT_ROOT}")
    print(f"{'=' * 60}")

    # =========================================================
    # 1. Инициализация
    # =========================================================
    loader = DataLoader(cat_strategy="int")
    meta_generator = MetaFeaturesGenerator(
        artifacts_dir=str(ARTIFACTS_DIR)
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
    oof_path = ARTIFACTS_DIR / "oof_predictions_STACKED_stage1.parquet"

    if not oof_path.exists():
        raise FileNotFoundError(
            f"❌ OOF файл не найден: {oof_path}\n"
            f"💡 Запусти сначала: python scripts/02_stage1_proxy_training.py"
        )

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

    # 🔥 Сохраняем промежуточные файлы
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_extended.to_parquet(ARTIFACTS_DIR / "X_train_extended_val.parquet", index=False)
    X_val_extended.to_parquet(ARTIFACTS_DIR / "X_val_extended_val.parquet", index=False)
    y_train_pd.to_parquet(ARTIFACTS_DIR / "y_train_val.parquet", index=False)
    y_val_pd.to_parquet(ARTIFACTS_DIR / "y_val_val.parquet", index=False)

    X_train_pl = pl.from_pandas(X_train_extended)
    y_train_pl = pl.from_pandas(y_train_pd)
    X_val_pl = pl.from_pandas(X_val_extended)
    y_val_pl = pl.from_pandas(y_val_pd)

    USE_FEATURE_SELECTION = False

    if USE_FEATURE_SELECTION:
        print(f"\n{'=' * 60}")
        print(f"🔍 FEATURE SELECTION ДЛЯ STAGE 2")
        print(f"{'=' * 60}")

        from utils.feature_selector import select_features_catboost

        selected_features, importance_df = select_features_catboost(
            X=X_train_pl,
            y=y_train_pl,
            cat_features=loader.cat_features,
            n_select=1000,
            random_state=42,
            verbose=True
        )

        importance_df.to_csv(ARTIFACTS_DIR / "feature_importance_stage2.csv", index=False)
        print(f"   💾 Важность сохранена: artifacts/feature_importance_stage2.csv")

        X_train_pl = X_train_pl.select(selected_features)
        X_val_pl = X_val_pl.select(selected_features)

        X_train_extended = X_train_extended[
            selected_features + [c for c in X_train_extended.columns if c in df_meta_train.columns]]
        X_val_extended = X_val_extended[
            selected_features + [c for c in X_val_extended.columns if c in df_meta_val.columns]]

        cat_features_filtered = [f for f in loader.cat_features if f in selected_features]

        print(f"   ✅ Признаков после отбора: {len(selected_features)} (было 2168)")
        print(f"   🏷️  Категориальных признаков: {len(cat_features_filtered)} (было {len(loader.cat_features)})")
        print(f"   📊 Экономия памяти: ~{100 * (1 - len(selected_features) / 2168):.1f}%")

    print(f"\n{'=' * 60}")
    print(f"🔍 ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ ВАЖНОСТИ ПРИЗНАКОВ")
    print(f"{'=' * 60}")

    importance_path = ARTIFACTS_DIR / "feature_importance_stage2.csv"

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
        raise FileNotFoundError(
            f"❌ Файл важности не найден: {importance_path}\n"
            f"💡 Запусти сначала feature selection или создай файл вручную"
        )

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

    # =========================================
    # CatBoost
    # =========================================
    if 'catboost' in train_models:
        print(f"\n{'=' * 40}")
        print(f"🌲 CATBOOST")
        print(f"{'=' * 40}")

        if not args.force_retrain and model_exists("stage2_catboost_validation_v1", "catboost"):
            print(f"   ⏭️  Пропускаем CatBoost (уже обучен)")
            manager = load_existing_model("stage2_catboost_validation_v1", "catboost", CatBoostManager)
            auc_cb = manager.metadata['metrics']['macro_roc_auc']
        else:
            config_path = CONFIGS_DIR / "catboost" / "catboost_config_stage2.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"❌ Конфиг не найден: {config_path}")

            manager = CatBoostManager(
                config_path=str(config_path),
                fold_folder="catboost"
            )

            manager.model_params['use_best_model'] = True

            preds, auc_cb = manager.train(
                X_train=X_train_pl,
                y_train=y_train_pl,
                X_val=X_val_pl,
                y_val=y_val_pl,
                cat_features=cat_features_filtered,
                version_name="stage2_catboost_validation_v1",
                save_model=True,
                verbose=True
            )

            gc.collect()

        results_dict['catboost'] = {'manager': manager, 'auc': auc_cb}
        print(f"   📈 CatBoost AUC: {auc_cb:.4f}")
    else:
        print(f"\n   ⏭️  Пропускаем CatBoost (не в списке)")
        if model_exists("stage2_catboost_validation_v1", "catboost"):
            manager = load_existing_model("stage2_catboost_validation_v1", "catboost", CatBoostManager)
            auc_cb = manager.metadata['metrics']['macro_roc_auc']
            results_dict['catboost'] = {'manager': manager, 'auc': auc_cb}
        else:
            print(f"   ⚠️  CatBoost не обучен и не в списке!")
            auc_cb = 0.0

    # =========================================
    # Neural Network
    # =========================================
    if 'nn' in train_models:
        print(f"\n{'=' * 40}")
        print(f"🧠 NEURAL NETWORK")
        print(f"{'=' * 40}")

        from models.nn_model import NNManager

        if not args.force_retrain and model_exists("stage2_nn_validation_v1", "neural_network"):
            print(f"   ⏭️  Пропускаем NN (уже обучена)")
            manager_nn = load_existing_model("stage2_nn_validation_v1", "neural_network", NNManager)
            auc_nn = manager_nn.metadata['metrics']['macro_roc_auc']
        else:
            config_path = CONFIGS_DIR / "neural_network" / "nn_config_stage2.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"❌ Конфиг не найден: {config_path}")

            manager_nn = NNManager(
                config_path=str(config_path),
                fold_folder="neural_network"
            )

            preds_nn, auc_nn = manager_nn.train(
                X_train=X_train_pl,
                y_train=y_train_pl,
                X_val=X_val_pl,
                y_val=y_val_pl,
                cat_features=cat_features_filtered,
                version_name="stage2_nn_validation_v1",
                save_model=True,
                verbose=True
            )

            gc.collect()

        results_dict['nn'] = {'manager': manager_nn, 'auc': auc_nn}
        print(f"   📈 NN AUC: {auc_nn:.4f}")
    else:
        print(f"\n   ⏭️  Пропускаем NN (не в списке)")
        if model_exists("stage2_nn_validation_v1", "neural_network"):
            from models.nn_model import NNManager
            manager_nn = load_existing_model("stage2_nn_validation_v1", "neural_network", NNManager)
            auc_nn = manager_nn.metadata['metrics']['macro_roc_auc']
            results_dict['nn'] = {'manager': manager_nn, 'auc': auc_nn}
        else:
            auc_nn = 0.0

    # =========================================
    # LightGBM
    # =========================================
    if 'lgbm' in train_models:
        print(f"\n{'=' * 40}")
        print(f"🌳 LIGHTGBM")
        print(f"{'=' * 40}")

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   🧹 GPU память очищена перед LightGBM")

        from models.lgbm_model import LGBMManager

        if not args.force_retrain and model_exists("stage2_lgbm_validation_v1", "lightgbm"):
            print(f"   ⏭️  Пропускаем LGBM (уже обучен)")
            manager_lgbm = load_existing_model("stage2_lgbm_validation_v1", "lightgbm", LGBMManager)
            auc_lgbm = manager_lgbm.metadata['metrics']['macro_roc_auc']
        else:
            config_path = CONFIGS_DIR / "lightgbm" / "lgbm_config_stage2.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"❌ Конфиг не найден: {config_path}")

            manager_lgbm = LGBMManager(
                config_path=str(config_path),
                fold_folder="lightgbm"
            )

            preds_lgbm, auc_lgbm = manager_lgbm.train(
                X_train=X_train_pl,
                y_train=y_train_pl,
                X_val=X_val_pl,
                y_val=y_val_pl,
                cat_features=cat_features_filtered,
                version_name="stage2_lgbm_validation_v1",
                save_model=True,
                verbose=True
            )

            gc.collect()

        results_dict['lgbm'] = {'manager': manager_lgbm, 'auc': auc_lgbm}
        print(f"   📈 LGBM AUC: {auc_lgbm:.4f}")
    else:
        print(f"\n   ⏭️  Пропускаем LGBM (не в списке)")
        if model_exists("stage2_lgbm_validation_v1", "lightgbm"):
            from models.lgbm_model import LGBMManager
            manager_lgbm = load_existing_model("stage2_lgbm_validation_v1", "lightgbm", LGBMManager)
            auc_lgbm = manager_lgbm.metadata['metrics']['macro_roc_auc']
            results_dict['lgbm'] = {'manager': manager_lgbm, 'auc': auc_lgbm}
        else:
            auc_lgbm = 0.0

    # =========================================
    # Сравнение моделей
    # =========================================
    print(f"\n{'=' * 60}")
    print(f"📊 СРАВНЕНИЕ МОДЕЛЕЙ STAGE 2")
    print(f"{'=' * 60}")
    print(f"   🌲 CatBoost AUC: {auc_cb:.4f}")
    print(f"   🧠 Neural Net AUC: {auc_nn:.4f}")
    print(f"   🌳 LightGBM AUC: {auc_lgbm:.4f}")

    valid_aucs = [(auc_cb, 'CatBoost'), (auc_nn, 'NN'), (auc_lgbm, 'LGBM')]
    valid_aucs = [(a, n) for a, n in valid_aucs if a > 0]
    if valid_aucs:
        print(f"   🏆 Лучшая: {max(valid_aucs, key=lambda x: x[0])[1]}")

    # =========================================================
    # 8. Сравнение Stage 1 vs Stage 2
    # =========================================================
    print(f"\n{'=' * 60}")
    print(f"📊 СРАВНЕНИЕ STAGE 1 vs STAGE 2")
    print(f"{'=' * 60}")

    stage1_results_path = ARTIFACTS_DIR / "stage1_cv_results.json"
    if stage1_results_path.exists():
        with open(stage1_results_path, 'r') as f:
            stage1_data = json.load(f)
        stage1_auc = stage1_data.get('mean_cv_auc', 0.7932)
    else:
        stage1_auc = 0.5932

    stage2_best_auc = max(auc_cb, auc_nn, auc_lgbm)
    improvement_cb = auc_cb - stage1_auc
    improvement_nn = auc_nn - stage1_auc
    improvement_lgbm = auc_lgbm - stage1_auc

    print(f"   📈 Stage 1 CV AUC: {stage1_auc:.4f}")
    print(f"   📈 Stage 2 CatBoost AUC: {auc_cb:.4f} (прирост: {improvement_cb:+.4f})")
    print(f"   📈 Stage 2 NN AUC: {auc_nn:.4f} (прирост: {improvement_nn:+.4f})")
    print(f"   📈 Stage 2 LGBM AUC: {auc_lgbm:.4f} (прирост: {improvement_lgbm:+.4f})")
    print(f"   🏆 Лучшая модель: {max(valid_aucs, key=lambda x: x[0])[1] if valid_aucs else 'N/A'}")

    if stage2_best_auc - stage1_auc > 0.003:
        print(f"   ✅ Прирост значимый (> 0.003) → можно обучать на 100%")
    elif stage2_best_auc - stage1_auc > 0:
        print(f"   ⚠️  Прирост есть, но маленький (< 0.003)")
    else:
        print(f"   ❌ Прироста нет → проверить мета-признаки")

    # =========================================================
    # 9. Оптимизация весов ансамбля + 🔥 Best Model Per Target
    # =========================================================
    print(f"\n{'=' * 60}")
    print(f"⚖️  ОПТИМИЗАЦИЯ ВЕСОВ АНСАМБЛЯ (PER-TARGET)")
    print(f"{'=' * 60}")

    from utils.ensemble_optimizer_per_target import EnsembleWeightOptimizerPerTarget

    y_val_np = y_val_pd.to_numpy()

    optimizer = EnsembleWeightOptimizerPerTarget(y_val_np, target_cols)

    # 🔥 СОЗДАЁМ predictions_dict
    predictions_dict = {}
    if 'catboost' in results_dict and results_dict['catboost']['auc'] > 0:
        preds_cb_val = results_dict['catboost']['manager'].predict(X_val_pl, cat_features=cat_features_filtered)
        optimizer.add_model_predictions('catboost', preds_cb_val)
        predictions_dict['catboost'] = preds_cb_val
        print(f"   ✅ Добавлен CatBoost")

    if 'nn' in results_dict and results_dict['nn']['auc'] > 0:
        preds_nn_val = results_dict['nn']['manager'].predict(X_val_pl, cat_features=cat_features_filtered)
        optimizer.add_model_predictions('neural_network', preds_nn_val)
        predictions_dict['neural_network'] = preds_nn_val
        print(f"   ✅ Добавлен NN")

    if 'lgbm' in results_dict and results_dict['lgbm']['auc'] > 0:
        preds_lgbm_val = results_dict['lgbm']['manager'].predict(X_val_pl, cat_features=cat_features_filtered)
        optimizer.add_model_predictions('lightgbm', preds_lgbm_val)
        predictions_dict['lightgbm'] = preds_lgbm_val
        print(f"   ✅ Добавлен LGBM")

    # 🔥 PER-TARGET ОПТИМИЗАЦИЯ (ТОЛЬКО differential_evolution)
    target_weights = optimizer.optimize_weights(
        n_iterations=70,
        verbose=True,
        n_jobs=1
    )

    # 🔥 Сохраняем per-target веса
    mapping_path = ARTIFACTS_DIR / "best_model_per_target_map.json"
    optimizer.save(mapping_path, metadata={
        'stage': 'stage2_validation',
        'n_iterations': 70,
        'method': 'differential_evolution_per_target'
    })

    print(f"\n💾 Per-target веса сохранены: {mapping_path}")
    print(f"   📊 Таргетов: {len(target_weights)}")

    # =============================================================================
    # 🔥 ОЦЕНКА КАЧЕСТВА PER-TARGET ВЕСОВ
    # =============================================================================
    print(f"\n{'=' * 60}")
    print(f"📊 ОЦЕНКА КАЧЕСТВА PER-TARGET ВЕСОВ")
    print(f"{'=' * 60}")

    # 1. Считаем AUC с per-target весами
    per_target_predictions = optimizer.get_blended_predictions()
    per_target_auc = calculate_ensemble_auc(y_val_np, per_target_predictions, target_cols)
    print(f"\n   🏆 Per-Target Ensemble AUC: {per_target_auc:.4f}")

    # 2. Считаем AUC с равными весами (baseline)
    equal_weights = {m: 1.0 / len(predictions_dict) for m in predictions_dict.keys()}
    equal_predictions = calculate_weighted_average_ensemble(
        predictions_dict, target_cols, equal_weights
    )
    equal_auc = calculate_ensemble_auc(y_val_np, equal_predictions, target_cols)
    print(f"   📊 Equal Weights Baseline:  {equal_auc:.4f}")

    # 3. Прирост
    improvement = per_target_auc - equal_auc
    print(f"\n   📈 Прирост от per-target оптимизации: {improvement:+.4f}")

    if improvement > 0.001:
        print(f"   ✅ Per-target оптимизация УЛУЧШАЕТ результат")
        best_method = 'per_target_weights'
    else:
        print(f"   ⚠️  Прирост незначительный (< 0.001)")
        best_method = 'per_target_weights'  # Всё равно используем per-target

    comparison_results = {
        'per_target_auc': float(per_target_auc),
        'equal_weights_auc': float(equal_auc),
        'improvement': float(improvement),
        'best_method': best_method,
        'target_weights': {k: {mk: float(mv) for mk, mv in v.items()} for k, v in target_weights.items()},
        'n_targets': len(target_cols),
        'n_models': len(predictions_dict)
    }

    # 🔥 Сохраняем full mapping для inference
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            'method': 'per_target_differential_evolution',
            'target_weights': target_weights,
            'model_names': list(predictions_dict.keys()),
            'comparison': comparison_results,
            'metadata': {
                'stage': 'stage2_validation',
                'n_iterations': 70,
                'date': pd.Timestamp.now().isoformat()
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n   💾 Full mapping сохранён: {mapping_path}")

    # =========================================================
    # 10. Сохранение результатов
    # =========================================================
    results = {
        'stage1_auc': stage1_auc,
        'stage2_cb_auc': auc_cb,
        'stage2_nn_auc': auc_nn,
        'stage2_lgbm_auc': auc_lgbm,
        'stage2_best_auc': stage2_best_auc,
        'improvement_cb': improvement_cb,
        'improvement_nn': improvement_nn,
        'improvement_lgbm': improvement_lgbm,
        'n_features_original': X_full.shape[1],
        'n_features_meta': df_meta_train.shape[1],
        'n_features_total': X_train_extended.shape[1],
        'train_size': len_X_train_pd,
        'val_size': len_X_val_pd,
        'best_model': max(valid_aucs, key=lambda x: x[0])[1] if valid_aucs else 'N/A',
        'per_target_auc': per_target_auc,
        'equal_weights_auc': equal_auc,
        'ensemble_improvement': improvement,
        'trained_models': train_models,
        'ensemble_comparison': comparison_results,
        'best_ensemble_method': best_method,
        'mapping_path': str(mapping_path),
        'project_root': str(PROJECT_ROOT)
    }

    with open(ARTIFACTS_DIR / "stage2_validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✅ ВАЛИДАЦИЯ STAGE 2 ЗАВЕРШЕНА")
    print(f"   📁 Результаты: {ARTIFACTS_DIR / 'stage2_validation_results.json'}")
    print(f"   📁 Веса: {mapping_path}")
    print(f"   📁 Модель: {MODELS_DIR / 'catboost' / 'stage2_catboost_validation_v1'}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()