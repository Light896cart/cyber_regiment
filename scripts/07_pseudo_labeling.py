# =============================================================================
# scripts/07_pseudo_labeling.py
# 🔥 PSEUDO-LABELING: Дообучение на тестовых данных (ИСПРАВЛЕНО)
# 🔧 ИСПРАВЛЕНО: Универсальные пути для команды и GitHub
# =============================================================================

import sys
import gc
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score


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
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models_weight"
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_FOLDS_ROOT = PROJECT_ROOT / "folds"
DEFAULT_SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# 🔥 Можно переопределить через env variables
DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
MODELS_DIR = Path(os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR))
CONFIGS_DIR = Path(os.getenv("CONFIGS_DIR", DEFAULT_CONFIGS_DIR))
FOLDS_ROOT = Path(os.getenv("FOLDS_ROOT", DEFAULT_FOLDS_ROOT))
SUBMISSIONS_DIR = Path(os.getenv("SUBMISSIONS_DIR", DEFAULT_SUBMISSIONS_DIR))

# Добавляем корень проекта в путь
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from models.catboost_model import CatBoostManager
from models.lgbm_model import LGBMManager
from models.nn_model import NNManager
from utils.meta_features import MetaFeaturesGenerator

# =============================================================================
# ⚙️ КОНФИГУРАЦИЯ
# =============================================================================

PSEUDO_CONFIG = {
    'threshold_high': 0.95,
    'threshold_low': 0.05,
    'pseudo_sample_weight': 0.6,
    'min_pseudo_samples': 1000,
    'max_pseudo_samples': 50000,
    'models_to_retrain': ['lgbm'],
    'stratify_by_class': True,
    'save_pseudo_labels': True,
}


# =============================================================================
# 🔥 ФУНКЦИИ
# =============================================================================

def select_confident_predictions(
        predictions_dict: dict,
        target_cols: list,
        threshold_high: float = 0.95,
        threshold_low: float = 0.05,
        max_samples: int = 50000,
        stratify: bool = True
) -> tuple:
    """
    Отбирает уверенные предсказания для pseudo-labeling.
    """
    n_test = len(list(predictions_dict.values())[0])
    confident_mask = np.zeros(n_test, dtype=bool)
    pseudo_labels = {col: [] for col in target_cols}

    stats = {
        'total_test': n_test,
        'confident_per_target': {},
    }

    for col in target_cols:
        preds = predictions_dict[col]
        mask_high = preds >= threshold_high
        mask_low = preds <= threshold_low
        mask = mask_high | mask_low
        confident_mask |= mask

        labels = np.zeros(n_test)
        labels[mask_high] = 1
        labels[mask_low] = 0
        labels[~mask] = -1
        pseudo_labels[col] = labels

        n_confident = mask.sum()
        stats['confident_per_target'][col] = {
            'total': int(n_confident),
            'class_1': int(mask_high.sum()),
            'class_0': int(mask_low.sum()),
            'ratio': float(n_confident / n_test)
        }

    confident_indices = np.where(confident_mask)[0]

    if len(confident_indices) > max_samples:
        if stratify:
            max_confidence = np.zeros(n_test)
            for col in target_cols:
                preds = predictions_dict[col]
                confidence = np.maximum(preds, 1 - preds)
                max_confidence = np.maximum(max_confidence, confidence)
            sorted_idx = np.argsort(max_confidence[confident_indices])[::-1]
            confident_indices = confident_indices[sorted_idx[:max_samples]]
        else:
            confident_indices = confident_indices[:max_samples]

    stats['selected_pseudo_samples'] = len(confident_indices)
    stats['selection_ratio'] = len(confident_indices) / n_test

    return confident_indices, pseudo_labels, stats


def create_pseudo_train_dataset(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        pseudo_indices: np.ndarray,
        pseudo_labels: dict,
        target_cols: list,
        sample_weight: float = 0.6
) -> tuple:
    """
    Создаёт комбинированный датасет: Train + Pseudo-Test.
    🔥 ИСПРАВЛЕНО: Конвертируем псевдо-метки в int (0/1) для CatBoost
    """
    # Pseudo-samples из test
    X_pseudo = X_test.iloc[pseudo_indices].reset_index(drop=True)

    y_pseudo_data = {}
    for col in target_cols:
        # 🔥 ИСПРАВЛЕНО: Конвертируем в int (0/1) для CatBoost MultiLogloss
        labels = pseudo_labels[col][pseudo_indices].astype(np.int8)
        y_pseudo_data[col] = labels

    y_pseudo = pd.DataFrame(y_pseudo_data)

    # Объединяем
    X_combined = pd.concat([
        X_train.reset_index(drop=True),
        X_pseudo
    ], ignore_index=True)

    y_combined = pd.concat([
        y_train.reset_index(drop=True),
        y_pseudo
    ], ignore_index=True)

    # 🔥 ИСПРАВЛЕНО: Конвертируем ВСЕ таргеты в int8 для CatBoost
    for col in target_cols:
        y_combined[col] = y_combined[col].astype(np.int8)

    # Веса сэмплов
    n_train = len(X_train)
    n_pseudo = len(X_pseudo)

    sample_weights = np.ones(len(X_combined))
    sample_weights[n_train:] = sample_weight  # Pseudo-samples имеют меньший вес

    stats = {
        'n_train': n_train,
        'n_pseudo': n_pseudo,
        'n_combined': len(X_combined),
        'pseudo_ratio': n_pseudo / len(X_combined)
    }

    return X_combined, y_combined, sample_weights, stats


def generate_meta_features_for_pseudo(
        meta_generator: MetaFeaturesGenerator,
        stage1_models: list,
        X_pseudo: pd.DataFrame,
        cat_features: list,
        target_cols: list
) -> pd.DataFrame:
    """
    🔥 Генерирует мета-признаки для pseudo-samples через Stage 1 инференс.
    ИСПРАВЛЕНО: Теперь реально генерирует мета-признаки, не заглушку!
    """
    print(f"   🔄 Генерация мета-признаков для {len(X_pseudo)} pseudo-samples...")

    # 🔥 Stage 1 инференс на pseudo-samples
    test_preds_stage1 = {col: np.zeros(len(X_pseudo)) for col in target_cols}

    for i, model_manager in enumerate(stage1_models):
        X_pseudo_pl = pl.from_pandas(X_pseudo)
        preds = model_manager.predict(X_pseudo_pl, cat_features=cat_features)

        for col in target_cols:
            if col in preds:
                test_preds_stage1[col] += preds[col]

        # Не очищаем модели здесь - они понадобятся снова
        gc.collect()

    # Усредняем предсказания от 3 фолдов
    for col in target_cols:
        test_preds_stage1[col] /= len(stage1_models)

    # 🔥 Генерация мета-признаков через MetaFeaturesGenerator
    df_test_preds = pd.DataFrame(test_preds_stage1)
    df_meta_pseudo = meta_generator.generate_from_dataframe(df_test_preds, n_best_corr=15)

    print(f"   ✅ Мета-признаков: {df_meta_pseudo.shape[1]}")

    # 🔥 Очищаем Stage 1 модели после использования
    for model_manager in stage1_models:
        model_manager.clear()
    gc.collect()

    return df_meta_pseudo


def retrain_models_with_pseudo(
        X_combined: pd.DataFrame,
        y_combined: pd.DataFrame,
        sample_weights: np.ndarray,
        cat_features: list,
        models_to_retrain: list = ['catboost', 'lgbm']
) -> dict:
    """
    Дообучает модели на комбинированных данных.
    🔥 ИСПРАВЛЕНО: GPU + прогрев + 1500 признаков (как в Stage 2)
    """
    retrained_models = {}

    X_combined_pl = pl.from_pandas(X_combined)
    y_combined_pl = pl.from_pandas(y_combined)

    # 🔥 1. FEATURE SELECTION (строго 1500 признаков!)
    # 🔥 ИСПРАВЛЕНО: Используем ARTIFACTS_DIR из универсальных путей
    importance_path = ARTIFACTS_DIR / "feature_importance_stage2.csv"

    if importance_path.exists():
        imp_df = pd.read_csv(importance_path)
        selected_features = imp_df.head(1500)['feature'].tolist()  # ← СТРОГО 1500

        # Сохраняем мета-признаки
        meta_cols = [c for c in X_combined.columns if c.startswith('meta_')]
        feature_cols = [c for c in selected_features if c in X_combined.columns and not c.startswith('meta_')]
        all_cols = list(set(feature_cols + meta_cols))

        X_combined_pl = X_combined_pl.select(all_cols)
        cat_features = [f for f in cat_features if f in all_cols]

        print(f"   ✅ После Feature Selection: {len(all_cols)} признаков (строго 1500)")
    else:
        raise FileNotFoundError(
            f"❌ Файл важности не найден: {importance_path}\n"
            f"💡 Запусти сначала: python scripts/03_train_stage2_validate.py"
        )

    # =====================================================================
    # 🔥 LightGBM (С GPU И ПРОГРЕВОМ - как в Stage 2)
    # =====================================================================
    if 'lgbm' in models_to_retrain:
        print(f"\n🌳 Дообучение LightGBM...")

        # 🔥 ИСПРАВЛЕНО: Используем CONFIGS_DIR из универсальных путей
        config_path = CONFIGS_DIR / "lightgbm" / "lgbm_config_stage2.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"❌ Конфиг не найден: {config_path}")

        manager = LGBMManager(
            config_path=str(config_path),
            fold_folder="lightgbm_pseudo"
        )

        manager.model_params['n_estimators'] = 2000
        manager.model_params['early_stopping_round'] = 100
        manager.model_params['device'] = 'gpu'  # ← GPU как в Stage 2
        manager.model_params['gpu_use_dp'] = True  # ← Double precision
        manager.model_params['force_col_wise'] = True  # ← Стабильнее для GPU

        # 🔥 Прогрев GPU (как в Stage 2 validation)
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   🧹 GPU память очищена перед LightGBM")

        _, auc = manager.train(
            X_train=X_combined_pl,
            y_train=y_combined_pl,
            X_val=None,  # Нет валидации при дообучении
            y_val=None,
            cat_features=cat_features,
            version_name="stage2_lgbm_pseudo_v1",
            save_model=True,
            verbose=True
        )

        retrained_models['lgbm'] = {'manager': manager, 'auc': auc}
        print(f"   ✅ LGBM дообучен | AUC: {auc:.4f}")

    # =====================================================================
    # 🔥 CatBoost (С GPU)
    # =====================================================================
    if 'catboost' in models_to_retrain:
        print(f"\n🌲 Дообучение CatBoost...")

        # 🔥 ИСПРАВЛЕНО: Используем CONFIGS_DIR из универсальных путей
        config_path = CONFIGS_DIR / "catboost" / "catboost_config_stage2.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"❌ Конфиг не найден: {config_path}")

        manager = CatBoostManager(
            config_path=str(config_path),
            fold_folder="catboost_pseudo"
        )

        manager.model_params['iterations'] = 2000
        manager.model_params['early_stopping_rounds'] = 100
        manager.model_params['use_best_model'] = True
        manager.model_params['task_type'] = 'GPU'  # ← GPU
        manager.model_params['subsample'] = 0.8

        _, auc = manager.train(
            X_train=X_combined_pl,
            y_train=y_combined_pl,
            X_val=None,
            y_val=None,
            cat_features=cat_features,
            version_name="stage2_catboost_pseudo_v1",
            save_model=True,
            verbose=True
        )

        retrained_models['catboost'] = {'manager': manager, 'auc': auc}
        print(f"   ✅ CatBoost дообучен | AUC: {auc:.4f}")

    return retrained_models


# =============================================================================
# 🚀 ОСНОВНОЙ СКРИПТ
# =============================================================================

def main():
    print(f"\n{'=' * 70}")
    print(f"🔥 PSEUDO-LABELING: Дообучение на тестовых данных")
    print(f"{'=' * 70}")
    print(f"   📁 Проект: {PROJECT_ROOT}")
    print(f"   📁 Data: {DATA_DIR}")
    print(f"   📁 Artifacts: {ARTIFACTS_DIR}")
    print(f"   📁 Models: {MODELS_DIR}")
    print(f"{'=' * 70}")

    # =========================================================
    # 1. Загрузка данных
    # =========================================================
    print(f"\n📂 Загрузка данных...")

    loader = DataLoader(cat_strategy="int")
    loader.load_full_data()
    target_cols = loader.target_cols
    cat_features = loader.cat_features

    X_train, y_train = loader.get_full_data()
    X_train_pd = X_train.to_pandas()
    y_train_pd = y_train.to_pandas()

    print(f"   📊 Train: {X_train_pd.shape}")
    print(f"   📊 Таргетов: {len(target_cols)}")

    # Test данные
    # 🔥 ИСПРАВЛЕНО: Используем DATA_DIR из универсальных путей
    test_path = DATA_DIR / "test_final.parquet"

    if not test_path.exists():
        raise FileNotFoundError(
            f"❌ Тестовые данные не найдены: {test_path}\n"
            f"💡 Убедитесь что test_final.parquet существует в data/"
        )

    df_test = pl.read_parquet(test_path)
    customer_ids = df_test["customer_id"].to_numpy()

    X_test_raw = df_test.select([col for col in df_test.columns if col != 'customer_id'])
    X_test_pd_raw = X_test_raw.to_pandas()

    print(f"   📊 Test: {X_test_pd_raw.shape}")

    # =========================================================
    # 2. Загрузка Stage 1 моделей (для мета-признаков)
    # =========================================================
    print(f"\n📂 Загрузка Stage 1 моделей (для мета-признаков)...")

    n_splits_stage1 = 3
    stage1_model_names = [
        "catboost_fold_0_stage1",
        "catboost_fold_1_stage1",
        "catboost_fold_2_stage1"
    ]

    stage1_models = []
    for model_name in stage1_model_names:
        # 🔥 ИСПРАВЛЕНО: Используем MODELS_DIR и FOLDS_ROOT из универсальных путей
        model_path = MODELS_DIR / f"folds_{n_splits_stage1}" / model_name / "model.cbm"
        if model_path.exists():
            manager = CatBoostManager(
                config_path=str(CONFIGS_DIR / "catboost" / "catboost_config.yaml"),
                fold_folder=f"folds_{n_splits_stage1}"
            )
            manager.load_model(model_name, fold_folder=f"folds_{n_splits_stage1}")
            stage1_models.append(manager)
            print(f"   ✅ Загружено: {model_name}")
        else:
            raise FileNotFoundError(
                f"❌ Модель не найдена: {model_path}\n"
                f"💡 Запусти сначала: python scripts/02_stage1_proxy_training.py"
            )

    print(f"   📊 Stage 1 моделей: {len(stage1_models)}")

    # =========================================================
    # 3. Загрузка Stage 2 моделей + Feature Selection
    # =========================================================
    print(f"\n📂 Загрузка Stage 2 моделей...")

    # 🔥 1. ЗАГРУЗКА СПИСКА ПРИЗНАКОВ
    # 🔥 ИСПРАВЛЕНО: Используем ARTIFACTS_DIR из универсальных путей
    importance_path = ARTIFACTS_DIR / "feature_importance_stage2.csv"

    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)
        n_select = 1500
        selected_features = importance_df.head(n_select)['feature'].tolist()
        print(f"   ✅ Отобрано топ-{n_select} признаков")
    else:
        raise FileNotFoundError(
            f"❌ Файл важности не найден: {importance_path}\n"
            f"💡 Запусти сначала: python scripts/03_train_stage2_validate.py"
        )

    # 🔥 2. Фильтрация cat_features
    cat_features_filtered = [f for f in cat_features if f in selected_features]
    print(f"   ✅ Категориальных признаков: {len(cat_features_filtered)}")

    # 🔥 3. Загрузка Stage 2 моделей
    models = {}

    # 🔥 ИСПРАВЛЕНО: Используем CONFIGS_DIR из универсальных путей
    cb_config_path = CONFIGS_DIR / "catboost" / "catboost_config_stage2.yaml"
    if not cb_config_path.exists():
        raise FileNotFoundError(f"❌ Конфиг не найден: {cb_config_path}")

    cb_manager = CatBoostManager(
        config_path=str(cb_config_path),
        fold_folder="catboost"
    )
    cb_manager.load_model("stage2_catboost_validation_v1", fold_folder="catboost")
    models['catboost'] = cb_manager
    print(f"   ✅ CatBoost загружен")

    lgbm_config_path = CONFIGS_DIR / "lightgbm" / "lgbm_config_stage2.yaml"
    if not lgbm_config_path.exists():
        raise FileNotFoundError(f"❌ Конфиг не найден: {lgbm_config_path}")

    lgbm_manager = LGBMManager(
        config_path=str(lgbm_config_path),
        fold_folder="lightgbm"
    )
    lgbm_manager.load_model("stage2_lgbm_validation_v1", fold_folder="lightgbm")
    models['lgbm'] = lgbm_manager
    print(f"   ✅ LGBM загружен")

    # =========================================================
    # 4. Генерация мета-признаков для Test
    # =========================================================
    print(f"\n🔧 Генерация мета-признаков для Test...")

    # 🔥 ИСПРАВЛЕНО: Используем ARTIFACTS_DIR из универсальных путей
    meta_generator = MetaFeaturesGenerator(
        artifacts_dir=str(ARTIFACTS_DIR)
    )
    meta_generator.load_correlation_matrix()

    # Stage 1 инференс
    test_preds_stage1 = {col: np.zeros(len(customer_ids)) for col in target_cols}

    for i, model_manager in enumerate(stage1_models):
        preds = model_manager.predict(X_test_raw, cat_features=cat_features)
        for col in target_cols:
            if col in preds:
                test_preds_stage1[col] += preds[col]
        model_manager.clear()
        gc.collect()

    for col in target_cols:
        test_preds_stage1[col] /= len(stage1_models)

    df_test_preds = pd.DataFrame(test_preds_stage1)
    df_meta_test = meta_generator.generate_from_dataframe(df_test_preds, n_best_corr=15)
    print(f"   ✅ Мета-признаков: {df_meta_test.shape[1]}")

    # 🔥 Объединение признаков + Feature Selection
    X_test_extended = pd.concat([
        X_test_pd_raw.reset_index(drop=True),
        df_meta_test.reset_index(drop=True)
    ], axis=1)

    meta_cols = list(df_meta_test.columns)
    feature_cols = [c for c in selected_features if c in X_test_extended.columns and not c.startswith('meta_')]
    all_cols = feature_cols + meta_cols

    # Удаление дубликатов
    if len(all_cols) != len(set(all_cols)):
        seen = set()
        unique_cols = []
        for col in all_cols:
            if col not in seen:
                unique_cols.append(col)
                seen.add(col)
        all_cols = unique_cols

    X_test_extended = X_test_extended[all_cols]
    print(f"   ✅ После Feature Selection: {X_test_extended.shape[1]} признаков")

    # Конвертация в Polars
    X_test_pl = pl.from_pandas(X_test_extended)

    # =========================================================
    # 5. Предсказание на Test для pseudo-labeling
    # =========================================================
    print(f"\n🔮 Предсказание на Test для pseudo-labeling...")

    predictions_dict = {}

    for model_name, manager in models.items():
        print(f"   🔄 {model_name}...")
        preds = manager.predict(X_test_pl, cat_features=cat_features_filtered)

        if not predictions_dict:
            predictions_dict = preds
        else:
            for col in target_cols:
                predictions_dict[col] = (predictions_dict[col] + preds[col]) / 2

        manager.clear()
        gc.collect()

    print(f"   ✅ Предсказания готовы для {len(target_cols)} таргетов")

    # =========================================================
    # 6. Отбор уверенных предсказаний
    # =========================================================
    print(f"\n🎯 Отбор уверенных предсказаний...")

    pseudo_indices, pseudo_labels, select_stats = select_confident_predictions(
        predictions_dict=predictions_dict,
        target_cols=target_cols,
        threshold_high=PSEUDO_CONFIG['threshold_high'],
        threshold_low=PSEUDO_CONFIG['threshold_low'],
        max_samples=PSEUDO_CONFIG['max_pseudo_samples'],
        stratify=PSEUDO_CONFIG['stratify_by_class']
    )

    print(f"\n📊 Статистика отбора:")
    print(f"   Всего test сэмплов: {select_stats['total_test']:,}")
    print(f"   Отобрано pseudo-samples: {select_stats['selected_pseudo_samples']:,} "
          f"({select_stats['selection_ratio']:.2%})")

    if select_stats['selected_pseudo_samples'] < PSEUDO_CONFIG['min_pseudo_samples']:
        print(f"\n⚠️  WARNING: Слишком мало pseudo-samples!")
        return

    # =========================================================
    # 7. Создание комбинированного датасета
    # =========================================================
    print(f"\n🔗 Создание комбинированного датасета...")

    # 🔥 ИСПРАВЛЕНО: X_test_extended вместо X_test_pd_raw
    X_combined, y_combined, sample_weights, combine_stats = create_pseudo_train_dataset(
        X_train=X_train_pd,
        y_train=y_train_pd,
        X_test=X_test_extended,  # ← С мета-признаками!
        pseudo_indices=pseudo_indices,
        pseudo_labels=pseudo_labels,
        target_cols=target_cols,
        sample_weight=PSEUDO_CONFIG['pseudo_sample_weight']
    )

    print(f"\n📊 Комбинированный датасет:")
    print(f"   Train сэмплов: {combine_stats['n_train']:,}")
    print(f"   Pseudo сэмплов: {combine_stats['n_pseudo']:,}")
    print(f"   Всего: {combine_stats['n_combined']:,}")

    # =========================================================
    # 8. Генерация мета-признаков для pseudo-samples
    # =========================================================
    print(f"\n🔧 Генерация мета-признаков для pseudo-samples...")

    # 🔥 Перезагрузка Stage 1 моделей
    print(f"   🔄 Перезагрузка Stage 1 моделей...")
    stage1_models = []
    for model_name in stage1_model_names:
        manager = CatBoostManager(
            config_path=str(CONFIGS_DIR / "catboost" / "catboost_config.yaml"),
            fold_folder=f"folds_{n_splits_stage1}"
        )
        manager.load_model(model_name, fold_folder=f"folds_{n_splits_stage1}")
        stage1_models.append(manager)
        print(f"   ✅ Загружено: {model_name}")

    # 🔥 Генерируем мета-признаки для pseudo-samples
    X_pseudo_raw = X_test_pd_raw.iloc[pseudo_indices].reset_index(drop=True)

    df_meta_pseudo = generate_meta_features_for_pseudo(
        meta_generator=meta_generator,
        stage1_models=stage1_models,
        X_pseudo=X_pseudo_raw,
        cat_features=cat_features,
        target_cols=target_cols
    )

    # 🔥 Очищаем Stage 1 модели после использования
    for model_manager in stage1_models:
        model_manager.clear()
    gc.collect()

    # Объединяем с остальными данными
    meta_cols = [c for c in X_combined.columns if c.startswith('meta_')]
    df_meta_train = X_combined[meta_cols].iloc[:combine_stats['n_train']].reset_index(drop=True)

    df_meta_combined = pd.concat([
        df_meta_train,
        df_meta_pseudo.reset_index(drop=True)
    ], ignore_index=True)

    # Заменяем мета-признаки в X_combined
    X_combined = X_combined.drop(columns=meta_cols)
    X_combined = pd.concat([
        X_combined.reset_index(drop=True),
        df_meta_combined.reset_index(drop=True)
    ], axis=1)

    print(f"   ✅ Мета-признаков: {len(meta_cols)}")

    # 🔥 FEATURE SELECTION (СТРОГО 1500 признаков - как в Stage 2!)
    # 🔥 ИСПРАВЛЕНО: Используем ARTIFACTS_DIR из универсальных путей
    importance_path = ARTIFACTS_DIR / "feature_importance_stage2.csv"
    if importance_path.exists():
        imp_df = pd.read_csv(importance_path)
        selected_features = imp_df.head(1500)['feature'].tolist()  # ← СТРОГО 1500

        meta_cols = [c for c in X_combined.columns if c.startswith('meta_')]
        feature_cols = [c for c in selected_features if c in X_combined.columns and not c.startswith('meta_')]
        all_cols = list(set(feature_cols + meta_cols))

        # Проверка что все колонки на месте
        missing_cols = set(all_cols) - set(X_combined.columns)
        if missing_cols:
            print(f"   ⚠️  Отсутствуют {len(missing_cols)} колонок → добавляем нули")
            for col in missing_cols:
                X_combined[col] = 0.0

        X_combined = X_combined[all_cols]
        cat_features_filtered = [f for f in cat_features if f in all_cols]

        print(f"   ✅ После Feature Selection: {len(all_cols)} признаков (строго 1500)")

    # 🔥 ФИНАЛЬНАЯ ПРОВЕРКА НА NaN
    nan_count = X_combined.isna().sum().sum()
    if nan_count > 0:
        print(f"   ⚠️  Осталось {nan_count:,} NaN → заполняем медианой")
        X_combined = X_combined.fillna(X_combined.median())

    print(f"\n   🔍 Финальная проверка данных:")
    print(f"      Train строк: {combine_stats['n_train']:,}")
    print(f"      Pseudo строк: {combine_stats['n_pseudo']:,}")
    print(f"      Признаков: {X_combined.shape[1]}")
    print(f"      NaN: {X_combined.isna().sum().sum():,}")

    # =========================================================
    # 9. Дообучение моделей
    # =========================================================
    print(f"\n🚀 Дообучение моделей...")

    retrained_models = retrain_models_with_pseudo(
        X_combined=X_combined,
        y_combined=y_combined,
        sample_weights=sample_weights,
        cat_features=cat_features_filtered,
        models_to_retrain=PSEUDO_CONFIG['models_to_retrain']
    )

    # =========================================================
    # 10. Сохранение результатов
    # =========================================================
    print(f"\n💾 Сохранение результатов...")

    pseudo_stats = {
        'config': PSEUDO_CONFIG,
        'selection_stats': select_stats,
        'combine_stats': combine_stats,
        'models_retrained': list(retrained_models.keys()),
        'timestamp': pd.Timestamp.now().isoformat(),
        'project_root': str(PROJECT_ROOT)
    }

    # 🔥 ИСПРАВЛЕНО: Используем ARTIFACTS_DIR из универсальных путей
    stats_path = ARTIFACTS_DIR / "pseudo_labeling_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(pseudo_stats, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Статистика: {stats_path}")

    if PSEUDO_CONFIG['save_pseudo_labels']:
        pseudo_df = pd.DataFrame({
            'customer_id': customer_ids[pseudo_indices],
            **{col: pseudo_labels[col][pseudo_indices] for col in target_cols}
        })

        # 🔥 ИСПРАВЛЕНО: Используем ARTIFACTS_DIR из универсальных путей
        pseudo_path = ARTIFACTS_DIR / "pseudo_labels.parquet"
        pseudo_df.to_parquet(pseudo_path, index=False)
        print(f"   ✅ Pseudo-метки: {pseudo_path}")

    # =========================================================
    # 11. Финальный инференс
    # =========================================================
    print(f"\n🔮 Финальный инференс на Test...")

    final_predictions = {}

    for model_name, model_data in retrained_models.items():
        manager = model_data['manager']
        # 🔥 ИСПРАВЛЕНО: X_test_pl вместо X_test
        preds = manager.predict(X_test_pl, cat_features=cat_features_filtered)

        if not final_predictions:
            final_predictions = preds
        else:
            for col in target_cols:
                final_predictions[col] = (final_predictions[col] + preds[col]) / 2

        manager.clear()
        gc.collect()

    # Генерация сабмита
    submission = pd.DataFrame({'customer_id': customer_ids})

    for col, preds in final_predictions.items():
        submit_col = col.replace('target_', 'predict_')
        submission[submit_col] = preds

    nan_count = submission.isna().sum().sum()
    if nan_count > 0:
        print(f"   ⚠️  WARNING: {nan_count} NaN в сабмите!")

    # 🔥 ИСПРАВЛЕНО: Используем SUBMISSIONS_DIR из универсальных путей
    submission_path = SUBMISSIONS_DIR / "submission_pseudo.parquet"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(submission_path, index=False)

    print(f"   ✅ Сабмит сохранён: {submission_path}")

    # =========================================================
    # 12. Итоги
    # =========================================================
    print(f"\n{'=' * 70}")
    print(f"✅ PSEUDO-LABELING ЗАВЕРШЁН")
    print(f"{'=' * 70}")
    print(f"📊 Pseudo-samples: {combine_stats['n_pseudo']:,} ({combine_stats['pseudo_ratio']:.2%})")
    print(f"🌲 Дообучено моделей: {len(retrained_models)}")
    print(f"📁 Сабмит: {submission_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()