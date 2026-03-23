# =============================================================================
# scripts/04_inference_submission.py
# Инференс на тестовых данных с использованием конкретных моделей
# 🔥 ИЗМЕНЕНО: Поддержка per-target differential evolution весов
# =============================================================================

import sys
import gc
import os
import json
from pathlib import Path

from src.data.loader import DataLoader

ROOT_DIR = Path(r"D:\Code\hackaton_cyberpolka_CV")
sys.path.append(str(ROOT_DIR))

from models.catboost_model import CatBoostManager
from models.lgbm_model import LGBMManager
from models.nn_model import NNManager
from utils.meta_features import MetaFeaturesGenerator

import polars as pl
import pandas as pd
import numpy as np


def main():
    # =========================================================
    # 1. Инициализация
    # =========================================================
    print(f"\n{'=' * 60}")
    print(f"🚀 INFERENCE: ГЕНЕРАЦИЯ САБМИТА")
    print(f"{'=' * 60}")

    loader = DataLoader(cat_strategy="int")
    meta_generator = MetaFeaturesGenerator(
        artifacts_dir=str(ROOT_DIR / "artifacts")
    )

    # Загружаем корреляционную матрицу
    meta_generator.load_correlation_matrix()

    # 🔥 Загружаем cat_features из метаданных фолдов
    folds_meta_path = ROOT_DIR / "folds" / "folds_3" / "fold_metadata.json"
    with open(folds_meta_path, 'r') as f:
        folds_meta = json.load(f)
    cat_features = folds_meta.get('cat_features', [])

    if not cat_features:
        cat_features = [col for col in loader._df_full.columns if
                        col.startswith('cat_')] if loader._df_full is not None else []

    print(f"   ✅ cat_features: {len(cat_features)}")

    # =========================================================
    # 2. Загрузка Stage 1 моделей
    # =========================================================
    print(f"\n📂 Загрузка Stage 1 моделей...")

    n_splits_stage1 = 3
    stage1_model_names = [
        "catboost_fold_0_stage1",
        "catboost_fold_1_stage1",
        'catboost_fold_2_stage1'
    ]

    stage1_models = []
    for model_name in stage1_model_names:
        model_path = ROOT_DIR / "models_weight" / f"folds_{n_splits_stage1}" / model_name / "model.cbm"
        if model_path.exists():
            manager = CatBoostManager(
                config_path=str(ROOT_DIR / "configs" / "catboost" / "catboost_config.yaml"),
                fold_folder=f"folds_{n_splits_stage1}"
            )
            manager.load_model(model_name, fold_folder=f"folds_{n_splits_stage1}")
            stage1_models.append(manager)
            print(f"   ✅ Загружено: {model_name}")
        else:
            print(f"   ❌ Не найдено: {model_name}")
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

    if len(stage1_models) == 0:
        raise FileNotFoundError("Нет Stage 1 моделей!")

    print(f"   📊 Stage 1 моделей: {len(stage1_models)}")

    # =========================================================
    # 3. Загрузка тестовых данных
    # =========================================================
    print(f"\n📂 Загрузка тестовых данных...")

    test_path = ROOT_DIR / "data" / "test_final.parquet"

    if not test_path.exists():
        raise FileNotFoundError(f"Тестовые данные не найдены: {test_path}")

    df_test = pl.read_parquet(test_path)
    customer_ids = df_test["customer_id"].to_numpy()
    df_test = loader._preprocess_categorical(df_test, cat_features)
    X_test = df_test.select([col for col in df_test.columns if col != 'customer_id'])

    print(f"   📊 Тест: {X_test.shape[0]} строк, {X_test.shape[1]} признаков")

    # =========================================================
    # 4. Stage 1 инференс
    # =========================================================
    print(f"\n🔮 Stage 1 инференс на тесте...")

    target_cols = meta_generator.target_cols
    test_preds_stage1 = {col: np.zeros(len(customer_ids)) for col in target_cols}

    for i, model_manager in enumerate(stage1_models):
        print(f"   🔄 Модель {i + 1}/{len(stage1_models)}...")
        preds = model_manager.predict(X_test, cat_features=cat_features)

        for col in target_cols:
            if col in preds:
                test_preds_stage1[col] += preds[col]

        model_manager.clear()
        gc.collect()

    for col in target_cols:
        test_preds_stage1[col] /= len(stage1_models)

    print(f"   ✅ Stage 1 предсказания усреднены")

    # =========================================================
    # 5. Генерация мета-признаков для теста
    # =========================================================
    print(f"\n🔧 Генерация мета-признаков для теста...")

    df_test_preds = pd.DataFrame(test_preds_stage1)
    df_meta_test = meta_generator.generate_from_dataframe(df_test_preds, n_best_corr=15)

    print(f"   📊 Мета-признаки: {df_meta_test.shape}")

    # =========================================================
    # 6. Загрузка Stage 2 моделей
    # =========================================================
    print(f"\n📂 Загрузка Stage 2 моделей...")

    # CatBoost
    stage2_cb_name = "stage2_catboost_validation_v1"
    stage2_cb = CatBoostManager(
        config_path=str(ROOT_DIR / "configs" / "catboost" / "catboost_config_stage2.yaml"),
        fold_folder="catboost"
    )
    stage2_cb.load_model(stage2_cb_name, fold_folder="catboost")
    print(f"   ✅ Загружено: CatBoost ({stage2_cb_name})")

    # Neural Network
    stage2_nn_name = "stage2_nn_validation_v1"
    stage2_nn = NNManager(
        config_path=str(ROOT_DIR / "configs" / "neural_network" / "nn_config_stage2.yaml"),
        fold_folder="neural_network"
    )
    stage2_nn.load_model(stage2_nn_name, fold_folder="neural_network")
    print(f"   ✅ Загружено: NN ({stage2_nn_name})")

    # LightGBM
    stage2_lgbm_name = "stage2_lgbm_validation_v1"
    stage2_lgbm = LGBMManager(
        config_path=str(ROOT_DIR / "configs" / "lightgbm" / "lgbm_config_stage2.yaml"),
        fold_folder="lightgbm"
    )
    stage2_lgbm.load_model(stage2_lgbm_name, fold_folder="lightgbm")
    print(f"   ✅ Загружено: LGBM ({stage2_lgbm_name})")

    # =========================================================
    # 7. Объединение признаков
    # =========================================================
    print(f"\n🔗 Объединение признаков...")

    X_test_pd = X_test.to_pandas()
    X_test_extended = pd.concat([
        X_test_pd.reset_index(drop=True),
        df_meta_test.reset_index(drop=True)
    ], axis=1)

    importance_path = ROOT_DIR / "artifacts" / "feature_importance_stage2.csv"
    if importance_path.exists():
        selected_features = pd.read_csv(importance_path)['feature'].tolist()
        meta_cols = list(df_meta_test.columns)

        feature_cols = [c for c in selected_features if c in X_test_extended.columns and c not in meta_cols]
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
        print(f"   ✅ Feature Selection применён: {len(feature_cols)} признаков + {len(meta_cols)} мета-признаков")

    print(f"   📊 Test extended: {X_test_extended.shape[1]} признаков")

    # =========================================================
    # 8. Stage 2 инференс
    # =========================================================
    print(f"\n🔮 Stage 2 инференс...")

    X_test_extended_pl = pl.from_pandas(X_test_extended)

    final_preds_cb = stage2_cb.predict(X_test_extended_pl, cat_features=cat_features)
    print(f"   ✅ CatBoost: {len(final_preds_cb)} таргетов")

    final_preds_nn = stage2_nn.predict(X_test_extended_pl, cat_features=cat_features)
    print(f"   ✅ NN: {len(final_preds_nn)} таргетов")

    final_preds_lgbm = stage2_lgbm.predict(X_test_extended_pl, cat_features=cat_features)
    print(f"   ✅ LGBM: {len(final_preds_lgbm)} таргетов")

    # =========================================================
    # 9. 🔥 Ансамблирование (🔥 НОВОЕ: Per-Target Differential Evolution)
    # =========================================================
    print(f"\n🔗 Ансамблирование предсказаний...")

    mapping_path = ROOT_DIR / "artifacts" / "best_model_per_target_map.json"

    best_method = 'weighted_average'
    best_model_map = {}
    ensemble_weights = {'catboost': 0.33, 'neural_network': 0.33, 'lightgbm': 0.34}
    target_weights = {}  # 🔥 НОВОЕ: для per-target весов

    if mapping_path.exists():
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        # Извлекаем данные из правильной структуры JSON
        best_method = mapping_data.get('method', 'weighted_average')
        best_model_map = mapping_data.get('best_model_map', {})

        # 🔥 НОВОЕ: Загружаем per-target веса (для differential_evolution)
        target_weights = mapping_data.get('target_weights', {})

        # Веса могут лежать внутри comparison
        if 'comparison' in mapping_data and 'ensemble_weights' in mapping_data['comparison']:
            ensemble_weights = mapping_data['comparison']['ensemble_weights']

        print(f"   ✅ Загружена конфигурация ансамбля: {best_method}")

        if best_method == 'per_target_differential_evolution':
            print(f"   📊 Per-Target весов загружено: {len(target_weights)}")
            print(f"   📊 Пример весов для target_1_1: {target_weights.get('target_1_1', {})}")
        elif best_method == 'best_per_target':
            print(f"   📊 Распределение моделей по таргетам:")
            model_counts = {}
            for m in best_model_map.values():
                model_counts[m] = model_counts.get(m, 0) + 1
            for m, c in model_counts.items():
                print(f"      {m}: {c} таргетов")
    else:
        print(f"   ⚠️  {mapping_path} не найден, используем weighted_average")

    # Словарь предсказаний для удобного доступа
    model_predictions = {
        'catboost': final_preds_cb,
        'neural_network': final_preds_nn,
        'lightgbm': final_preds_lgbm
    }

    final_preds = {}

    # =============================================================================
    # 🔥 ЛОГИКА АНСАМБЛИРОВАНИЯ (3 варианта)
    # =============================================================================

    if best_method == 'per_target_differential_evolution' and target_weights:
        # 🔥 ВАРИАНТ 1: Per-Target Differential Evolution (123 веса)
        print(f"\n   🏆 Используем PER-TARGET DIFFERENTIAL EVOLUTION...")

        for col in target_cols:
            if col in target_weights:
                weights = target_weights[col]  # 🔥 Отдельные веса для этого таргета
                blended = 0.0
                for model_name, weight in weights.items():
                    if model_name in model_predictions and col in model_predictions[model_name]:
                        blended += weight * model_predictions[model_name][col]
                final_preds[col] = blended
            else:
                # Fallback на равные веса если нет весов для таргета
                cb_pred = final_preds_cb.get(col, 0)
                nn_pred = final_preds_nn.get(col, 0)
                lgbm_pred = final_preds_lgbm.get(col, 0)
                final_preds[col] = 0.33 * cb_pred + 0.33 * nn_pred + 0.34 * lgbm_pred

        print(f"   ✅ Per-Target веса применены: {len(final_preds)} таргетов")

    elif best_method == 'best_per_target' and best_model_map:
        # 🔥 ВАРИАНТ 2: Best Model Per Target (1 модель на таргет)
        print(f"\n   🏆 Используем Best Model Per Target...")

        for col in target_cols:
            if col in best_model_map:
                best_model = best_model_map[col]
                if best_model in model_predictions and col in model_predictions[best_model]:
                    final_preds[col] = model_predictions[best_model][col]
                else:
                    print(f"   ⚠️  Фоллбэк для {col}: {best_model} не найдена")
                    final_preds[col] = final_preds_cb.get(col, np.zeros(len(customer_ids)))
            else:
                final_preds[col] = final_preds_cb.get(col, np.zeros(len(customer_ids)))

        print(f"   ✅ Best Model Per Target применён: {len(final_preds)} таргетов")

    else:
        # 🔥 ВАРИАНТ 3: Weighted Average (3 глобальных веса)
        print(f"\n   ⚖️  Используем Weighted Average...")
        for col in target_cols:
            cb_pred = final_preds_cb.get(col, 0)
            nn_pred = final_preds_nn.get(col, 0)
            lgbm_pred = final_preds_lgbm.get(col, 0)

            final_preds[col] = (
                    ensemble_weights.get('catboost', 0.33) * cb_pred +
                    ensemble_weights.get('neural_network', 0.33) * nn_pred +
                    ensemble_weights.get('lightgbm', 0.34) * lgbm_pred
            )

        print(f"   ✅ Взвешено: {len(final_preds)} таргетов")

    # =========================================================
    # 10. Генерация сабмита
    # =========================================================
    print(f"\n📝 Генерация сабмита...")

    submission = pd.DataFrame({'customer_id': customer_ids})

    for col, preds in final_preds.items():
        submit_col = col.replace('target_', 'predict_')
        submission[submit_col] = preds

    # Проверка на NaN
    nan_count = submission.isna().sum().sum()
    if nan_count > 0:
        print(f"   ⚠️  ВНИМАНИЕ: {nan_count} NaN в сабмите!")

    # Проверка диапазона
    out_of_range = 0
    for col in submission.columns:
        if col != 'customer_id':
            min_val = submission[col].min()
            max_val = submission[col].max()
            if min_val < 0 or max_val > 1:
                out_of_range += 1

    if out_of_range > 0:
        print(f"   ⚠️  Колонок вне диапазона [0,1]: {out_of_range}")

    # Сохранение
    submission_path = ROOT_DIR / "submissions" / "submission_ensemble.parquet"
    submission_path.parent.mkdir(exist_ok=True)
    submission.to_parquet(submission_path, index=False)

    print(f"   💾 Сохранено: {submission_path}")

    # =========================================================
    # 11. Сохранение метаданных
    # =========================================================
    inference_meta = {
        'stage1_models': stage1_model_names,
        'stage2_models': {
            'catboost': stage2_cb_name,
            'neural_network': stage2_nn_name,
            'lightgbm': stage2_lgbm_name
        },
        'ensemble_weights': ensemble_weights,
        'target_weights': target_weights if best_method == 'per_target_differential_evolution' else {},
        'ensemble_method': best_method,
        'best_model_map': best_model_map if best_method == 'best_per_target' else {},
        'n_test_samples': len(customer_ids),
        'n_features': X_test_extended.shape[1],
        'n_targets': len(target_cols),
        'submission_path': str(submission_path),
        'timestamp': pd.Timestamp.now().isoformat()
    }

    with open(ROOT_DIR / "submissions" / "inference_meta.json", 'w') as f:
        json.dump(inference_meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✅ САБМИТ СГЕНЕРИРОВАН")
    print(f"   📁 {submission_path}")
    print(f"   📊 Метод ансамбля: {best_method}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()