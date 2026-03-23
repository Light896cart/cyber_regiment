# =============================================================================
# scripts/04_inference_submission.py
# Инференс на тестовых данных с использованием конкретных моделей
# 🔥 ИЗМЕНЕНО: Поддержка per-target differential evolution весов
# 🔧 ИСПРАВЛЕНО: Универсальные пути для команды и GitHub
# 🔥 НОВОЕ: Автоматическая фиксация типов данных после сохранения
# =============================================================================

import sys
import gc
import os
import json
from pathlib import Path
from typing import Optional, Dict, List


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
DEFAULT_FOLDS_ROOT = PROJECT_ROOT / "folds"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models_weight"
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# 🔥 Можно переопределить через env variables
DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
FOLDS_ROOT = Path(os.getenv("FOLDS_ROOT", DEFAULT_FOLDS_ROOT))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
MODELS_DIR = Path(os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR))
CONFIGS_DIR = Path(os.getenv("CONFIGS_DIR", DEFAULT_CONFIGS_DIR))
SUBMISSIONS_DIR = Path(os.getenv("SUBMISSIONS_DIR", DEFAULT_SUBMISSIONS_DIR))

# Добавляем корень проекта в путь
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from models.catboost_model import CatBoostManager
from models.lgbm_model import LGBMManager
from models.nn_model import NNManager
from utils.meta_features import MetaFeaturesGenerator

import polars as pl
import pandas as pd
import numpy as np


# =============================================================================
# 🔧 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def load_stage1_models(
        n_splits: int = 3,
        model_names: Optional[List[str]] = None
) -> List[CatBoostManager]:
    """
    Загружает Stage 1 модели с проверкой существования.
    """
    if model_names is None:
        model_names = [
            "catboost_fold_0_stage1",
            "catboost_fold_1_stage1",
            "catboost_fold_2_stage1"
        ]

    stage1_models = []
    fold_folder = f"folds_{n_splits}"

    for model_name in model_names:
        model_path = MODELS_DIR / fold_folder / model_name / "model.cbm"

        if not model_path.exists():
            raise FileNotFoundError(
                f"❌ Stage 1 модель не найдена: {model_path}\n"
                f"💡 Запусти сначала: python scripts/02_stage1_proxy_training.py"
            )

        manager = CatBoostManager(
            config_path=str(CONFIGS_DIR / "catboost" / "catboost_config.yaml"),
            fold_folder=fold_folder
        )
        manager.load_model(model_name, fold_folder=fold_folder)
        stage1_models.append(manager)
        print(f"   ✅ Загружено: {model_name}")

    return stage1_models


def load_stage2_models() -> Dict[str, object]:
    """
    Загружает Stage 2 модели с проверкой существования.
    """
    models = {}

    # CatBoost
    cb_name = "stage2_catboost_validation_v1"
    cb_path = MODELS_DIR / "catboost" / cb_name / "model.cbm"
    if not cb_path.exists():
        raise FileNotFoundError(
            f"❌ CatBoost Stage 2 не найдена: {cb_path}\n"
            f"💡 Запусти сначала: python scripts/03_train_stage2_validate.py"
        )

    stage2_cb = CatBoostManager(
        config_path=str(CONFIGS_DIR / "catboost" / "catboost_config_stage2.yaml"),
        fold_folder="catboost"
    )
    stage2_cb.load_model(cb_name, fold_folder="catboost")
    models['catboost'] = {'manager': stage2_cb, 'name': cb_name}
    print(f"   ✅ Загружено: CatBoost ({cb_name})")

    # Neural Network
    nn_name = "stage2_nn_validation_v1"
    nn_path = MODELS_DIR / "neural_network" / nn_name / "model.pth"
    if not nn_path.exists():
        raise FileNotFoundError(
            f"❌ NN Stage 2 не найдена: {nn_path}\n"
            f"💡 Запусти сначала: python scripts/03_train_stage2_validate.py"
        )

    stage2_nn = NNManager(
        config_path=str(CONFIGS_DIR / "neural_network" / "nn_config_stage2.yaml"),
        fold_folder="neural_network"
    )
    stage2_nn.load_model(nn_name, fold_folder="neural_network")
    models['neural_network'] = {'manager': stage2_nn, 'name': nn_name}
    print(f"   ✅ Загружено: NN ({nn_name})")

    # LightGBM
    lgbm_name = "stage2_lgbm_validation_v1"
    lgbm_path = MODELS_DIR / "lightgbm" / lgbm_name / "model_0.txt"
    if not lgbm_path.exists():
        raise FileNotFoundError(
            f"❌ LGBM Stage 2 не найдена: {lgbm_path}\n"
            f"💡 Запусти сначала: python scripts/03_train_stage2_validate.py"
        )

    stage2_lgbm = LGBMManager(
        config_path=str(CONFIGS_DIR / "lightgbm" / "lgbm_config_stage2.yaml"),
        fold_folder="lightgbm"
    )
    stage2_lgbm.load_model(lgbm_name, fold_folder="lightgbm")
    models['lightgbm'] = {'manager': stage2_lgbm, 'name': lgbm_name}
    print(f"   ✅ Загружено: LGBM ({lgbm_name})")

    return models


def load_ensemble_config() -> tuple:
    """
    Загружает конфигурацию ансамбля из mapping файла.
    Returns:
        best_method, target_weights, best_model_map, ensemble_weights
    """
    mapping_path = ARTIFACTS_DIR / "best_model_per_target_map.json"

    best_method = 'weighted_average'
    target_weights = {}
    best_model_map = {}
    ensemble_weights = {'catboost': 0.33, 'neural_network': 0.33, 'lightgbm': 0.34}

    if mapping_path.exists():
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        best_method = mapping_data.get('method', 'weighted_average')
        best_model_map = mapping_data.get('best_model_map', {})
        target_weights = mapping_data.get('target_weights', {})

        if 'comparison' in mapping_data and 'ensemble_weights' in mapping_data['comparison']:
            ensemble_weights = mapping_data['comparison']['ensemble_weights']

        print(f"   ✅ Загружена конфигурация ансамбля: {best_method}")

        if best_method == 'per_target_differential_evolution':
            print(f"   📊 Per-Target весов загружено: {len(target_weights)}")
        elif best_method == 'best_per_target':
            model_counts = {}
            for m in best_model_map.values():
                model_counts[m] = model_counts.get(m, 0) + 1
            print(f"   📊 Распределение моделей по таргетам:")
            for m, c in model_counts.items():
                print(f"      {m}: {c} таргетов")
    else:
        print(f"   ⚠️  {mapping_path} не найден, используем weighted_average")

    return best_method, target_weights, best_model_map, ensemble_weights


def blend_predictions(
        model_predictions: Dict[str, Dict[str, np.ndarray]],
        target_cols: List[str],
        best_method: str,
        target_weights: Dict,
        best_model_map: Dict,
        ensemble_weights: Dict
) -> Dict[str, np.ndarray]:
    """
    Применяет логику ансамблирования (3 варианта).
    """
    final_preds = {}

    if best_method == 'per_target_differential_evolution' and target_weights:
        # 🔥 ВАРИАНТ 1: Per-Target Differential Evolution (123 веса)
        print(f"\n   🏆 Используем PER-TARGET DIFFERENTIAL EVOLUTION...")

        for col in target_cols:
            if col in target_weights:
                weights = target_weights[col]
                blended = 0.0
                for model_name, weight in weights.items():
                    if model_name in model_predictions and col in model_predictions[model_name]:
                        blended += weight * model_predictions[model_name][col]
                final_preds[col] = blended
            else:
                # Fallback на равные веса
                cb_pred = model_predictions.get('catboost', {}).get(col, 0)
                nn_pred = model_predictions.get('neural_network', {}).get(col, 0)
                lgbm_pred = model_predictions.get('lightgbm', {}).get(col, 0)
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
                    final_preds[col] = model_predictions.get('catboost', {}).get(col, np.zeros(len(customer_ids)))
            else:
                final_preds[col] = model_predictions.get('catboost', {}).get(col, np.zeros(len(customer_ids)))

        print(f"   ✅ Best Model Per Target применён: {len(final_preds)} таргетов")

    else:
        # 🔥 ВАРИАНТ 3: Weighted Average (3 глобальных веса)
        print(f"\n   ⚖️  Используем Weighted Average...")
        for col in target_cols:
            cb_pred = model_predictions.get('catboost', {}).get(col, 0)
            nn_pred = model_predictions.get('neural_network', {}).get(col, 0)
            lgbm_pred = model_predictions.get('lightgbm', {}).get(col, 0)

            final_preds[col] = (
                    ensemble_weights.get('catboost', 0.33) * cb_pred +
                    ensemble_weights.get('neural_network', 0.33) * nn_pred +
                    ensemble_weights.get('lightgbm', 0.34) * lgbm_pred
            )

        print(f"   ✅ Взвешено: {len(final_preds)} таргетов")

    return final_preds


# =============================================================================
# 🔥 НОВОЕ: Функция фиксации типов данных
# =============================================================================

def fix_dtypes_and_save(input_file: Path, output_file: Optional[Path] = None) -> Path:
    """
    🔥 Фиксирует типы данных в файле сабмита для совместимости с платформой.

    Args:
        input_file: Путь к исходному файлу
        output_file: Путь к выходному файлу (если None, добавляется _fixed_types)

    Returns:
        Path к сохранённому файлу
    """
    if not input_file.exists():
        print(f"❌ Файл {input_file} не найден!")
        return input_file

    if output_file is None:
        base, ext = os.path.splitext(str(input_file))
        output_file = Path(f"{base}_fixed_types{ext}")

    try:
        print(f"\n🔄 Фиксация типов данных: {input_file} ...")
        df = pd.read_parquet(input_file)

        # 1. Конвертируем customer_id в int32
        print("🔧 Конвертация customer_id в int32...")
        if df['customer_id'].max() > 2147483647 or df['customer_id'].min() < -2147483648:
            print("⚠️ Ошибка: значения customer_id не влезают в int32!")
            return input_file
        df['customer_id'] = df['customer_id'].astype('int32')

        # 2. Находим все колонки predict_* и приводим их к float64
        print("🔧 Приведение всех predict_... колонок к float64...")
        predict_cols = [col for col in df.columns if col.startswith('predict_')]
        print(f"   Найдено колонок для конвертации: {len(predict_cols)}")

        for col in predict_cols:
            original_dtype = df[col].dtype
            df[col] = df[col].astype('float64')
            if original_dtype != 'float64':
                print(f"   • {col}: {original_dtype} → float64")

        # 3. Сохраняем
        print(f"💾 Сохранение в: {output_file} ...")
        df.to_parquet(output_file, index=False)

        # 4. Проверка результата
        print("\n✅ Готово! Проверка сохранённого файла:")
        df_check = pd.read_parquet(output_file)
        print(f"🔢 customer_id dtype: {df_check['customer_id'].dtype}")
        if predict_cols:
            print(f"🔢 predict_... dtype (пример): {df_check[predict_cols[0]].dtype}")

        # Сравнение размеров
        original_size = os.path.getsize(input_file) / (1024 * 1024)
        new_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n📊 Размер оригинала: {original_size:.2f} MB")
        print(f"📊 Размер нового файла: {new_size:.2f} MB")

        print(f"\n✅ Файл с исправленными типами: {output_file}")

        return output_file

    except Exception as e:
        print(f"❌ Ошибка при фиксации типов: {e}")
        import traceback
        traceback.print_exc()
        return input_file


# =============================================================================
# 🔧 ОСНОВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    # =========================================================
    # 1. Инициализация
    # =========================================================
    print(f"\n{'=' * 60}")
    print(f"🚀 INFERENCE: ГЕНЕРАЦИЯ САБМИТА")
    print(f"{'=' * 60}")
    print(f"   📁 Проект: {PROJECT_ROOT}")
    print(f"   📁 Data: {DATA_DIR}")
    print(f"   📁 Models: {MODELS_DIR}")
    print(f"   📁 Artifacts: {ARTIFACTS_DIR}")
    print(f"{'=' * 60}")

    loader = DataLoader(cat_strategy="int")
    meta_generator = MetaFeaturesGenerator(
        artifacts_dir=str(ARTIFACTS_DIR)
    )

    # Загружаем корреляционную матрицу
    meta_generator.load_correlation_matrix()

    # 🔥 Загружаем cat_features из метаданных фолдов
    # 🔥 ИСПРАВЛЕНО: Используем FOLDS_ROOT из универсальных путей
    n_splits_stage1 = 3
    folds_meta_path = FOLDS_ROOT / f"folds_{n_splits_stage1}" / "fold_metadata.json"

    if not folds_meta_path.exists():
        raise FileNotFoundError(
            f"❌ Метаданные фолдов не найдены: {folds_meta_path}\n"
            f"💡 Запусти сначала: python scripts/01_generate_folds.py --n-splits {n_splits_stage1}"
        )

    with open(folds_meta_path, 'r') as f:
        folds_meta = json.load(f)

    cat_features = folds_meta.get('cat_features', [])

    if not cat_features:
        # Fallback: пытаемся получить из загруженных данных
        try:
            loader.load_full_data()
            cat_features = [col for col in loader._df_full.columns if col.startswith('cat_')]
        except:
            cat_features = []

    print(f"   ✅ cat_features: {len(cat_features)}")

    # =========================================================
    # 2. Загрузка Stage 1 моделей
    # =========================================================
    print(f"\n📂 Загрузка Stage 1 моделей...")

    stage1_models = load_stage1_models(n_splits=n_splits_stage1)

    if len(stage1_models) == 0:
        raise FileNotFoundError("Нет Stage 1 моделей!")

    print(f"   📊 Stage 1 моделей: {len(stage1_models)}")

    # =========================================================
    # 3. Загрузка тестовых данных
    # =========================================================
    print(f"\n📂 Загрузка тестовых данных...")

    # 🔥 ИСПРАВЛЕНО: Используем DATA_DIR из универсальных путей
    test_path = DATA_DIR / "test_final.parquet"

    if not test_path.exists():
        raise FileNotFoundError(
            f"❌ Тестовые данные не найдены: {test_path}\n"
            f"💡 Убедитесь что test_final.parquet существует в data/"
        )

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

    stage2_models = load_stage2_models()

    # =========================================================
    # 7. Объединение признаков
    # =========================================================
    print(f"\n🔗 Объединение признаков...")

    X_test_pd = X_test.to_pandas()
    X_test_extended = pd.concat([
        X_test_pd.reset_index(drop=True),
        df_meta_test.reset_index(drop=True)
    ], axis=1)

    # 🔥 ИСПРАВЛЕНО: Используем ARTIFACTS_DIR из универсальных путей
    importance_path = ARTIFACTS_DIR / "feature_importance_stage2.csv"

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
    else:
        print(f"   ⚠️  {importance_path} не найден, используем все признаки")

    print(f"   📊 Test extended: {X_test_extended.shape[1]} признаков")

    # =========================================================
    # 8. Stage 2 инференс
    # =========================================================
    print(f"\n🔮 Stage 2 инференс...")

    X_test_extended_pl = pl.from_pandas(X_test_extended)

    final_preds_cb = stage2_models['catboost']['manager'].predict(
        X_test_extended_pl,
        cat_features=cat_features
    )
    print(f"   ✅ CatBoost: {len(final_preds_cb)} таргетов")

    final_preds_nn = stage2_models['neural_network']['manager'].predict(
        X_test_extended_pl,
        cat_features=cat_features
    )
    print(f"   ✅ NN: {len(final_preds_nn)} таргетов")

    final_preds_lgbm = stage2_models['lightgbm']['manager'].predict(
        X_test_extended_pl,
        cat_features=cat_features
    )
    print(f"   ✅ LGBM: {len(final_preds_lgbm)} таргетов")

    # =========================================================
    # 9. 🔥 Ансамблирование
    # =========================================================
    print(f"\n🔗 Ансамблирование предсказаний...")

    best_method, target_weights, best_model_map, ensemble_weights = load_ensemble_config()

    # Словарь предсказаний для удобного доступа
    model_predictions = {
        'catboost': final_preds_cb,
        'neural_network': final_preds_nn,
        'lightgbm': final_preds_lgbm
    }

    # 🔥 Применяем логику ансамблирования
    final_preds = blend_predictions(
        model_predictions=model_predictions,
        target_cols=target_cols,
        best_method=best_method,
        target_weights=target_weights,
        best_model_map=best_model_map,
        ensemble_weights=ensemble_weights
    )

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
    # 🔥 ИСПРАВЛЕНО: Используем SUBMISSIONS_DIR из универсальных путей
    submission_path = SUBMISSIONS_DIR / "submission_ensemble.parquet"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(submission_path, index=False)

    print(f"   💾 Сохранено: {submission_path}")

    # =========================================================
    # 🔥 11. Фиксация типов данных (НОВОЕ!)
    # =========================================================
    print(f"\n{'=' * 60}")
    print(f"🔧 ФИКСАЦИЯ ТИПОВ ДАННЫХ")
    print(f"{'=' * 60}")

    fixed_submission_path = fix_dtypes_and_save(
        input_file=submission_path,
        output_file=SUBMISSIONS_DIR / "submission_ensemble_fixed.parquet"
    )

    # =========================================================
    # 12. Сохранение метаданных
    # =========================================================
    inference_meta = {
        'project_root': str(PROJECT_ROOT),
        'stage1_models': [
            "catboost_fold_0_stage1",
            "catboost_fold_1_stage1",
            "catboost_fold_2_stage1"
        ],
        'stage2_models': {
            'catboost': stage2_models['catboost']['name'],
            'neural_network': stage2_models['neural_network']['name'],
            'lightgbm': stage2_models['lightgbm']['name']
        },
        'ensemble_weights': ensemble_weights,
        'target_weights': target_weights if best_method == 'per_target_differential_evolution' else {},
        'ensemble_method': best_method,
        'best_model_map': best_model_map if best_method == 'best_per_target' else {},
        'n_test_samples': int(len(customer_ids)),
        'n_features': int(X_test_extended.shape[1]),
        'n_targets': len(target_cols),
        'submission_path': str(submission_path),
        'fixed_submission_path': str(fixed_submission_path),
        'timestamp': pd.Timestamp.now().isoformat()
    }

    # 🔥 ИСПРАВЛЕНО: Используем SUBMISSIONS_DIR из универсальных путей
    with open(SUBMISSIONS_DIR / "inference_meta.json", 'w') as f:
        json.dump(inference_meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✅ САБМИТ СГЕНЕРИРОВАН")
    print(f"   📁 Оригинальный: {submission_path}")
    print(f"   📁 С фикс. типов: {fixed_submission_path}")
    print(f"   📊 Метод ансамбля: {best_method}")
    print(f"{'=' * 60}")
    print(f"\n💡 РЕКОМЕНДАЦИЯ: Используй файл с _fixed_types для сабмита!")

if __name__ == "__main__":
    main()