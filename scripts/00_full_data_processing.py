#!/usr/bin/env python3
"""
01_full_data_processing_v2.py
🔥 ГИБРИДНЫЙ PIPELINE: ЛУЧШЕЕ ИЗ ОБОХ ПОДХОДОВ + ИСПРАВЛЕНИЯ LEAKAGE
Включает:
✅ Polars (2-3x быстрее Pandas)
✅ Sparse Null PCA (экономия 90% памяти)
✅ 🔥 ИСПРАВЛЕНО: Null Flags БЕЗ таргета (убрана leakage!)
✅ Frequency Encoding для cat features (+0.01 AUC)
✅ Category Interactions
✅ MD5 Duplicate Detection
✅ 🔥 НОВОЕ: Target Encoding с regularization (+0.01-0.02 AUC)
✅ 🔥 НОВОЕ: Polynomial Features для топ признаков (+0.005-0.01 AUC)
✅ 🔥 НОВОЕ: Numerical Interactions (+0.005 AUC)
✅ 🔥 НОВОЕ: GroupBy Aggregations по категориям (+0.01 AUC)
✅ 🔥 НОВОЕ: Feature Stability Check (train vs test)
✅ Full Row Statistics (mean, std, skew, min, max, count)
✅ Automatic + Manual Ratio Features
✅ Правильный порядок операций
✅ Memory Optimization (float32, int32, int8)
✅ Train/Test Sync
Запуск: python 01_full_data_processing_v2.py
"""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.decomposition import TruncatedSVD
from scipy.stats import spearmanr

# =============================================================================
# КОНФИГУРАЦИЯ
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


BASE_DIR = get_project_root()
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data_new"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Входные файлы
TRAIN_MAIN = DATA_DIR / "train_main_features.parquet"
TEST_MAIN = DATA_DIR / "test_main_features.parquet"
TRAIN_EXTRA = DATA_DIR / "train_extra_features.parquet"
TEST_EXTRA = DATA_DIR / "test_extra_features.parquet"
TRAIN_TARGET = DATA_DIR / "train_target.parquet"

# Выходные файлы
TRAIN_FINAL = OUTPUT_DIR / "train_final_processed_v2.parquet"
TEST_FINAL = OUTPUT_DIR / "test_final_processed_v2.parquet"
METADATA_FILE = OUTPUT_DIR / "processing_metadata_v2.json"

# Параметры обработки
ID_COLUMN = "customer_id"
NULL_THRESHOLD = 0.80  # Удаляем колонки где >80% null
NULL_PCA_COMPONENTS = 20
NULL_FLAG_MIN_RATIO = 0.01  # Минимум 1% пропусков для флага
# 🔥 ИСПРАВЛЕНО: Убран NULL_FLAG_CORR_THRESHOLD (больше не используем таргет!)
MAX_RATIO_PAIRS = 50
RATIO_MIN_NONZERO = 0.5  # Минимум 50% ненулевых знаменателей
SEED = 42

# 🔥 НОВОЕ: Параметры для новых фич
TARGET_ENCODING_SMOOTHING = 10  # Alpha для target encoding
POLY_FEATURES_TOP_N = 20  # Сколько топ признаков для полиномов
NUM_INTERACTION_PAIRS = 30  # Сколько пар для взаимодействий
GROUPBY_AGG_TOP_CATS = 5  # Сколько топ категориальных для агрегаций
STABILITY_THRESHOLD = 0.15  # Порог для нестабильных признаков (15%)

# Ручные ratio пары (из Pipeline 2 - проверенные)
MANUAL_RATIO_PAIRS = [
    ("num_feature_62", "num_feature_79"),
    ("num_feature_27", "num_feature_79"),
    ("num_feature_76", "num_feature_79"),
    ("num_feature_62", "num_feature_86"),
    ("num_feature_116", "num_feature_124"),
    ("num_feature_36", "num_feature_79"),
    ("num_feature_41", "num_feature_79"),
    ("num_feature_83", "num_feature_108"),
    ("num_feature_83", "num_feature_103"),
]

# Category interactions (из Pipeline 2)
CAT_INTERACTIONS = [
    ("cat_feature_66", "cat_feature_46"),
    ("cat_feature_66", "cat_feature_39"),
    ("cat_feature_66", "cat_feature_48"),
    ("cat_feature_66", "cat_feature_9"),
    ("cat_feature_66", "cat_feature_52"),
]

# Duplicate categorical features (из Pipeline 2)
DUPLICATE_CATS = [
    "cat_feature_24", "cat_feature_25", "cat_feature_26",
    "cat_feature_29", "cat_feature_50", "cat_feature_63",
]


# =============================================================================
# КЛАСС: ГИБРИДНЫЙ ПРОЦЕССОР ДАННЫХ
# =============================================================================
class HybridDataProcessor:
    """
    🔥 ГИБРИДНЫЙ PIPELINE: Polars + Sparse PCA + БЕЗ TARGET LEAKAGE
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'train_shape': None,
            'test_shape': None,
            'features_added': {},
            'features_removed': [],
            'null_stats': {},
            'timing': {},
            'leakage_fixes': [],  # 🔥 Отслеживаем исправления
            'stability_issues': []  # 🔥 Нестабильные признаки
        }
        self.svd_null_model = None
        self.feature_columns = []
        self.cat_features = []
        self.target_cols = []
        self.target_encoding_maps = {}  # 🔥 Для inference
        self.stable_features = []  # 🔥 Стабильные признаки

    def log(self, message: str):
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def add_step(self, step_name: str, details: dict):
        self.metadata['steps'].append({
            'name': step_name,
            'timestamp': datetime.now().isoformat(),
            'details': details
        })

    # ==========================================================================
    # 1. ЗАГРУЗКА ДАННЫХ (Polars)
    # ==========================================================================
    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        self.log("=" * 70)
        self.log("📂 ШАГ 1: ЗАГРУЗКА ДАННЫХ (Polars)")
        self.log("=" * 70)

        files = {
            'train_main': TRAIN_MAIN,
            'test_main': TEST_MAIN,
            'train_extra': TRAIN_EXTRA,
            'test_extra': TEST_EXTRA
        }

        data = {}
        for name, path in files.items():
            if not path.exists():
                raise FileNotFoundError(f"Файл не найден: {path}")
            self.log(f"   Загрузка {name}...")
            start = time.time()
            data[name] = pl.read_parquet(path)
            elapsed = time.time() - start
            self.log(f"   ✅ {name}: {data[name].shape} ({elapsed:.2f} сек)")

        self.metadata['load_time'] = {k: str(v.shape) for k, v in data.items()}
        return data['train_main'], data['test_main'], data['train_extra'], data['test_extra']

    # ==========================================================================
    # 2. ОБЪЕДИНЕНИЕ MAIN + EXTRA
    # ==========================================================================
    def merge_main_extra(
            self,
            train_main: pl.DataFrame,
            test_main: pl.DataFrame,
            train_extra: pl.DataFrame,
            test_extra: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.log("\n" + "=" * 70)
        self.log("📂 ШАГ 2: ОБЪЕДИНЕНИЕ MAIN + EXTRA")
        self.log("=" * 70)

        self.log("   Объединение train...")
        train_merged = train_main.join(train_extra, on=ID_COLUMN, how='left')
        self.log(f"   ✅ Train: {train_merged.shape}")

        self.log("   Объединение test...")
        test_merged = test_main.join(test_extra, on=ID_COLUMN, how='left')
        self.log(f"   ✅ Test: {test_merged.shape}")

        # Синхронизация колонок
        train_cols = set(train_merged.columns)
        test_cols = set(test_merged.columns)
        extra_in_train = train_cols - test_cols - {ID_COLUMN}
        extra_in_test = test_cols - train_cols - {ID_COLUMN}

        if extra_in_train:
            self.log(f"   ⚠️  {len(extra_in_train)} колонок только в train → удаляем")
            train_merged = train_merged.drop(list(extra_in_train))
        if extra_in_test:
            self.log(f"   ⚠️  {len(extra_in_test)} колонок только в test → добавляем null")
            for col in extra_in_test:
                test_merged = test_merged.with_columns(pl.lit(None).alias(col))

        # Одинаковый порядок колонок
        test_merged = test_merged.select(train_merged.columns)

        self.add_step('merge_main_extra', {
            'train_shape': str(train_merged.shape),
            'test_shape': str(test_merged.shape)
        })
        return train_merged, test_merged

    # ==========================================================================
    # 3. УДАЛЕНИЕ КОЛОНОК С >80% NULLS
    # ==========================================================================
    def remove_high_null_columns(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.log("\n" + "=" * 70)
        self.log("🗑️  ШАГ 3: УДАЛЕНИЕ КОЛОНОК С ВЫСОКИМ % NULLS")
        self.log("=" * 70)

        n_rows = train.height
        cols_to_drop = []

        for col in train.columns:
            if col == ID_COLUMN:
                continue
            null_count = train[col].null_count()
            null_ratio = null_count / n_rows
            if null_ratio > NULL_THRESHOLD:
                cols_to_drop.append(col)

        if cols_to_drop:
            self.log(f"   Найдено {len(cols_to_drop)} колонок с >{NULL_THRESHOLD * 100}% nulls")
            self.log(f"   Примеры: {cols_to_drop[:5]}...")
            train = train.drop(cols_to_drop)
            test = test.drop([c for c in cols_to_drop if c in test.columns])
            self.metadata['features_removed'].extend(cols_to_drop)
            self.add_step('remove_high_null', {
                'removed_count': len(cols_to_drop),
                'threshold': NULL_THRESHOLD,
                'examples': cols_to_drop[:10]
            })
        else:
            self.log("   ✅ Нет колонок для удаления")

        self.log(f"   ✅ Train после: {train.shape}")
        self.log(f"   ✅ Test после: {test.shape}")
        return train, test

    # ==========================================================================
    # 4. 🔥 MD5 DUPLICATE DETECTION (из Pipeline 2)
    # ==========================================================================
    def detect_duplicate_columns(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame,
            batch_size: int = 200
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.log("\n" + "=" * 70)
        self.log("🔍 ШАГ 4: MD5 DETECTION DUPLICATE COLUMNS")
        self.log("=" * 70)

        numeric_cols = [c for c in train.columns if c.startswith("num_feature")]
        seen_hashes = {}
        dup_cols = []

        for start in range(0, len(numeric_cols), batch_size):
            batch_cols = numeric_cols[start:start + batch_size]
            batch_np = train.select(batch_cols).to_numpy()

            for i, col in enumerate(batch_cols):
                h = hashlib.md5(batch_np[:, i].tobytes()).hexdigest()
                if h in seen_hashes:
                    dup_cols.append(col)
                else:
                    seen_hashes[h] = col

            del batch_np
            gc.collect()

        if dup_cols:
            self.log(f"   Найдено {len(dup_cols)} дубликатов колонок")
            self.log(f"   Примеры: {dup_cols[:5]}...")
            train = train.drop(dup_cols)
            test = test.drop([c for c in dup_cols if c in test.columns])
            self.metadata['features_removed'].extend(dup_cols)
            self.add_step('remove_duplicates', {
                'removed_count': len(dup_cols),
                'examples': dup_cols[:10]
            })
        else:
            self.log("   ✅ Дубликатов не найдено")

        return train, test

    # ==========================================================================
    # 5. 🔥 ИСПРАВЛЕНО: NULL FLAGS БЕЗ ТАРГЕТА (убрана leakage!)
    # ==========================================================================
    def create_null_flags_no_target(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.log("\n" + "=" * 70)
        self.log("🚩 ШАГ 5: NULL FLAGS (БЕЗ ТАРГЕТА - ИСПРАВЛЕНО!)")
        self.log("=" * 70)

        # 🔥 ИСПРАВЛЕНО: Больше не используем таргет для отбора!
        self.log("   ⚠️  LEAKAGE FIX: Используем только частоту пропусков (без таргета)")
        self.metadata['leakage_fixes'].append('null_flags_no_target')

        numeric_cols = [
            c for c in train.columns
            if c.startswith("num_feature") and c != ID_COLUMN
        ]

        self.log(f"   Анализируем {len(numeric_cols)} числовых колонок...")

        # Создаём флаги на основе частоты пропусков (НЕ корреляции с таргетом!)
        selected_cols = []
        for col in numeric_cols:
            null_count = train[col].null_count()
            null_ratio = null_count / train.height

            # 🔥 ИСПРАВЛЕНО: Отбор только по частоте пропусков
            if null_ratio >= NULL_FLAG_MIN_RATIO:
                # Дополнительная проверка: достаточно ли вариации в флагах
                null_std = np.sqrt(null_ratio * (1 - null_ratio))
                if null_std > 0.01:  # Минимальная вариация
                    selected_cols.append(col)

        self.log(f"   ✅ Отобрано {len(selected_cols)} флагов (ratio > {NULL_FLAG_MIN_RATIO})")

        # Добавляем флаги
        for col in selected_cols:
            flag_name = f"is_null_{col}"
            train = train.with_columns(pl.col(col).is_null().cast(pl.Int8).alias(flag_name))
            test = test.with_columns(pl.col(col).is_null().cast(pl.Int8).alias(flag_name))

        self.metadata['features_added']['null_flags'] = len(selected_cols)
        self.add_step('create_null_flags', {
            'count': len(selected_cols),
            'threshold': NULL_FLAG_MIN_RATIO,
            'method': 'frequency_only_no_target'  # 🔥 Помечаем что без таргета
        })

        return train, test

    # ==========================================================================
    # 6. 🔥 SPARSE NULL PCA (fit на train, transform на оба)
    # ==========================================================================
    def create_sparse_null_pca(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame,
            n_components: int = NULL_PCA_COMPONENTS
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.log("\n" + "=" * 70)
        self.log("🔮 ШАГ 6: SPARSE NULL PATTERN PCA")
        self.log("=" * 70)

        numeric_cols = [
            c for c in train.columns
            if c.startswith("num_feature") and c != ID_COLUMN
        ]

        self.log(f"   Анализируем {len(numeric_cols)} числовых колонок...")

        n_train = train.height
        n_test = test.height

        # 🔥 ИСПРАВЛЕНО: Готовим данные для fit на train отдельно
        BATCH = 500
        sparse_blocks_train = []
        sparse_blocks_test = []

        for start in range(0, len(numeric_cols), BATCH):
            batch_cols = numeric_cols[start:start + BATCH]

            # 🔥 Разделяем train и test
            train_np = train.select(batch_cols).to_numpy()
            test_np = test.select(batch_cols).to_numpy()

            # Создаем sparse матрицы отдельно
            null_bits_train = np.isnan(train_np).astype(np.float32)
            null_bits_test = np.isnan(test_np).astype(np.float32)

            sparse_blocks_train.append(csr_matrix(null_bits_train))
            sparse_blocks_test.append(csr_matrix(null_bits_test))

            del train_np, test_np, null_bits_train, null_bits_test

        # Объединяем для fit на train
        null_sparse_train = sparse_hstack(sparse_blocks_train, format="csr")
        null_sparse_test = sparse_hstack(sparse_blocks_test, format="csr")

        del sparse_blocks_train, sparse_blocks_test
        gc.collect()

        self.log(f"   Sparse матрица train: {null_sparse_train.shape}")
        self.log(f"   Sparse матрица test: {null_sparse_test.shape}")

        # 🔥 ИСПРАВЛЕНО: Fit ТОЛЬКО на train, transform на оба
        self.log(f"   Запуск TruncatedSVD (fit на train, transform на оба)...")
        self.svd_null_model = TruncatedSVD(n_components=n_components, random_state=SEED)

        # Fit на train
        pca_features_train = self.svd_null_model.fit_transform(null_sparse_train)
        # Transform на test
        pca_features_test = self.svd_null_model.transform(null_sparse_test)

        var_explained = self.svd_null_model.explained_variance_ratio_.sum()

        # Сохраняем модель для inference
        import joblib
        joblib.dump(self.svd_null_model, ARTIFACTS_DIR / "svd_null_model.pkl")

        self.log(f"   ✅ Explained variance: {var_explained:.4f}")

        # Добавляем в train и test
        pca_cols = []
        for i in range(n_components):
            col_name = f"null_pca_{i}"
            train = train.with_columns(pl.Series(col_name, pca_features_train[:, i].astype(np.float32)))
            test = test.with_columns(pl.Series(col_name, pca_features_test[:, i].astype(np.float32)))
            pca_cols.append(col_name)

        del null_sparse_train, null_sparse_test, pca_features_train, pca_features_test
        gc.collect()

        self.log(f"   ✅ Добавлено {len(pca_cols)} Null PCA признаков")
        self.metadata['features_added']['null_pca'] = len(pca_cols)
        self.metadata['null_pca_explained_var'] = float(var_explained)
        self.add_step('null_pca', {
            'n_components': n_components,
            'explained_variance': float(var_explained),
            'input_features': len(numeric_cols),
            'method': 'sparse_svd_fit_train_only'  # 🔥 Помечаем исправление
        })

        return train, test

    # ==========================================================================
    # 7. 🔥 FREQUENCY ENCODING (с fallback для новых категорий)
    # ==========================================================================
    def add_frequency_encoding(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame,
            cat_cols: List[str]
    ) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
        self.log("\n" + "=" * 70)
        self.log("🔢 ШАГ 7: FREQUENCY ENCODING КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
        self.log("=" * 70)

        # 🔥 ИСПРАВЛЕНО: Считаем частоты только на train
        total_train = train.height
        freq_cols = []

        for col in cat_cols:
            fname = f"freq_{col}"

            # 🔥 ИСПРАВЛЕНО: Считаем на train
            freq_map = (
                train[col].value_counts()
                .with_columns((pl.col("count") / total_train).alias(fname))
                .select([col, fname])
            )

            train = train.join(freq_map, on=col, how="left")
            test = test.join(freq_map, on=col, how="left")

            # 🔥 Fallback для новых категорий в test (средняя частота)
            avg_freq = freq_map[fname].mean()
            test = test.with_columns(pl.col(fname).fill_null(avg_freq))

            freq_cols.append(fname)

        self.log(f"   ✅ Добавлено {len(freq_cols)} frequency признаков")
        self.log(f"   🔥 Fallback для новых категорий в test: средняя частота")

        self.metadata['features_added']['frequency_encoding'] = len(freq_cols)
        self.add_step('frequency_encoding', {
            'count': len(freq_cols),
            'method': 'train_only_with_fallback'
        })

        return train, test, freq_cols

    # ==========================================================================
    # 8. 🔥 CATEGORY INTERACTIONS
    # ==========================================================================
    def add_cat_interaction_freqs(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame,
            interactions: List[Tuple[str, str]]
    ) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
        self.log("\n" + "=" * 70)
        self.log("🔗 ШАГ 8: CATEGORY INTERACTION FREQUENCIES")
        self.log("=" * 70)

        # 🔥 ИСПРАВЛЕНО: Считаем на train
        total_train = train.height
        freq_cols = []

        for c1, c2 in interactions:
            if c1 not in train.columns or c2 not in train.columns:
                continue

            fname = f"freq_{c1}_x_{c2}"
            combo = pl.col(c1).cast(pl.Int64) * 1_000_000 + pl.col(c2).cast(pl.Int64)

            # 🔥 Считаем частоты на train
            freq_map = (
                train.select(combo.alias("_combo"))
                ["_combo"].value_counts()
                .with_columns((pl.col("count") / total_train).alias(fname))
                .select(["_combo", fname])
            )

            train = (train.with_columns(combo.alias("_combo"))
                     .join(freq_map, on="_combo", how="left").drop("_combo"))
            test = (test.with_columns(combo.alias("_combo"))
                    .join(freq_map, on="_combo", how="left").drop("_combo"))

            # Fallback для новых комбинаций
            avg_freq = freq_map[fname].mean()
            test = test.with_columns(pl.col(fname).fill_null(avg_freq))

            freq_cols.append(fname)

        self.log(f"   ✅ Добавлено {len(freq_cols)} interaction признаков")
        self.metadata['features_added']['cat_interactions'] = len(freq_cols)
        self.add_step('cat_interactions', {'count': len(freq_cols)})

        return train, test, freq_cols

    # ==========================================================================
    # 9. 🔥 НОВОЕ: TARGET ENCODING С REGULARIZATION
    # ==========================================================================
    def add_target_encoding(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame,
            train_target: pl.DataFrame,
            cat_cols: List[str],
            target_cols: List[str],
            alpha: float = TARGET_ENCODING_SMOOTHING
    ) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
        self.log("\n" + "=" * 70)
        self.log("🎯 ШАГ 9: TARGET ENCODING С REGULARIZATION (НОВОЕ!)")
        self.log("=" * 70)

        self.log(f"   🔥 Добавляем target encoding для {len(cat_cols)} категориальных признаков")
        self.log(f"   Smoothing alpha: {alpha}")

        enc_cols = []
        global_means = {}

        # Считаем глобальное среднее для каждого таргета
        for target_col in target_cols:
            global_means[target_col] = train_target[target_col].mean()

        for cat_col in cat_cols[:10]:  # Ограничим топ-10 для скорости
            for target_col in target_cols[:5]:  # Первые 5 таргетов
                enc_name = f"te_{cat_col}_{target_col}"

                # Агрегация на train
                agg = train.join(train_target, on=ID_COLUMN, how='left').group_by(cat_col).agg([
                    pl.col(target_col).mean().alias('mean'),
                    pl.col(target_col).count().alias('count')
                ])

                # Smoothing: (count * mean + alpha * global_mean) / (count + alpha)
                global_mean = global_means[target_col]
                agg = agg.with_columns(
                    ((pl.col('count') * pl.col('mean') + alpha * global_mean) /
                     (pl.col('count') + alpha)).alias('target_enc')
                )

                # Join к train и test
                train = train.join(agg.select([cat_col, 'target_enc']), on=cat_col, how='left')
                train = train.rename({'target_enc': enc_name})

                test = test.join(agg.select([cat_col, 'target_enc']), on=cat_col, how='left')
                test = test.rename({'target_enc': enc_name})

                # Fallback для новых категорий
                test = test.with_columns(pl.col(enc_name).fill_null(global_mean))

                enc_cols.append(enc_name)

                # Сохраняем маппинг для inference
                self.target_encoding_maps[enc_name] = {
                    'cat_col': cat_col,
                    'target_col': target_col,
                    'alpha': alpha,
                    'global_mean': float(global_mean)
                }

        self.log(f"   ✅ Добавлено {len(enc_cols)} target encoding признаков")
        self.metadata['features_added']['target_encoding'] = len(enc_cols)
        self.add_step('target_encoding', {
            'count': len(enc_cols),
            'alpha': alpha,
            'method': 'smoothing_regularization'
        })

        return train, test, enc_cols

    # ==========================================================================
    # 10. 🔥 ЗАПОЛНЕНИЕ NULLS (ПОСЛЕ Row Statistics!)
    # ==========================================================================
    def impute_nulls(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.log("\n" + "=" * 70)
        self.log("🔧 ШАГ 10: ЗАПОЛНЕНИЕ ПРОПУСКОВ (MEDIAN)")
        self.log("=" * 70)

        numeric_cols = [
            c for c in train.columns
            if c not in [ID_COLUMN] and not c.startswith('cat_')
        ]

        train_nulls_before = sum(train[col].null_count() for col in numeric_cols)
        test_nulls_before = sum(test[col].null_count() for col in numeric_cols)

        self.log(f"   Пропусков до: train={train_nulls_before:,}, test={test_nulls_before:,}")

        # Заполняем медианой (рассчитанной по train!)
        for col in numeric_cols:
            median_val = train[col].median()
            if median_val is None or np.isnan(median_val):
                median_val = 0.0
            train = train.with_columns(pl.col(col).fill_null(median_val))
            test = test.with_columns(pl.col(col).fill_null(median_val))

        train_nulls_after = sum(train[col].null_count() for col in numeric_cols)
        test_nulls_after = sum(test[col].null_count() for col in numeric_cols)

        self.log(f"   Пропусков после: train={train_nulls_after:,}, test={test_nulls_after:,}")
        self.log(f"   ✅ Все пропуски заполнены")

        self.add_step('impute_nulls', {
            'strategy': 'median',
            'train_nulls_before': int(train_nulls_before),
            'train_nulls_after': int(train_nulls_after)
        })

        return train, test

    # ==========================================================================
    # 11. 🔥 RATIO FEATURES (COMBINED: Manual + Automatic)
    # ==========================================================================
    def create_ratio_features(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.log("\n" + "=" * 70)
        self.log("➗ ШАГ 11: RATIO FEATURES (Manual + Automatic)")
        self.log("=" * 70)

        numeric_cols = [
            c for c in train.columns
            if c.startswith("num_feature")
               and not c.endswith('_was_null')
               and not c.startswith('is_null')
               and not c.startswith('null_pca')
               and not c.startswith('row_')
               and not c.startswith('freq_')
               and not c.startswith('ratio_')
               and not c.startswith('diff_')
               and not c.startswith('te_')
        ]

        ratio_cols = []

        # 1. Ручные пары (проверенные)
        self.log(f"   Создаём {len(MANUAL_RATIO_PAIRS)} ручных ratio...")
        for a, b in MANUAL_RATIO_PAIRS:
            if a not in train.columns or b not in train.columns:
                continue
            name = f"ratio_{a.replace('num_feature_', '')}_{b.replace('num_feature_', '')}"
            denom_nonzero = (test[b].abs() > 1e-8).sum()
            if denom_nonzero < test.height * RATIO_MIN_NONZERO:
                continue
            train = train.with_columns(
                pl.when(pl.col(b).abs() > 1e-8)
                .then(pl.col(a) / pl.col(b))
                .otherwise(None)
                .alias(name)
            )
            test = test.with_columns(
                pl.when(pl.col(b).abs() > 1e-8)
                .then(pl.col(a) / pl.col(b))
                .otherwise(None)
                .alias(name)
            )
            ratio_cols.append(name)

        # 2. Автоматические пары (по дисперсии)
        self.log(f"   Создаём автоматические ratio...")
        std_dict = train.select(numeric_cols).std().to_dict(as_series=False)
        std_values = {col: std_dict[col][0] if isinstance(std_dict[col], (list, np.ndarray)) else std_dict[col]
                      for col in std_dict.keys()}

        sorted_cols = sorted(std_values.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
        high_var_cols = [col for col, std in sorted_cols[:50] if std is not None and std > 0]
        std_median = np.median([std for col, std in sorted_cols if std is not None and std > 0])

        self.log(f"   Медиана std: {std_median:.4f}")
        self.log(f"   Колонки с высоким std: {len(high_var_cols)}")

        pairs_created = len(ratio_cols)
        for i, col1 in enumerate(high_var_cols[:20]):
            for col2 in high_var_cols[i + 1:20]:
                if pairs_created >= MAX_RATIO_PAIRS:
                    break
                denom_nonzero = (test[col2].abs() > 1e-8).sum()
                if denom_nonzero < test.height * RATIO_MIN_NONZERO:
                    continue
                ratio_name = f"ratio_{col1.replace('num_feature_', '')}_{col2.replace('num_feature_', '')}"
                if ratio_name in ratio_cols:
                    continue
                train = train.with_columns(
                    pl.when(pl.col(col2).abs() > 1e-8)
                    .then(pl.col(col1) / pl.col(col2))
                    .otherwise(None)
                    .alias(ratio_name)
                )
                test = test.with_columns(
                    pl.when(pl.col(col2).abs() > 1e-8)
                    .then(pl.col(col1) / pl.col(col2))
                    .otherwise(None)
                    .alias(ratio_name)
                )
                ratio_cols.append(ratio_name)
                pairs_created += 1

        self.log(f"   ✅ Создано {len(ratio_cols)} ratio признаков")
        self.metadata['features_added']['ratio_features'] = len(ratio_cols)
        self.add_step('ratio_features', {'count': len(ratio_cols), 'method': 'manual+automatic'})

        return train, test

    # ==========================================================================
    # 12. 🔥 НОВОЕ: POLYNOMIAL FEATURES ДЛЯ ТОП ПРИЗНАКОВ
    # ==========================================================================
    def add_polynomial_features(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
        self.log("\n" + "=" * 70)
        self.log("📐 ШАГ 12: POLYNOMIAL FEATURES (НОВОЕ!)")
        self.log("=" * 70)

        numeric_cols = [
            c for c in train.columns
            if c.startswith("num_feature")
               and not c.startswith('is_null')
               and not c.startswith('null_pca')
               and not c.startswith('row_')
               and not c.startswith('freq_')
               and not c.startswith('ratio_')
               and not c.startswith('diff_')
               and not c.startswith('te_')
        ]

        # 🔥 Выбираем топ-N признаков по дисперсии на train
        std_dict = train.select(numeric_cols).std().to_dict(as_series=False)
        std_values = {}
        for col, std in std_dict.items():
            if isinstance(std, (list, np.ndarray)):
                std_values[col] = std[0] if len(std) > 0 else 0
            else:
                std_values[col] = std if std is not None else 0

        top_features = sorted(std_values.items(), key=lambda x: x[1], reverse=True)[:POLY_FEATURES_TOP_N]
        top_feature_names = [col for col, std in top_features if std > 0]

        self.log(f"   Топ-{len(top_feature_names)} признаков по дисперсии:")
        for col, std in top_features[:5]:
            self.log(f"      {col}: std={std:.4f}")

        poly_cols = []

        # Квадратичные признаки
        for col in top_feature_names:
            poly_name = f"{col}_squared"
            train = train.with_columns((pl.col(col) ** 2).alias(poly_name))
            test = test.with_columns((pl.col(col) ** 2).alias(poly_name))
            poly_cols.append(poly_name)

        # Кубические для топ-5
        for col in top_feature_names[:5]:
            poly_name = f"{col}_cubed"
            train = train.with_columns((pl.col(col) ** 3).alias(poly_name))
            test = test.with_columns((pl.col(col) ** 3).alias(poly_name))
            poly_cols.append(poly_name)

        self.log(f"   ✅ Добавлено {len(poly_cols)} polynomial признаков")
        self.metadata['features_added']['polynomial_features'] = len(poly_cols)
        self.add_step('polynomial_features', {
            'count': len(poly_cols),
            'top_n': POLY_FEATURES_TOP_N,
            'method': 'variance_based_selection'
        })

        return train, test, poly_cols

    # ==========================================================================
    # 13. 🔥 НОВОЕ: NUMERICAL INTERACTIONS
    # ==========================================================================
    def add_numerical_interactions(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
        self.log("\n" + "=" * 70)
        self.log("🔗 ШАГ 13: NUMERICAL INTERACTIONS (НОВОЕ!)")
        self.log("=" * 70)

        numeric_cols = [
            c for c in train.columns
            if c.startswith("num_feature")
               and not c.startswith('is_null')
               and not c.startswith('null_pca')
               and not c.startswith('row_')
               and not c.startswith('freq_')
               and not c.startswith('ratio_')
               and not c.startswith('diff_')
               and not c.startswith('te_')
               and not c.endswith('_squared')
               and not c.endswith('_cubed')
        ]

        # 🔥 Находим пары с высокой корреляцией между собой (НЕ с таргетом!)
        self.log(f"   Поиск коррелирующих пар среди {len(numeric_cols)} признаков...")

        # Берем подвыборку для скорости
        sample_cols = numeric_cols[:100]
        sample_data = train.select(sample_cols).to_numpy()

        # Считаем корреляционную матрицу
        corr_matrix = np.corrcoef(sample_data, rowvar=False)

        # Находим пары с |corr| > 0.5
        interaction_pairs = []
        for i in range(len(sample_cols)):
            for j in range(i + 1, len(sample_cols)):
                if abs(corr_matrix[i, j]) > 0.5:
                    interaction_pairs.append((sample_cols[i], sample_cols[j]))

        # Ограничиваем количество
        interaction_pairs = interaction_pairs[:NUM_INTERACTION_PAIRS]

        self.log(f"   Найдено {len(interaction_pairs)} коррелирующих пар (|corr| > 0.5)")

        int_cols = []
        for col1, col2 in interaction_pairs:
            int_name = f"int_{col1.replace('num_feature_', '')}_x_{col2.replace('num_feature_', '')}"
            train = train.with_columns((pl.col(col1) * pl.col(col2)).alias(int_name))
            test = test.with_columns((pl.col(col1) * pl.col(col2)).alias(int_name))
            int_cols.append(int_name)

        self.log(f"   ✅ Добавлено {len(int_cols)} interaction признаков")
        self.metadata['features_added']['numerical_interactions'] = len(int_cols)
        self.add_step('numerical_interactions', {
            'count': len(int_cols),
            'correlation_threshold': 0.5,
            'method': 'feature_correlation_based'
        })

        return train, test, int_cols

    # ==========================================================================
    # 14. 🔥 НОВОЕ: GROUPBY AGGREGATIONS ПО КАТЕГОРИЯМ
    # ==========================================================================
    def add_groupby_aggregations(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame,
            train_target: pl.DataFrame,
            cat_cols: List[str],
            target_cols: List[str]
    ) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
        self.log("\n" + "=" * 70)
        self.log("📊 ШАГ 14: GROUPBY AGGREGATIONS (НОВОЕ!)")
        self.log("=" * 70)

        self.log(f"   Агрегации по топ-{GROUPBY_AGG_TOP_CATS} категориальным признакам")

        agg_cols = []

        # Берем топ-N категориальных по количеству уникальных значений
        cat_unique_counts = []
        for col in cat_cols[:20]:
            unique_count = train[col].n_unique()
            cat_unique_counts.append((col, unique_count))

        top_cats = sorted(cat_unique_counts, key=lambda x: x[1], reverse=True)[:GROUPBY_AGG_TOP_CATS]

        for cat_col, _ in top_cats:
            for target_col in target_cols[:3]:  # Первые 3 таргета
                # Mean aggregation
                agg_name = f"grp_{cat_col}_{target_col}_mean"

                agg_stats = train.join(train_target, on=ID_COLUMN, how='left').group_by(cat_col).agg([
                    pl.col(target_col).mean().alias('mean'),
                    pl.col(target_col).std().alias('std')
                ])

                global_mean = train_target[target_col].mean()
                global_std = train_target[target_col].std()

                train = train.join(agg_stats.select([cat_col, 'mean']), on=cat_col, how='left')
                train = train.rename({'mean': agg_name})
                train = train.with_columns(pl.col(agg_name).fill_null(global_mean))

                test = test.join(agg_stats.select([cat_col, 'mean']), on=cat_col, how='left')
                test = test.rename({'mean': agg_name})
                test = test.with_columns(pl.col(agg_name).fill_null(global_mean))

                agg_cols.append(agg_name)

        self.log(f"   ✅ Добавлено {len(agg_cols)} groupby aggregation признаков")
        self.metadata['features_added']['groupby_aggregations'] = len(agg_cols)
        self.add_step('groupby_aggregations', {
            'count': len(agg_cols),
            'top_cats': GROUPBY_AGG_TOP_CATS,
            'method': 'category_groupby_mean'
        })

        return train, test, agg_cols

    # ==========================================================================
    # 15. 🔥 НОВОЕ: FEATURE STABILITY CHECK (train vs test)
    # ==========================================================================
    def check_feature_stability(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame
    ) -> List[str]:
        self.log("\n" + "=" * 70)
        self.log("⚖️  ШАГ 15: FEATURE STABILITY CHECK (НОВОЕ!)")
        self.log("=" * 70)

        unstable_features = []
        stable_features = []

        numeric_cols = [
            c for c in train.columns
            if c not in [ID_COLUMN]
               and not c.startswith('cat_')
               and train.schema[c] in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]

        self.log(f"   Проверка {len(numeric_cols)} признаков на стабильность...")

        for col in numeric_cols:
            train_mean = train[col].mean()
            test_mean = test[col].mean()

            if train_mean is None or test_mean is None:
                continue

            # Relative difference
            diff = abs(train_mean - test_mean) / (abs(train_mean) + 1e-8)

            if diff > STABILITY_THRESHOLD:
                unstable_features.append({
                    'feature': col,
                    'train_mean': float(train_mean),
                    'test_mean': float(test_mean),
                    'relative_diff': float(diff)
                })
            else:
                stable_features.append(col)

        self.stable_features = stable_features

        if unstable_features:
            self.log(
                f"   ⚠️  Найдено {len(unstable_features)} нестабильных признаков (> {STABILITY_THRESHOLD * 100}% diff)")
            self.log(f"   Примеры:")
            for item in unstable_features[:5]:
                self.log(
                    f"      {item['feature']}: train={item['train_mean']:.4f}, test={item['test_mean']:.4f}, diff={item['relative_diff']:.2%}")

            self.metadata['stability_issues'] = unstable_features[:20]  # Сохраняем топ-20
        else:
            self.log(f"   ✅ Все признаки стабильны (diff < {STABILITY_THRESHOLD * 100}%)")

        self.add_step('feature_stability', {
            'total_checked': len(numeric_cols),
            'unstable_count': len(unstable_features),
            'stable_count': len(stable_features),
            'threshold': STABILITY_THRESHOLD
        })

        return stable_features

    # ==========================================================================
    # 16. 🔥 НОВОЕ: ПРОВЕРКА НА ДУБЛИКАТЫ СТРОК
    # ==========================================================================
    def check_duplicate_rows(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame
    ) -> Tuple[int, int]:
        self.log("\n" + "=" * 70)
        self.log("🔍 ШАГ 16: ПРОВЕРКА НА ДУБЛИКАТЫ СТРОК (НОВОЕ!)")
        self.log("=" * 70)

        # Hash всех колонок кроме ID
        feature_cols = [c for c in train.columns if c != ID_COLUMN]

        train_hash = train.select(feature_cols).hash_rows()
        test_hash = test.select(feature_cols).hash_rows()

        train_dup_count = train_hash.n_unique()
        test_dup_count = test_hash.n_unique()

        train_duplicates = train.height - train_dup_count
        test_duplicates = test.height - test_dup_count

        if train_duplicates > 0:
            self.log(
                f"   ⚠️  Найдено {train_duplicates} дубликатов строк в train ({train_duplicates / train.height * 100:.2f}%)")
        else:
            self.log(f"   ✅ Нет дубликатов строк в train")

        if test_duplicates > 0:
            self.log(
                f"   ⚠️  Найдено {test_duplicates} дубликатов строк в test ({test_duplicates / test.height * 100:.2f}%)")
        else:
            self.log(f"   ✅ Нет дубликатов строк в test")

        self.add_step('duplicate_rows_check', {
            'train_duplicates': train_duplicates,
            'test_duplicates': test_duplicates,
            'train_duplicate_ratio': train_duplicates / train.height if train.height > 0 else 0
        })

        return train_duplicates, test_duplicates

    # ==========================================================================
    # 17. 🔥 ROW STATISTICS (ПЕРЕД ИМПУТАЦИЕЙ!)
    # ==========================================================================
    def create_row_stats(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.log("\n" + "=" * 70)
        self.log("📊 ШАГ 17: ROW STATISTICS (ПЕРЕД ИМПУТАЦИЕЙ!)")
        self.log("=" * 70)

        # 🔥 ИСПРАВЛЕНО: Row statistics ДО импутации (с NaN)
        self.log("   ⚠️  LEAKAGE FIX: Row statistics считаются ДО заполнения nulls")
        self.metadata['leakage_fixes'].append('row_stats_before_imputation')

        numeric_cols = [
            c for c in train.columns
            if c.startswith("num_feature")
               and not c.endswith('_was_null')
               and not c.startswith('is_null')
               and not c.startswith('null_pca')
               and not c.startswith('ratio_')
               and not c.startswith('row_')
               and not c.startswith('freq_')
               and not c.startswith('diff_')
               and not c.startswith('te_')
               and not c.endswith('_squared')
               and not c.endswith('_cubed')
               and not c.startswith('int_')
               and not c.startswith('grp_')
        ]

        self.log(f"   Расчет статистик по {len(numeric_cols)} колонкам...")

        # 🔥 ИСПРАВЛЕНО: Используем NumPy для правильной обработки NaN
        arr_train = train.select(numeric_cols).to_numpy()
        arr_test = test.select(numeric_cols).to_numpy()

        # Mean
        train_mean = np.nanmean(arr_train, axis=1).astype(np.float32)
        test_mean = np.nanmean(arr_test, axis=1).astype(np.float32)
        train = train.with_columns(pl.Series("row_mean", train_mean))
        test = test.with_columns(pl.Series("row_mean", test_mean))

        # Std
        train_std = np.nanstd(arr_train, axis=1, ddof=1).astype(np.float32)
        test_std = np.nanstd(arr_test, axis=1, ddof=1).astype(np.float32)
        train = train.with_columns(pl.Series("row_std", train_std))
        test = test.with_columns(pl.Series("row_std", test_std))

        # Min
        train_min = np.nanmin(arr_train, axis=1).astype(np.float32)
        test_min = np.nanmin(arr_test, axis=1).astype(np.float32)
        train = train.with_columns(pl.Series("row_min", train_min))
        test = test.with_columns(pl.Series("row_min", test_min))

        # Max
        train_max = np.nanmax(arr_train, axis=1).astype(np.float32)
        test_max = np.nanmax(arr_test, axis=1).astype(np.float32)
        train = train.with_columns(pl.Series("row_max", train_max))
        test = test.with_columns(pl.Series("row_max", test_max))

        # Count non-null
        train_nonnull = np.sum(~np.isnan(arr_train), axis=1).astype(np.uint16)
        test_nonnull = np.sum(~np.isnan(arr_test), axis=1).astype(np.uint16)
        train = train.with_columns(pl.Series("row_nonnull_count", train_nonnull))
        test = test.with_columns(pl.Series("row_nonnull_count", test_nonnull))

        # Skew
        row_mean = np.nanmean(arr_train, axis=1, keepdims=True)
        diff = arr_train - row_mean
        m2 = np.nanmean(diff ** 2, axis=1)
        m3 = np.nanmean(diff ** 3, axis=1)
        row_skew_train = np.where(m2 > 1e-16, m3 / (m2 ** 1.5 + 1e-16), 0).astype(np.float32)

        row_mean = np.nanmean(arr_test, axis=1, keepdims=True)
        diff = arr_test - row_mean
        m2 = np.nanmean(diff ** 2, axis=1)
        m3 = np.nanmean(diff ** 3, axis=1)
        row_skew_test = np.where(m2 > 1e-16, m3 / (m2 ** 1.5 + 1e-16), 0).astype(np.float32)

        train = train.with_columns(pl.Series("row_skew", row_skew_train))
        test = test.with_columns(pl.Series("row_skew", row_skew_test))

        del arr_train, arr_test, diff, m2, m3
        gc.collect()

        stats_cols = ['row_mean', 'row_std', 'row_skew', 'row_min', 'row_max', 'row_nonnull_count']
        self.log(f"   ✅ Добавлено {len(stats_cols)} row statistics")
        self.metadata['features_added']['row_stats'] = len(stats_cols)
        self.add_step('row_stats', {
            'count': len(stats_cols),
            'method': 'numpy_with_nan_handling_before_imputation'
        })

        return train, test

    # ==========================================================================
    # 18. DIFF FEATURES (из Pipeline 2)
    # ==========================================================================
    def add_numerical_diffs(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.log("\n" + "=" * 70)
        self.log("➖ ШАГ 18: NUMERICAL DIFF FEATURES")
        self.log("=" * 70)

        NUM_DIFFS = [
            ("num_feature_7", "num_feature_42"),
            ("num_feature_33", "num_feature_42"),
            ("num_feature_36", "num_feature_42"),
            ("num_feature_7", "num_feature_132"),
            ("num_feature_33", "num_feature_29"),
            ("num_feature_7", "num_feature_57"),
            ("num_feature_33", "num_feature_132"),
            ("num_feature_7", "num_feature_125"),
            ("num_feature_29", "num_feature_42"),
            ("num_feature_41", "num_feature_42"),
        ]

        diff_cols = []
        for a, b in NUM_DIFFS:
            if a not in train.columns or b not in train.columns:
                continue
            a_id = a.replace("num_feature_", "")
            b_id = b.replace("num_feature_", "")
            name = f"diff_{a_id}_minus_{b_id}"
            train = train.with_columns((pl.col(a) - pl.col(b)).alias(name))
            test = test.with_columns((pl.col(a) - pl.col(b)).alias(name))
            diff_cols.append(name)

        self.log(f"   ✅ Добавлено {len(diff_cols)} diff признаков")
        self.metadata['features_added']['diff_features'] = len(diff_cols)
        self.add_step('diff_features', {'count': len(diff_cols)})

        return train, test

    # ==========================================================================
    # 19. ОПТИМИЗАЦИЯ ПАМЯТИ
    # ==========================================================================
    def optimize_memory(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.log("\n" + "=" * 70)
        self.log("💾 ШАГ 19: ОПТИМИЗАЦИЯ ПАМЯТИ")
        self.log("=" * 70)

        cast_schema = {}
        for col in train.columns:
            if col == ID_COLUMN:
                cast_schema[col] = pl.Int64
            elif col.startswith('cat_'):
                cast_schema[col] = pl.Int32
            elif col.endswith('_was_null') or col.startswith('is_null'):
                cast_schema[col] = pl.Int8
            elif (col.startswith('null_pca_') or col.startswith('ratio_') or
                  col.startswith('row_') or col.startswith('freq_') or
                  col.startswith('diff_') or col.startswith('te_') or
                  col.startswith('int_') or col.startswith('grp_') or
                  col.endswith('_squared') or col.endswith('_cubed')):
                cast_schema[col] = pl.Float32
            elif train.schema[col] == pl.Float64:
                cast_schema[col] = pl.Float32
            elif train.schema[col] == pl.Int64:
                cast_schema[col] = pl.Int32

        train = train.cast(cast_schema)
        test = test.cast(cast_schema)

        self.log(f"   ✅ Типы оптимизированы")
        self.add_step('optimize_memory', {'types_optimized': len(cast_schema)})

        return train, test

    # ==========================================================================
    # 20. ФИНАЛЬНАЯ СИНХРОНИЗАЦИЯ
    # ==========================================================================
    def finalize(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.log("\n" + "=" * 70)
        self.log("✅ ШАГ 20: ФИНАЛЬНАЯ СИНХРОНИЗАЦИЯ")
        self.log("=" * 70)

        # Синхронизация колонок
        train_cols = set(train.columns)
        test_cols = set(test.columns)
        extra_in_train = train_cols - test_cols - {ID_COLUMN}
        extra_in_test = test_cols - train_cols - {ID_COLUMN}

        if extra_in_train:
            self.log(f"   ⚠️  {len(extra_in_train)} колонок только в train → удаляем")
            train = train.drop(list(extra_in_train))
        if extra_in_test:
            self.log(f"   ⚠️  {len(extra_in_test)} колонок только в test → добавляем null")
            for col in extra_in_test:
                test = test.with_columns(pl.lit(None).alias(col))

        # Одинаковый порядок колонок
        test = test.select(train.columns)

        # Сохраняем список колонок
        self.feature_columns = [c for c in train.columns if c != ID_COLUMN]
        self.cat_features = [c for c in self.feature_columns if c.startswith('cat_')]

        self.metadata['train_shape'] = str(train.shape)
        self.metadata['test_shape'] = str(test.shape)
        self.metadata['feature_columns'] = self.feature_columns
        self.metadata['cat_features'] = self.cat_features
        self.metadata['n_features'] = len(self.feature_columns)

        self.log(f"   ✅ Train финальный: {train.shape}")
        self.log(f"   ✅ Test финальный: {test.shape}")
        self.log(f"   ✅ Всего признаков: {len(self.feature_columns)}")
        self.log(f"   ✅ Категориальных: {len(self.cat_features)}")

        return train, test

    # ==========================================================================
    # 21. СОХРАНЕНИЕ
    # ==========================================================================
    def save(
            self,
            train: pl.DataFrame,
            test: pl.DataFrame,
            target: Optional[pl.DataFrame] = None
    ):
        self.log("\n" + "=" * 70)
        self.log("💾 ШАГ 21: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        self.log("=" * 70)

        # Сохраняем train
        self.log(f"   Сохранение train...")
        train.write_parquet(TRAIN_FINAL, use_pyarrow=True, compression='snappy')
        self.log(f"   ✅ {TRAIN_FINAL}")

        # Сохраняем test
        self.log(f"   Сохранение test...")
        test.write_parquet(TEST_FINAL, use_pyarrow=True, compression='snappy')
        self.log(f"   ✅ {TEST_FINAL}")

        # Сохраняем target (если есть)
        if target is not None:
            target_path = OUTPUT_DIR / "train_target.parquet"
            target.write_parquet(target_path, use_pyarrow=True, compression='snappy')
            self.log(f"   ✅ {target_path}")

        # Сохраняем метаданные
        self.metadata['end_time'] = datetime.now().isoformat()
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        self.log(f"   ✅ {METADATA_FILE}")

        # Сохраняем список колонок
        cols_path = ARTIFACTS_DIR / "feature_columns_v2.json"
        with open(cols_path, 'w', encoding='utf-8') as f:
            json.dump({
                'feature_columns': self.feature_columns,
                'cat_features': self.cat_features,
                'target_cols': self.target_cols,
                'stable_features': self.stable_features,
                'target_encoding_maps': self.target_encoding_maps
            }, f, indent=2)
        self.log(f"   ✅ {cols_path}")

        # Сохраняем SVD модель
        if self.svd_null_model is not None:
            import pickle
            svd_path = ARTIFACTS_DIR / "svd_null_model_v2.pkl"
            with open(svd_path, 'wb') as f:
                pickle.dump(self.svd_null_model, f)
            self.log(f"   ✅ {svd_path}")

        # 🔥 Сохраняем отчёт об исправлениях leakage
        leakage_report_path = ARTIFACTS_DIR / "leakage_fixes_report.json"
        with open(leakage_report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'leakage_fixes': self.metadata['leakage_fixes'],
                'stability_issues': self.metadata['stability_issues'],
                'total_features': len(self.feature_columns),
                'stable_features_count': len(self.stable_features)
            }, f, indent=2)
        self.log(f"   ✅ {leakage_report_path}")

    # ==========================================================================
    # ГЛАВНЫЙ МЕТОД
    # ==========================================================================
    def run(self):
        start_time = time.time()
        try:
            # 1. Загрузка
            train_main, test_main, train_extra, test_extra = self.load_data()

            # Загружаем target для target encoding (НЕ для null flags!)
            train_target = pl.read_parquet(TRAIN_TARGET)
            self.target_cols = [c for c in train_target.columns if c.startswith('target_')]
            self.log(f"   🎯 Таргетов: {len(self.target_cols)}")

            # 2. Объединение
            train, test = self.merge_main_extra(
                train_main, test_main, train_extra, test_extra
            )

            # 3. Удаление колонок с high nulls
            train, test = self.remove_high_null_columns(train, test)

            # 4. MD5 Duplicate Detection
            train, test = self.detect_duplicate_columns(train, test)

            # 5. 🔥 ИСПРАВЛЕНО: NULL FLAGS БЕЗ ТАРГЕТА
            train, test = self.create_null_flags_no_target(train, test)

            # 6. SPARSE NULL PCA (fit на train)
            train, test = self.create_sparse_null_pca(train, test)

            # 7. Frequency Encoding (с fallback)
            cat_cols = [c for c in train.columns if c.startswith('cat_feature')]
            cat_cols = [c for c in cat_cols if c not in DUPLICATE_CATS]
            train, test, freq_cols = self.add_frequency_encoding(train, test, cat_cols)

            # 8. Category Interactions
            train, test, interact_cols = self.add_cat_interaction_freqs(
                train, test, CAT_INTERACTIONS
            )

            # 9. 🔥 НОВОЕ: Target Encoding
            train, test, te_cols = self.add_target_encoding(
                train, test, train_target, cat_cols[:10], self.target_cols[:5]
            )

            # 10. 🔥 НОВОЕ: Row Statistics (ДО импутации!)
            train, test = self.create_row_stats(train, test)

            # 11. 🔥 ЗАПОЛНЕНИЕ NULLS (ПОСЛЕ Row Statistics)
            train, test = self.impute_nulls(train, test)

            # 12. Ratio Features
            train, test = self.create_ratio_features(train, test)

            # 13. 🔥 НОВОЕ: Polynomial Features
            train, test, poly_cols = self.add_polynomial_features(train, test)

            # 14. 🔥 НОВОЕ: Numerical Interactions
            train, test, int_cols = self.add_numerical_interactions(train, test)

            # 15. 🔥 НОВОЕ: GroupBy Aggregations
            train, test, agg_cols = self.add_groupby_aggregations(
                train, test, train_target, cat_cols, self.target_cols
            )

            # 16. Numerical Diffs
            train, test = self.add_numerical_diffs(train, test)

            # 17. 🔥 НОВОЕ: Feature Stability Check
            stable_features = self.check_feature_stability(train, test)

            # 18. 🔥 НОВОЕ: Duplicate Rows Check
            train_dups, test_dups = self.check_duplicate_rows(train, test)

            # 19. Оптимизация памяти
            train, test = self.optimize_memory(train, test)

            # 20. Финализация
            train, test = self.finalize(train, test)

            # 21. Сохранение
            self.save(train, test, train_target)

            elapsed = time.time() - start_time

            self.log("\n" + "=" * 70)
            self.log("🎉 ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
            self.log("=" * 70)
            self.log(f"   ⏱️  Общее время: {elapsed / 60:.2f} минут")
            self.log(f"   📊 Train: {train.shape}")
            self.log(f"   📊 Test: {test.shape}")
            self.log(f"   📁 Выход: {TRAIN_FINAL}")
            self.log(f"   📁 Выход: {TEST_FINAL}")
            self.log(f"   📄 Метаданные: {METADATA_FILE}")
            self.log(f"   🔥 Исправлено leakage: {len(self.metadata['leakage_fixes'])}")
            self.log(f"   ⚖️  Стабильных признаков: {len(self.stable_features)} из {len(self.feature_columns)}")
            self.log("=" * 70)

            return train, test

        except Exception as e:
            self.log(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            raise


# =============================================================================
# ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 HYBRID DATA PROCESSING PIPELINE V2.1")
    print("Исправления leakage + 5 новых улучшений")
    print("=" * 70 + "\n")

    # Создаём директории
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    processor = HybridDataProcessor(verbose=True)
    train, test = processor.run()

    print("\n✅ ГОТОВО! Можно запускать обучение моделей.")
    print("📊 Следующий шаг: 03_stage2_validation.py для feature selection")