# =============================================================================
# src/data/loader.py
# Единый DataLoader для всех моделей (CatBoost, XGBoost, LightGBM, NN)
# 🔧 ИСПРАВЛЕНО: Универсальные пути для команды и GitHub
# =============================================================================

import polars as pl
import json
import os
import numpy as np
import gc
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import pandas as pd


# =============================================================================
# 🔧 КОНСТАНТЫ И ПУТИ (ИСПРАВЛЕНО!)
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
    return Path(__file__).resolve().parent.parent.parent


# 🔥 КОРЕНЬ ПРОЕКТА (авто-определение)
PROJECT_ROOT = get_project_root()

# 🔥 Пути относительно корня проекта (работают везде!)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FOLDS_ROOT = PROJECT_ROOT / "folds"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Входные файлы
DEFAULT_TRAIN_PATH = DEFAULT_DATA_DIR / "train_final.parquet"
DEFAULT_TARGET_PATH = DEFAULT_DATA_DIR / "train_target.parquet"
DEFAULT_TEST_PATH = DEFAULT_DATA_DIR / "test_final.parquet"


# =============================================================================
# 🔧 КЛАСС DATALOADER (ИСПРАВЛЕНО!)
# =============================================================================

class DataLoader:
    """
    Единый загрузчик данных для всех моделей.

    🔧 ИЗМЕНЕНИЯ:
    - Пути теперь относительные от корня проекта
    - Работает на Windows/Linux/Mac
    - Можно переопределить через параметры или env variables
    """

    def __init__(
            self,
            train_path: Optional[str] = None,
            target_path: Optional[str] = None,
            test_path: Optional[str] = None,
            folds_root: Optional[str] = None,
            cat_strategy: str = "int"  # "int" или "string"
    ):
        # 🔥 Приоритет: параметр → env variable → дефолт
        self.train_path = Path(train_path) if train_path else Path(
            os.getenv("TRAIN_PATH", DEFAULT_TRAIN_PATH)
        )
        self.target_path = Path(target_path) if target_path else Path(
            os.getenv("TARGET_PATH", DEFAULT_TARGET_PATH)
        )
        self.test_path = Path(test_path) if test_path else Path(
            os.getenv("TEST_PATH", DEFAULT_TEST_PATH)
        )
        self.folds_root = Path(folds_root) if folds_root else Path(
            os.getenv("FOLDS_ROOT", DEFAULT_FOLDS_ROOT)
        )
        self.cat_strategy = cat_strategy

        # Кэш полных данных
        self._df_full: Optional[pl.DataFrame] = None
        self._cat_features: Optional[List[str]] = None
        self._feature_cols: Optional[List[str]] = None
        self._target_cols: Optional[List[str]] = None
        self._is_loaded = False

        # Для фолдов
        self._current_fold_dir: Optional[Path] = None
        self._folds_metadata: Optional[Dict] = None
        self._n_splits: Optional[int] = None

    # ==========================================================================
    # 🔧 ЗАГРУЗКА ДАННЫХ (ИСПРАВЛЕНО!)
    # ==========================================================================

    def load_full_data(self) -> None:
        """
        Загружает 100% обучающих данных в память (один раз).
        🔧 Добавлена проверка существования файлов
        """
        if self._is_loaded:
            print("   ℹ️ Данные уже загружены")
            return

        print("🔄 Загрузка полных данных...")

        # 🔥 Проверка существования файлов
        if not self.train_path.exists():
            raise FileNotFoundError(
                f"❌ Файл train не найден: {self.train_path}\n"
                f"💡 Убедитесь что данные подготовлены (запустите 01_full_data_processing.py)"
            )

        if not self.target_path.exists():
            raise FileNotFoundError(
                f"❌ Файл target не найден: {self.target_path}"
            )

        # Загрузка parquet
        df_train = pl.read_parquet(self.train_path)
        df_target = pl.read_parquet(self.target_path)

        # Объединение по customer_id
        self._df_full = df_train.join(df_target, on="customer_id", how="inner")

        # Определение колонок
        self._target_cols = [col for col in self._df_full.columns if col.startswith('target_')]
        self._cat_features = [col for col in self._df_full.columns if col.startswith('cat_')]
        self._feature_cols = [
            col for col in self._df_full.columns
            if col not in self._target_cols and col != 'customer_id'
        ]

        # Предобработка категориальных признаков
        self._df_full = self._preprocess_categorical(self._df_full, self._cat_features)

        self._is_loaded = True
        print(f"   ✅ Загружено: {self._df_full.shape[0]} строк, {len(self._feature_cols)} признаков")
        print(f"   📊 Целевые колонки: {len(self._target_cols)}")
        print(f"   🏷️  Категориальные признаки: {len(self._cat_features)}")

    def load_folds_from_disk(self, n_splits: int = 5) -> None:
        """
        Загружает метаданные фолдов (индексы, customer_id).
        🔧 Использует Path вместо os.path.join
        """
        fold_dir = self.folds_root / f"folds_{n_splits}"
        meta_path = fold_dir / "fold_metadata.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"❌ Фолды не найдены: {meta_path}\n"
                f"💡 Запусти сначала: python scripts/01_generate_folds.py --n-splits {n_splits}"
            )

        with open(meta_path, 'r', encoding='utf-8') as f:
            self._folds_metadata = json.load(f)

        self._current_fold_dir = fold_dir
        self._n_splits = n_splits

        # Синхронизируем target_cols из метаданных фолдов
        if self._target_cols is None:
            self._target_cols = self._folds_metadata.get('target_cols', [])

        # Синхронизируем cat_features из метаданных фолдов
        if self._cat_features is None:
            self._cat_features = self._folds_metadata.get('cat_features', [])

        print(f"✅ Фолды загружены: {fold_dir} ({n_splits} фолдов)")
        print(f"   📊 {self._folds_metadata.get('n_samples', 'N/A')} строк")

    # ==========================================================================
    # ПОЛУЧЕНИЕ ДАННЫХ ФОЛДА
    # ==========================================================================

    def get_fold_data(
            self,
            fold_idx: int
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Возвращает данные для конкретного фолда.
        🔧 Использует Path вместо os.path.join
        """
        if self._current_fold_dir is None:
            raise ValueError("Сначала вызовите load_folds_from_disk()")
        if not self._is_loaded:
            self.load_full_data()

        # 🔥 Загружаем customer_id для этого фолда (Path вместо os.path.join)
        train_ids = np.load(self._current_fold_dir / f"fold_{fold_idx}_train_ids.npy")
        val_ids = np.load(self._current_fold_dir / f"fold_{fold_idx}_val_ids.npy")

        # Фильтруем данные по customer_id
        X_train, y_train = self._split_and_filter(train_ids)
        X_val, y_val = self._split_and_filter(val_ids)

        return X_train, y_train, X_val, y_val

    def _split_and_filter(
            self,
            customer_ids: np.ndarray
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Разделяет данные на X и y, фильтрует по customer_id.
        """
        # Создаем DataFrame с ID для фильтрации
        ids_df = pl.DataFrame({"customer_id": customer_ids})

        # Фильтр по customer_id (inner join)
        df_filtered = self._df_full.join(ids_df, on="customer_id", how="inner")

        # Разделение на X и y
        X = df_filtered.select(
            [col for col in df_filtered.columns if col in self._feature_cols]
        )
        y = df_filtered.select(self._target_cols)

        return X, y

    # ==========================================================================
    # ПОЛУЧЕНИЕ ИНДЕКСОВ И ID
    # ==========================================================================

    def get_fold_ids(self, fold_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает customer_id для train и val части фолда.
        🔧 Использует Path вместо os.path.join
        """
        if self._current_fold_dir is None:
            raise ValueError("Сначала вызовите load_folds_from_disk()")

        train_ids = np.load(self._current_fold_dir / f"fold_{fold_idx}_train_ids.npy")
        val_ids = np.load(self._current_fold_dir / f"fold_{fold_idx}_val_ids.npy")

        return train_ids, val_ids

    def get_fold_idx(self, fold_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает row indices (0, 1, 2...) для OOF предсказаний.
        🔧 Использует Path вместо os.path.join
        """
        if self._current_fold_dir is None:
            raise ValueError("Сначала вызовите load_folds_from_disk()")

        train_idx = np.load(self._current_fold_dir / f"fold_{fold_idx}_train_idx.npy")
        val_idx = np.load(self._current_fold_dir / f"fold_{fold_idx}_val_idx.npy")

        return train_idx, val_idx

    # ==========================================================================
    # ПОЛУЧЕНИЕ ПОЛНЫХ ДАННЫХ (для Stage 2 / финального обучения)
    # ==========================================================================

    def get_full_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Возвращает 100% обучающих данных (без разделения на фолды).
        """
        if not self._is_loaded:
            self.load_full_data()

        X = self._df_full.select(self._feature_cols)
        y = self._df_full.select(self._target_cols)

        return X, y

    def get_test_data(self) -> Tuple[pl.DataFrame, np.ndarray]:
        """
        Загружает и возвращает тестовые данные.
        🔧 Проверка существования файла
        """
        print("🔄 Загрузка тестовых данных...")

        if not self.test_path.exists():
            raise FileNotFoundError(
                f"❌ Тестовые данные не найдены: {self.test_path}\n"
                f"💡 Убедитесь что test_final.parquet существует в data/"
            )

        df_test = pl.read_parquet(self.test_path)

        customer_ids = df_test["customer_id"].to_numpy()

        # Предобработка категориальных признаков (такая же как для train)
        cat_features = [col for col in df_test.columns if col.startswith('cat_')]
        df_test = self._preprocess_categorical(df_test, cat_features)

        X_test = df_test.select(
            [col for col in df_test.columns if col != 'customer_id']
        )

        print(f"   ✅ Тест: {X_test.shape[0]} клиентов, {X_test.shape[1]} признаков")

        return X_test, customer_ids

    # ==========================================================================
    # УПРАВЛЕНИЕ ПАМЯТЬЮ
    # ==========================================================================

    def clear_cache(self) -> None:
        """
        Очищает кэш данных для экономии памяти.
        """
        if self._df_full is not None:
            del self._df_full
            self._df_full = None
        self._is_loaded = False
        gc.collect()
        print("   🧹 Кэш DataLoader очищен")

    def reload_data(self) -> None:
        """
        Перезагружает данные после очистки кэша.
        """
        self._is_loaded = False
        self.load_full_data()

    # ==========================================================================
    # ПРЕДОБРАБОТКА КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
    # ==========================================================================

    def _preprocess_categorical(
            self,
            df: pl.DataFrame,
            cat_features: List[str]
    ) -> pl.DataFrame:
        """
        Единая предобработка категориальных признаков для всех моделей.
        """
        df_processed = df.clone()

        for col in cat_features:
            if col not in df_processed.columns:
                continue

            col_type = df_processed.schema[col]

            if self.cat_strategy == "int":
                if col_type in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                    df_processed = df_processed.with_columns(
                        pl.col(col).fill_null(-1).cast(pl.Int32)
                    )
                else:
                    df_processed = df_processed.with_columns(
                        pl.col(col).cast(pl.Utf8).fill_null("missing")
                    )
            else:  # string strategy
                df_processed = df_processed.with_columns(
                    pl.col(col).cast(pl.Utf8).fill_null("missing")
                )

        return df_processed

    # ==========================================================================
    # ПРОПЕРТИ И МЕТАДАННЫЕ
    # ==========================================================================

    @property
    def cat_features(self) -> List[str]:
        """Список категориальных признаков."""
        if self._cat_features is None:
            raise ValueError("Данные ещё не загружены. Вызовите load_full_data()")
        return self._cat_features

    @property
    def feature_cols(self) -> List[str]:
        """Список всех признаков (features)."""
        if self._feature_cols is None:
            raise ValueError("Данные ещё не загружены. Вызовите load_full_data()")
        return self._feature_cols

    @property
    def target_cols(self) -> List[str]:
        """Список целевых колонок."""
        if self._target_cols is None:
            raise ValueError("Данные ещё не загружены. Вызовите load_full_data()")
        return self._target_cols

    @property
    def n_splits(self) -> Optional[int]:
        """Количество фолдов."""
        return self._n_splits

    def get_metadata(self) -> Dict[str, Any]:
        """Возвращает метаданные о данных и фолдах."""
        metadata = {
            "n_samples": self._df_full.shape[0] if self._df_full is not None else 0,
            "n_features": len(self._feature_cols) if self._feature_cols else 0,
            "n_targets": len(self._target_cols) if self._target_cols else 0,
            "target_cols": self._target_cols,
            "cat_features": self._cat_features,
            "feature_cols": self._feature_cols,
            "n_splits": self._n_splits,
            "project_root": str(PROJECT_ROOT),
        }

        if self._folds_metadata:
            metadata.update(self._folds_metadata)

        return metadata

    # ==========================================================================
    # МЕТОД ДЛЯ БЫСТРОЙ КОНВЕРТАЦИИ (удобно для моделей)
    # ==========================================================================

    def to_pandas(
            self,
            df: pl.DataFrame
    ) -> pd.DataFrame:
        """
        Конвертирует Polars DataFrame в Pandas.
        """
        return df.to_pandas()

    def to_numpy(
            self,
            df: pl.DataFrame
    ) -> np.ndarray:
        """
        Конвертирует Polars DataFrame в Numpy array.
        """
        return df.to_numpy()


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ТЕСТА
# =============================================================================

def load_test_data(
        test_path: Optional[str] = None,
        cat_strategy: str = "int"
) -> Tuple[pl.DataFrame, np.ndarray, List[str]]:
    """
    Загружает и предобрабатывает тестовые данные.
    🔧 Использует Path вместо os.path
    """
    print("🔄 Загрузка тестовых данных...")

    test_path = Path(test_path) if test_path else DEFAULT_TEST_PATH

    if not test_path.exists():
        raise FileNotFoundError(f"❌ Тестовые данные не найдены: {test_path}")

    df_test = pl.read_parquet(test_path)
    print(f"   📊 Test: {df_test.height} строк")

    customer_ids = df_test["customer_id"].to_numpy()
    feature_cols = [col for col in df_test.columns if col != 'customer_id']
    X_test = df_test.select(feature_cols)

    cat_features = [col for col in feature_cols if col.startswith('cat_')]

    # Предобработка
    loader_temp = DataLoader(cat_strategy=cat_strategy)
    X_test = loader_temp._preprocess_categorical(X_test, cat_features)

    print(f"   ✅ Тест: {X_test.shape[0]} клиентов, {X_test.shape[1]} признаков")
    return X_test, customer_ids, feature_cols
