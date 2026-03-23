# =============================================================================
# src/utils/meta_features.py
# Генерация мета-признаков для Stage 2
# 🔧 ИСПРАВЛЕНО: Универсальные пути для команды и GitHub
# =============================================================================

import os
import json
import numpy as np
import polars as pl
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


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
    return Path(__file__).resolve().parent.parent.parent


# 🔥 КОРЕНЬ ПРОЕКТА (авто-определение)
PROJECT_ROOT = get_project_root()

# =============================================================================
# 🔧 КОНСТАНТЫ И ПУТИ (ИСПРАВЛЕНО!)
# =============================================================================

# 🔥 Пути относительно корня проекта (работают везде!)
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# 🔥 Можно переопределить через env variables
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))


# =============================================================================
# КЛАСС META FEATURES GENERATOR
# =============================================================================

class MetaFeaturesGenerator:
    """
    Генератор мета-признаков на основе корреляционной матрицы.

    🔧 ИСПРАВЛЕНИЯ:
    - Универсальные пути для Windows/Linux/Mac
    - Авто-определение корня проекта
    - Поддержка переопределения через env variables
    - Path вместо os.path.join
    - Проверка существования файлов с подсказками

    Использование:
        generator = MetaFeaturesGenerator(artifacts_dir)
        generator.load_correlation_matrix()
        meta_features = generator.generate(X_preds, target_cols)
    """

    def __init__(
            self,
            artifacts_dir: Optional[str] = None
    ):
        # 🔥 ИСПРАВЛЕНО: Используем универсальные пути
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
        self.corr_matrix: Optional[np.ndarray] = None
        self.target_cols: Optional[List[str]] = None
        self.n_best_corr: int = 10
        self.corr_threshold: float = 0.05
        self.metadata: Dict[str, Any] = {}

    def load_correlation_matrix(self) -> None:
        """
        Загружает корреляционную матрицу из JSON.
        🔥 ИСПРАВЛЕНО: Проверка существования файла с подсказками
        """
        # 🔥 ИСПРАВЛЕНО: Используем Path вместо os.path.join
        corr_path = self.artifacts_dir / "corr_matrix_stage1.json"

        if not corr_path.exists():
            raise FileNotFoundError(
                f"❌ Нет корреляционной матрицы: {corr_path}\n"
                f"💡 Запусти сначала: python scripts/02_stage1_proxy_training.py"
            )

        with open(corr_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.target_cols = data['target_cols']
        self.corr_matrix = np.array(data['matrix'])
        self.metadata = data.get('metadata', {})
        self.n_best_corr = self.metadata.get('n_best_corr', 15)
        self.corr_threshold = self.metadata.get('corr_threshold', 0.05)

        print(f"✅ Загружена корреляционная матрица: {self.corr_matrix.shape}")
        print(f"   📊 Таргетов: {len(self.target_cols)}")
        print(f"   🔝 NBest: {self.n_best_corr}, Threshold: {self.corr_threshold}")

    def generate(
            self,
            predictions: Dict[str, np.ndarray],
            n_best_corr: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Генерирует мета-признаки на основе предсказаний.

        Формула: meta_target_i_vs_target_j = pred_j - pred_i

        Args:
            predictions: Словарь {target_col: predictions_array}
            n_best_corr: Количество топ корреляций

        Returns:
            meta_features: Словарь {meta_feature_name: values_array}
        """
        if self.corr_matrix is None:
            self.load_correlation_matrix()

        if n_best_corr is None:
            n_best_corr = self.n_best_corr

        meta_features = {}
        n_samples = len(list(predictions.values())[0])

        for i, target_i in enumerate(self.target_cols):
            if target_i not in predictions:
                continue

            # Топ коррелирующие таргеты
            corr_row = self.corr_matrix[i]
            top_indices = np.argsort(np.abs(corr_row))[-n_best_corr:][::-1]

            for j in top_indices:
                target_j = self.target_cols[j]
                if target_j not in predictions:
                    continue

                # 🔥 ИСПРАВЛЕНО: Убран дублирующийся код
                meta_name = f"meta_{target_i}_vs_{target_j}"
                meta_features[meta_name] = predictions[target_j] - predictions[target_i]

        return meta_features

    def generate_from_dataframe(
            self,
            df_preds: pd.DataFrame,
            n_best_corr: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Генерирует мета-признаки из DataFrame с предсказаниями.

        Args:
            df_preds: DataFrame с колонками target_*
            n_best_corr: Количество топ корреляций

        Returns:
            df_meta: DataFrame с мета-признаками
        """
        predictions = {col: df_preds[col].values for col in df_preds.columns}
        meta_features = self.generate(predictions, n_best_corr)

        return pd.DataFrame(meta_features)

    def get_feature_names(self) -> List[str]:
        """Возвращает имена всех мета-признаков."""
        if self.corr_matrix is None:
            self.load_correlation_matrix()

        feature_names = []
        for i, target_i in enumerate(self.target_cols):
            corr_row = self.corr_matrix[i]
            top_indices = np.argsort(np.abs(corr_row))[-self.n_best_corr:][::-1]

            for j in top_indices:
                if corr_row[j] != 0:
                    target_j = self.target_cols[j]
                    feature_names.append(f"meta_{target_i}_vs_{target_j}")

        return feature_names

    def save_correlation_matrix(
            self,
            corr_matrix: np.ndarray,
            target_cols: List[str],
            metadata: Optional[Dict] = None
    ) -> None:
        """
        🔥 НОВОЕ: Сохраняет корреляционную матрицу в JSON.

        Args:
            corr_matrix: Матрица корреляций
            target_cols: Список таргетов
            metadata: Дополнительные метаданные
        """
        # 🔥 Создаём директорию если не существует
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        corr_path = self.artifacts_dir / "corr_matrix_stage1.json"

        data = {
            'target_cols': target_cols,
            'matrix': corr_matrix.tolist(),
            'metadata': metadata or {}
        }

        with open(corr_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"   💾 Корреляционная матрица сохранена: {corr_path}")

    def save_oof_predictions(
            self,
            oof_predictions: Dict[str, np.ndarray],
            filename: str = "oof_predictions_stage1.parquet"
    ) -> None:
        """
        🔥 НОВОЕ: Сохраняет OOF предсказания в parquet.

        Args:
            oof_predictions: Словарь {target_col: predictions_array}
            filename: Имя файла
        """
        # 🔥 Создаём директорию если не существует
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        save_path = self.artifacts_dir / filename
        df_oof = pd.DataFrame(oof_predictions)
        df_oof.to_parquet(save_path, index=False)

        print(f"   💾 OOF предсказания сохранены: {save_path} ({df_oof.shape})")

    def load_oof_predictions(
            self,
            filename: str = "oof_predictions_stage1.parquet"
    ) -> Dict[str, np.ndarray]:
        """
        🔥 НОВОЕ: Загружает OOF предсказания из parquet.

        Args:
            filename: Имя файла

        Returns:
            oof_predictions: Словарь {target_col: predictions_array}
        """
        load_path = self.artifacts_dir / filename

        if not load_path.exists():
            raise FileNotFoundError(
                f"❌ Нет OOF предсказаний: {load_path}\n"
                f"💡 Запусти сначала: python scripts/02_stage1_proxy_training.py"
            )

        df_oof = pd.read_parquet(load_path)
        oof_predictions = {col: df_oof[col].values for col in df_oof.columns}

        print(f"✅ Загружены OOF предсказания: {len(df_oof)} строк, {len(df_oof.columns)} таргетов")

        return oof_predictions

