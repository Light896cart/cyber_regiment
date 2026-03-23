# =============================================================================
# src/utils/oof_stacking.py
# Генерация OOF предсказаний и корреляционной матрицы для Stage 1 Stacking
# 🔧 ИСПРАВЛЕНО: Универсальные пути для команды и GitHub
# =============================================================================

import os
import json
import gc
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


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
# КЛАСС OOF STACKING MANAGER
# =============================================================================

class OOFStackingManager:
    """
    Менеджер для генерации OOF предсказаний и корреляционной матрицы.

    🔧 ИСПРАВЛЕНИЯ:
    - Универсальные пути для Windows/Linux/Mac
    - Авто-определение корня проекта
    - Поддержка переопределения через env variables
    - Path вместо os.path.join
    - Проверка существования файлов с подсказками

    Использование:
        manager = OOFStackingManager()
        manager.generate_oof_predictions(loader, model_manager, n_splits=5)
        manager.generate_correlation_matrix()
        manager.save_artifacts()
    """

    def __init__(
            self,
            artifacts_dir: Optional[str] = None,
            oof_filename: str = "oof_predictions_stage1.parquet",
            corr_filename: str = "corr_matrix_stage1.json"
    ):
        # 🔥 ИСПРАВЛЕНО: Используем универсальные пути
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
        self.oof_filename = oof_filename
        self.corr_filename = corr_filename

        # 🔥 Создаём директорию если не существует
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Состояние
        self.oof_predictions: Optional[Dict[str, np.ndarray]] = None
        self.corr_matrix: Optional[np.ndarray] = None
        self.target_cols: Optional[List[str]] = None
        self.n_samples: Optional[int] = None
        self.meta_data: Dict[str, Any] = {}

    # ==========================================================================
    # ГЕНЕРАЦИЯ OOF ПРЕДСКАЗАНИЙ
    # ==========================================================================


    def generate_oof_predictions(
            self,
            loader,  # DataLoader
            model_manager,  # CatBoostManager (или любой другой)
            n_splits: int = 5,
            save_per_fold: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Генерирует OOF предсказания для всех строк обучающей выборки.

        Args:
            loader: DataLoader с загруженными данными и фолдами
            model_manager: Менеджер модели (CatBoostManager, LGBMManager, etc.)
            n_splits: Количество фолдов
            save_per_fold: Сохранять ли предсказания после каждого фолда

        Returns:
            oof_predictions: Словарь {target_col: predictions_array}
        """
        print(f"\n{'=' * 60}")
        print(f"🔮 ГЕНЕРАЦИЯ OOF ПРЕДСКАЗАНИЙ (Stage 1 Proxy Models)")
        print(f"{'=' * 60}")
        print(f"   📁 Artifacts: {self.artifacts_dir}")

        # Инициализация
        self.n_samples = loader.get_metadata()['n_samples']
        self.target_cols = loader.target_cols

        # Создаем массивы для OOF предсказаний (заполнены NaN для детекции проблем)
        self.oof_predictions = {
            col: np.full(self.n_samples, np.nan, dtype=np.float32)
            for col in self.target_cols
        }

        fold_scores = []

        # Цикл по фолдам
        for fold_idx in range(n_splits):
            print(f"\n{'=' * 40}")
            print(f"🔁 ФОЛД {fold_idx + 1}/{n_splits}")
            print(f"{'=' * 40}")

            # Получаем данные фолда
            X_train, y_train, X_val, y_val = loader.get_fold_data(fold_idx)
            train_idx, val_idx = loader.get_fold_idx(fold_idx)
            train_ids, val_ids = loader.get_fold_ids(fold_idx)

            print(f"   📊 Train: {len(train_idx)} строк")
            print(f"   📊 Val: {len(val_idx)} строк")

            # Обучаем модель
            preds, auc = model_manager.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                cat_features=loader.cat_features,
                version_name=f"proxy_fold_{fold_idx}_stage1",
                save_model=save_per_fold,  # Сохраняем модели Stage 1
                verbose=True
            )

            # ✅ СОХРАНЯЕМ OOF ПРЕДСКАЗАНИЯ ПО ИНДЕКСАМ
            for col in self.target_cols:
                if col in preds:
                    self.oof_predictions[col][val_idx] = preds[col]

            fold_scores.append(auc)
            print(f"   📈 Fold AUC: {auc:.4f}")

            # Очистка памяти
            model_manager.clear()
            del X_train, y_train, X_val, y_val
            gc.collect()

        # Проверка на NaN (должно быть 0)
        nan_counts = {col: np.isnan(self.oof_predictions[col]).sum() for col in self.target_cols}
        total_nan = sum(nan_counts.values())

        if total_nan > 0:
            print(f"\n   ⚠️  ВНИМАНИЕ: {total_nan} NaN в OOF предсказаниях!")
            print(f"   Возможно, не все строки были покрыты фолдами")
        else:
            print(f"\n   ✅ Все строки покрыты OOF предсказаниями")

        # Сохраняем OOF предсказания
        self._save_oof_predictions()

        # Метрики
        mean_auc = np.mean(fold_scores)
        std_auc = np.std(fold_scores)

        self.metadata['oof'] = {
            'n_splits': n_splits,
            'n_samples': self.n_samples,
            'n_targets': len(self.target_cols),
            'mean_fold_auc': mean_auc,
            'std_fold_auc': std_auc,
            'nan_count': total_nan,
            'timestamp': datetime.now().isoformat(),
            'project_root': str(PROJECT_ROOT)
        }

        print(f"\n{'=' * 60}")
        print(f"✅ OOF ПРЕДСКАЗАНИЯ СГЕНЕРИРОВАНЫ")
        print(f"   📈 Mean CV AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
        print(f"   📁 Сохранено: {self.artifacts_dir / self.oof_filename}")
        print(f"{'=' * 60}")

        return self.oof_predictions

    # ==========================================================================
    # ГЕНЕРАЦИЯ КОРРЕЛЯЦИОННОЙ МАТРИЦЫ
    # ==========================================================================

    def generate_correlation_matrix(
            self,
            corr_threshold: float = 0.05,
            n_best_corr: int = 10
    ) -> np.ndarray:
        """
        Генерирует корреляционную матрицу на основе OOF предсказаний.

        Args:
            corr_threshold: Порог отсечения слабых корреляций
            n_best_corr: Количество топ коррелирующих таргетов для каждого

        Returns:
            corr_matrix: Матрица корреляций (n_targets, n_targets)
        """
        print(f"\n{'=' * 60}")
        print(f"🔢 ГЕНЕРАЦИЯ КОРРЕЛЯЦИОННОЙ МАТРИЦЫ")
        print(f"{'=' * 60}")

        if self.oof_predictions is None:
            raise ValueError("Сначала вызовите generate_oof_predictions()")

        # Формируем матрицу предсказаний (n_samples, n_targets)
        oof_matrix = np.column_stack([
            self.oof_predictions[col] for col in self.target_cols
        ])

        print(f"   📊 Матрица OOF: {oof_matrix.shape}")

        # Считаем корреляцию Пирсона
        print(f"   🔄 Расчет корреляционной матрицы...")
        self.corr_matrix = np.corrcoef(oof_matrix, rowvar=False)

        print(f"   📊 Матрица корреляций: {self.corr_matrix.shape}")

        # Обнуляем диагональ (корреляция с самим собой не нужна)
        np.fill_diagonal(self.corr_matrix, 0)

        # Обнуляем слабые корреляции
        weak_count = np.abs(self.corr_matrix) < corr_threshold
        self.corr_matrix[weak_count] = 0

        zero_corr_ratio = np.sum(self.corr_matrix == 0) / self.corr_matrix.size
        print(f"   ✂️  Отсечено корреляций < {corr_threshold}: {zero_corr_ratio * 100:.1f}%")

        # Находим топ корреляции для каждого таргета
        corr_summary = {}
        for i, col in enumerate(self.target_cols):
            corr_row = self.corr_matrix[i]
            top_indices = np.argsort(np.abs(corr_row))[-n_best_corr:][::-1]
            top_corrs = [(self.target_cols[j], float(corr_row[j])) for j in top_indices if corr_row[j] != 0]
            corr_summary[col] = top_corrs

        self.metadata['correlation'] = {
            'corr_threshold': corr_threshold,
            'n_best_corr': n_best_corr,
            'n_pairs_nonzero': int(np.sum(self.corr_matrix != 0) / 2),  # Делим на 2 (симметричная)
            'target_corr_summary': corr_summary,
            'timestamp': datetime.now().isoformat()
        }

        # Сохраняем
        self._save_correlation_matrix()

        print(f"\n{'=' * 60}")
        print(f"✅ КОРРЕЛЯЦИОННАЯ МАТРИЦА СГЕНЕРИРОВАНА")
        print(f"   📁 Сохранено: {self.artifacts_dir / self.corr_filename}")
        print(f"{'=' * 60}")

        return self.corr_matrix

    # ==========================================================================
    # СОХРАНЕНИЕ АРТЕФАКТОВ
    # ==========================================================================

    def _save_oof_predictions(self) -> None:
        """Сохраняет OOF предсказания в parquet."""
        if self.oof_predictions is None:
            return

        # 🔥 Создаём директорию если не существует
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Создаем DataFrame
        df_oof = pd.DataFrame(self.oof_predictions)

        # 🔥 ИСПРАВЛЕНО: Используем Path вместо os.path.join
        save_path = self.artifacts_dir / self.oof_filename
        df_oof.to_parquet(save_path, index=False)

        print(f"   💾 OOF предсказания сохранены: {df_oof.shape}")

    def _save_correlation_matrix(self) -> None:
        """Сохраняет корреляционную матрицу в JSON."""
        if self.corr_matrix is None:
            return

        # 🔥 Создаём директорию если не существует
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 🔥 ИСПРАВЛЕНО: Используем Path вместо os.path.join
        save_path = self.artifacts_dir / self.corr_filename

        # Сериализуем матрицу
        corr_data = {
            'target_cols': self.target_cols,
            'matrix': self.corr_matrix.tolist(),
            'metadata': self.metadata.get('correlation', {}),
            'project_root': str(PROJECT_ROOT)
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(corr_data, f, indent=2)

        print(f"   💾 Корреляционная матрица сохранена: {self.corr_matrix.shape}")

    def load_correlation_matrix(self) -> np.ndarray:
        """
        Загружает корреляционную матрицу из JSON.
        🔥 ИСПРАВЛЕНО: Проверка существования файла с подсказками
        """
        # 🔥 ИСПРАВЛЕНО: Используем Path вместо os.path.join
        load_path = self.artifacts_dir / self.corr_filename

        if not load_path.exists():
            raise FileNotFoundError(
                f"❌ Нет корреляционной матрицы: {load_path}\n"
                f"💡 Запусти сначала: python scripts/02_stage1_proxy_training.py"
            )

        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.target_cols = data['target_cols']
        self.corr_matrix = np.array(data['matrix'])
        self.metadata['correlation'] = data.get('metadata', {})

        print(f"✅ Загружена корреляционная матрица: {self.corr_matrix.shape}")

        return self.corr_matrix

    def load_oof_predictions(self) -> Dict[str, np.ndarray]:
        """
        Загружает OOF предсказания из parquet.
        🔥 ИСПРАВЛЕНО: Проверка существования файла с подсказками
        """
        # 🔥 ИСПРАВЛЕНО: Используем Path вместо os.path.join
        load_path = self.artifacts_dir / self.oof_filename

        if not load_path.exists():
            raise FileNotFoundError(
                f"❌ Нет OOF предсказаний: {load_path}\n"
                f"💡 Запусти сначала: python scripts/02_stage1_proxy_training.py"
            )

        df_oof = pd.read_parquet(load_path)
        self.oof_predictions = {col: df_oof[col].values for col in df_oof.columns}
        self.target_cols = list(df_oof.columns)
        self.n_samples = len(df_oof)

        print(f"✅ Загружены OOF предсказания: {self.n_samples} строк, {len(self.target_cols)} таргетов")

        return self.oof_predictions

    # ==========================================================================
    # ГЕНЕРАЦИЯ МЕТА-ПРИЗНАКОВ (для Stage 2)
    # ==========================================================================

    def generate_meta_features(
            self,
            predictions: Dict[str, np.ndarray],
            n_best_corr: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Генерирует мета-признаки на основе корреляционной матрицы.

        Формула: для каждого таргета создаем признаки на основе коррелирующих таргетов
        meta_target_i_top_K = pred_target_j - pred_target_i

        Args:
            predictions: Предсказания модели (OOF или тест)
            n_best_corr: Количество топ корреляций (если None, берется из metadata)

        Returns:
            meta_features: Словарь {meta_feature_name: values_array}
        """
        if self.corr_matrix is None:
            raise ValueError("Сначала загрузите корреляционную матрицу")

        if n_best_corr is None:
            n_best_corr = self.metadata.get('correlation', {}).get('n_best_corr', 10)

        meta_features = {}
        n_samples = len(list(predictions.values())[0])

        print(f"   🔄 Генерация мета-признаков...")

        for i, target_i in enumerate(self.target_cols):
            if target_i not in predictions:
                continue

            # Находим топ коррелирующие таргеты
            corr_row = self.corr_matrix[i]
            top_indices = np.argsort(np.abs(corr_row))[-n_best_corr:][::-1]

            for j in top_indices:
                if corr_row[j] == 0:
                    continue

                target_j = self.target_cols[j]
                if target_j not in predictions:
                    continue

                # Создаем мета-признак: разность предсказаний
                meta_name = f"meta_{target_i}_vs_{target_j}"
                meta_features[meta_name] = predictions[target_j] - predictions[target_i]

        print(f"   ✅ Сгенерировано мета-признаков: {len(meta_features)}")

        return meta_features

    def get_metadata(self) -> Dict[str, Any]:
        """Возвращает все метаданные."""
        return self.metadata