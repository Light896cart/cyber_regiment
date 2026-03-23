# =============================================================================
# src/models/catboost_model.py
# Менеджер для CatBoostClassifier с поддержкой MultiLogloss (41 таргет)
# 🔧 ИСПРАВЛЕНО: Универсальные пути для команды и GitHub
# =============================================================================

import os
import json
import yaml
import datetime
import gc
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from typing import Dict, Tuple, Optional, List, Any
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
    return Path(__file__).resolve().parent.parent.parent.parent


# 🔥 КОРЕНЬ ПРОЕКТА (авто-определение)
PROJECT_ROOT = get_project_root()

# =============================================================================
# 🔧 КОНСТАНТЫ И ПУТИ (ИСПРАВЛЕНО!)
# =============================================================================

# 🔥 Пути относительно корня проекта (работают везде!)
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models_weight"

# 🔥 Можно переопределить через env variables
CONFIGS_DIR = Path(os.getenv("CONFIGS_DIR", DEFAULT_CONFIGS_DIR))
MODELS_DIR = Path(os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR))

# 🔥 Пути к конфигам по умолчанию
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "catboost" / "catboost_config.yaml"
DEFAULT_SAVE_DIR = MODELS_DIR


# =============================================================================
# КЛАСС CATBOOST MANAGER
# =============================================================================

class CatBoostManager:
    """
    Менеджер для обучения, сохранения и инференса CatBoost моделей.

    Особенности:
    - Поддержка MultiLogloss для multi-label классификации (41 таргет)
    - Работа с Polars DataFrame (конвертирует в Pandas внутри)
    - Поддержка OOF предсказаний для CV
    - Сохранение метаданных для воспроизводимости
    - Агрессивная очистка памяти для работы с большими данными
    - 🔧 Универсальные пути для Windows/Linux/Mac

    Использование:
        manager = CatBoostManager()
        preds, auc = manager.train(X_train, y_train, X_val, y_val, cat_features)
        preds = manager.predict(X_test)
    """

    def __init__(
            self,
            config_path: Optional[str] = None,
            save_dir: Optional[str] = None,
            fold_folder: Optional[str] = None
    ):
        """
        Инициализация менеджера.

        Args:
            config_path: Путь к YAML конфигу (переопределяет дефолт)
            save_dir: Путь для сохранения моделей (переопределяет конфиг)
            fold_folder: Папка для фолдов (например, "folds_2" для Stage 1, "catboost" для Stage 2)
        """
        # 🔥 ИСПРАВЛЕНО: Используем универсальные пути
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.save_dir = Path(save_dir) if save_dir else DEFAULT_SAVE_DIR
        self.fold_folder = fold_folder

        # Загрузка конфига
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"❌ Конфиг не найден: {self.config_path}\n"
                f"💡 Убедитесь что configs/catboost/catboost_config.yaml существует"
            )

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_params = self.config.get('model_params', {})
        self.training_config = self.config.get('training', {})

        # Состояние
        self.model: Optional[CatBoostClassifier] = None
        self.metadata: Dict[str, Any] = {}
        self._is_trained = False

    # ==========================================================================
    # ОБУЧЕНИЕ МОДЕЛИ
    # ==========================================================================

    def train(
            self,
            X_train: pl.DataFrame,
            y_train: pl.DataFrame,
            X_val: Optional[pl.DataFrame] = None,
            y_val: Optional[pl.DataFrame] = None,
            cat_features: Optional[List[str]] = None,
            version_name: Optional[str] = None,
            save_model: bool = True,
            verbose: bool = True,
            optuna_trial=None
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Обучает CatBoost модель.
        """
        if verbose:
            print(f"🚀 Начало обучения CatBoost...")
            print(f"   📊 Train: {X_train.shape[0]} строк, {X_train.shape[1]} признаков")
            if X_val is not None:
                print(f"   📊 Val: {X_val.shape[0]} строк")

        # =========================================================
        # 1. Конвертация Polars → Pandas
        # =========================================================
        try:
            X_train_pd = X_train.to_pandas()
            y_train_pd = y_train.to_pandas()
            target_cols = list(y_train_pd.columns)
        except Exception as e:
            raise RuntimeError(f"Ошибка конвертации Polars → Pandas: {e}")

        del X_train, y_train
        gc.collect()

        use_eval = X_val is not None and y_val is not None
        callbacks = []

        if optuna_trial is not None:
            try:
                from optuna.integration import CatBoostPruningCallback
                pruning_callback = CatBoostPruningCallback(optuna_trial, 'MultiLogloss')
                callbacks.append(pruning_callback)
            except ImportError:
                if verbose:
                    print(f"   ⚠️  Optuna не установлен, пропускаем pruning callback")

        if use_eval:
            try:
                X_val_pd = X_val.to_pandas()
                y_val_pd = y_val.to_pandas()
            except Exception as e:
                raise RuntimeError(f"Ошибка конвертации Val Polars → Pandas: {e}")
            del X_val
            gc.collect()
        else:
            X_val_pd, y_val_pd = None, None

        # =========================================================
        # 2. Создание Pool
        # =========================================================
        cat_features_list = cat_features or []

        train_pool = Pool(
            data=X_train_pd,
            label=y_train_pd,
            cat_features=cat_features_list
        )
        del X_train_pd, y_train_pd
        gc.collect()

        eval_pool = None
        if use_eval:
            eval_pool = Pool(
                data=X_val_pd,
                label=y_val_pd,
                cat_features=cat_features_list
            )
            del X_val_pd
            gc.collect()

        # =========================================================
        # 3. Параметры модели
        # =========================================================
        params = self.model_params.copy()
        params['cat_features'] = cat_features_list

        if 'loss_function' not in params:
            params['loss_function'] = 'MultiLogloss'
        if 'eval_metric' not in params:
            params['eval_metric'] = 'MultiLogloss'

        if not use_eval:
            params['early_stopping_rounds'] = None
            params['use_best_model'] = False

        # =========================================================
        # 4. ✅ ИНИЦИАЛИЗАЦИЯ МОДЕЛИ (ПЕРЕД fit!)
        # =========================================================
        print(f"   🔄 Запуск model.fit()...")
        self.model = CatBoostClassifier(**params)

        # ✅ Теперь безопасно вызываем fit
        if use_eval:
            self.model.fit(train_pool, eval_set=eval_pool, verbose=verbose,
                           callbacks=callbacks if callbacks else None)
        else:
            self.model.fit(train_pool, verbose=verbose,
                           callbacks=callbacks if callbacks else None)

        self._is_trained = True
        print(f"   ✅ Обучение завершено")

        # =========================================================
        # 5. Метрики и предсказания
        # =========================================================
        predictions_matrix = None
        best_auc = 0.0
        best_iteration = params.get('iterations', 0)

        if use_eval and eval_pool is not None:
            predictions_matrix = self.model.predict_proba(eval_pool)
            y_val_np = y_val.to_numpy() if 'y_val' in locals() else None
            del y_val
            if y_val_np is None and y_val_pd is not None:
                y_val_np = y_val_pd.to_numpy()

            if y_val_np is not None:
                try:
                    best_auc = roc_auc_score(
                        y_val_np,
                        predictions_matrix,
                        average='macro',
                        multi_class='ovr'
                    )
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Не удалось рассчитать AUC: {e}")
                    best_auc = 0.0

            best_iteration = self.model.get_best_iteration()
            del eval_pool
            gc.collect()

        del train_pool
        gc.collect()

        # =========================================================
        # 6. Метаданные
        # =========================================================
        safe_version = (
                version_name or
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        safe_version = safe_version.replace('/', '_').replace('\\', '_').replace(':', '_')

        tree_count = self.model.tree_count_ if self.model else 0

        self.metadata = {
            "version": safe_version,
            "timestamp": datetime.datetime.now().isoformat(),
            "config_path": str(self.config_path),
            "params": params,
            "metrics": {
                "macro_roc_auc": best_auc,
                "best_iteration": best_iteration
            },
            "target_cols": target_cols,
            "cat_features": cat_features_list,
            "trained_on_full_data": not use_eval,
            "n_trees": tree_count,
            "project_root": str(PROJECT_ROOT)
        }

        # =========================================================
        # 7. Сохранение
        # =========================================================
        if save_model:
            self._save_model()
            if verbose:
                print(f"✅ Модель сохранена: {safe_version}")
                print(f"   📈 AUC: {best_auc:.4f}, Iterations: {best_iteration}")
        else:
            if verbose:
                print(f"✅ Модель обучена (не сохранена, CV режим)")

        # =========================================================
        # 8. Формирование предсказаний
        # =========================================================
        predictions: Dict[str, np.ndarray] = {}
        if predictions_matrix is not None:
            for i, col in enumerate(target_cols):
                predictions[col] = predictions_matrix[:, i]

        return predictions, best_auc

    # ==========================================================================
    # СОХРАНЕНИЕ И ЗАГРУЗКА
    # ==========================================================================

    def _save_model(self) -> None:
        """Сохраняет модель и метаданные на диск."""
        version = self.metadata['version']

        # 🔥 ИСПРАВЛЕНО: Используем Path вместо os.path.join
        if self.fold_folder:
            model_path = self.save_dir / self.fold_folder / version
        else:
            model_path = self.save_dir / version

        model_path.mkdir(parents=True, exist_ok=True)

        # Сохранение модели
        self.model.save_model(model_path / 'model.cbm')

        # Сохранение метаданных
        with open(model_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4, default=str)

    def load_model(self, version: str, fold_folder: Optional[str] = None) -> None:
        """
        Загружает модель и метаданные с диска.

        Args:
            version: Имя версии модели (папка в save_dir)
            fold_folder: Папка для фолдов (например, "folds_2" для Stage 1, "catboost" для Stage 2)
        """
        # 🔥 ИСПРАВЛЕНО: используем fold_folder если передан
        folder = fold_folder or self.fold_folder

        # 🔥 ИСПРАВЛЕНО: Используем Path вместо os.path.join
        if folder:
            model_path = self.save_dir / folder / version / 'model.cbm'
            meta_path = self.save_dir / folder / version / 'metadata.json'
        else:
            model_path = self.save_dir / version / 'model.cbm'
            meta_path = self.save_dir / version / 'metadata.json'

        if not model_path.exists():
            raise FileNotFoundError(
                f"❌ Нет модели: {model_path}\n"
                f"💡 Проверьте что модель была обучена и сохранена"
            )

        if not meta_path.exists():
            raise FileNotFoundError(
                f"❌ Нет метаданных: {meta_path}\n"
                f"💡 Проверьте что metadata.json существует"
            )

        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        self._is_trained = True

        print(f"✅ Загружена модель: {version}")

    # ==========================================================================
    # ИНФЕРЕНС
    # ==========================================================================

    def predict(
            self,
            X: pl.DataFrame,
            cat_features: Optional[List[str]] = None,
            verbose: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Делает предсказания на новых данных.

        Args:
            X: Признаки (Polars DataFrame)
            cat_features: Список категориальных колонок
            verbose: Выводить ли лог

        Returns:
            predictions: Словарь {target_col: predictions_array}
        """
        if self.model is None:
            raise ValueError("Сначала обучите или загрузите модель!")

        if verbose:
            print(f"🔮 Инференс: {X.shape[0]} строк")

        # Конвертация
        X_pd = X.to_pandas()
        del X
        gc.collect()

        # Создание Pool
        pool = Pool(
            data=X_pd,
            cat_features=cat_features or self.metadata.get('cat_features', [])
        )
        del X_pd
        gc.collect()

        # Предсказание
        preds = self.model.predict_proba(pool)
        del pool
        gc.collect()

        # Формирование словаря
        cols = self.metadata.get(
            'target_cols',
            [f"target_{i}" for i in range(preds.shape[1])]
        )
        predictions = {col: preds[:, i] for i, col in enumerate(cols)}

        return predictions

    # ==========================================================================
    # УТИЛИТЫ
    # ==========================================================================

    def get_feature_importance(
            self,
            top_n: int = 20
    ) -> pd.DataFrame:
        """
        Возвращает важность признаков.

        Args:
            top_n: Количество топ признаков

        Returns:
            DataFrame с признаками и их важностью
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        importance = self.model.get_feature_importance()
        feature_names = self.model.feature_names_

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df.head(top_n)

    def clear(self) -> None:
        """Очищает модель из памяти."""
        if self.model is not None:
            del self.model
            self.model = None
        self._is_trained = False
        gc.collect()
        print("   🧹 CatBoostManager очищен")

    @property
    def is_trained(self) -> bool:
        """Проверяет, обучена ли модель."""
        return self._is_trained

    @property
    def target_cols(self) -> List[str]:
        """Возвращает список целевых колонок."""
        return self.metadata.get('target_cols', [])

    @property
    def cat_features(self) -> List[str]:
        """Возвращает список категориальных признаков."""
        return self.metadata.get('cat_features', [])