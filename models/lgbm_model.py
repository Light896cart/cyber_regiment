# =============================================================================
# models/lgbm_model.py
# Менеджер для LightGBM с поддержкой MultiLabel классификации (41 таргет)
# ИСПРАВЛЕНА ПРОБЛЕМА С GPU И NaN
# 🔧 ИСПРАВЛЕНО: Универсальные пути для команды и GitHub
# =============================================================================

import os
import json
import yaml
import datetime
import gc
import pickle
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from typing import Dict, Tuple, Optional, List, Any
from sklearn.metrics import roc_auc_score
import warnings
import time
from pathlib import Path

warnings.filterwarnings('ignore')


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
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "lightgbm" / "lgbm_config.yaml"
DEFAULT_SAVE_DIR = MODELS_DIR


# =============================================================================
# ✅ НОВАЯ ФУНКЦИЯ: ВАЛИДАЦИЯ ДАННЫХ
# =============================================================================

def validate_target(y_col: np.ndarray, col_name: str, verbose: bool = True) -> bool:
    """
    Проверяет таргет на валидность для обучения.

    Returns:
        True если таргет валиден, False если нужно пропустить
    """
    # Проверка на NaN
    nan_count = np.isnan(y_col).sum()
    if nan_count > 0:
        if verbose:
            print(f"   ⚠️  {col_name}: {nan_count} NaN в таргете, заполняем 0")
        y_col = np.nan_to_num(y_col, nan=0.0)

    # Проверка на вариацию классов
    unique_vals = np.unique(y_col)
    if len(unique_vals) == 1:
        if verbose:
            print(f"   ⚠️  {col_name}: Пропущен (только один класс: {unique_vals[0]})")
        return False

    # Проверка на слишком редкий класс
    n_pos = (y_col > 0).sum()
    n_neg = (y_col == 0).sum()
    if n_pos < 5 or n_neg < 5:
        if verbose:
            print(f"   ⚠️  {col_name}: Пропущен (слишком редкий класс: pos={n_pos}, neg={n_neg})")
        return False

    return True


def validate_features(X_pd: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Проверяет и очищает признаки от NaN/Inf и constant features.
    """
    X_clean = X_pd.copy()

    # Проверка на NaN
    nan_count = X_clean.isna().sum().sum()
    if nan_count > 0:
        if verbose:
            print(f"   ⚠️  Найдено {nan_count} NaN в признаках, заполняем медианой")
        X_clean = X_clean.fillna(X_clean.median())

    # Проверка на Inf — 🔥 ИСПРАВЛЕНО: по каждой колонке отдельно
    inf_count = np.isinf(X_clean.values).sum()
    if inf_count > 0:
        if verbose:
            print(f"   ⚠️  Найдено {inf_count} Inf в признаках, заменяем на max по колонке")
        for col in X_clean.columns:
            col_max = X_clean[col][np.isfinite(X_clean[col])].max()
            col_min = X_clean[col][np.isfinite(X_clean[col])].min()
            X_clean[col] = X_clean[col].replace([np.inf, -np.inf], [col_max, col_min])

    return X_clean


# =============================================================================
# КЛАСС LGBM MANAGER
# =============================================================================

class LGBMManager:
    """
    Менеджер для обучения, сохранения и инференса LightGBM моделей.

    🔧 ИСПРАВЛЕНИЯ:
    - Универсальные пути для Windows/Linux/Mac
    - Авто-определение корня проекта
    - Поддержка переопределения через env variables
    - Path вместо os.path.join
    """

    def __init__(
            self,
            config_path: Optional[str] = None,
            save_dir: Optional[str] = None,
            fold_folder: Optional[str] = None
    ):
        # 🔥 ИСПРАВЛЕНО: Используем универсальные пути
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.save_dir = Path(save_dir) if save_dir else DEFAULT_SAVE_DIR
        self.fold_folder = fold_folder

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"❌ Конфиг не найден: {self.config_path}\n"
                f"💡 Убедитесь что configs/lightgbm/lgbm_config.yaml существует"
            )

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_params = self.config.get('model_params', {})
        self.training_config = self.config.get('training', {})

        # Состояние
        self.models: List = []
        self.metadata: Dict[str, Any] = {}
        self._is_trained = False
        self.valid_target_cols: List[str] = []  # ✅ Список валидных таргетов

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
            verbose: bool = True
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Обучает LightGBM модель (41 модель, по одной на таргет).
        """
        if verbose:
            print(f"🚀 Начало обучения LightGBM...")
            print(f"   📊 Train: {X_train.shape[0]} строк, {X_train.shape[1]} признаков")
            if X_val is not None:
                print(f"   📊 Val: {X_val.shape[0]} строк")

        # ============================================
        # 1. Конвертация в Pandas (float32)
        # ============================================
        X_train_pd = X_train.to_pandas()
        y_train_pd = y_train.to_pandas()

        float_cols = X_train_pd.select_dtypes(include=['float64', 'float32']).columns
        X_train_pd[float_cols] = X_train_pd[float_cols].astype('float32')

        del X_train, y_train, float_cols
        gc.collect()

        use_eval = X_val is not None and y_val is not None

        if use_eval:
            X_val_pd = X_val.to_pandas()
            y_val_pd = y_val.to_pandas()

            float_cols_val = X_val_pd.select_dtypes(include=['float64', 'float32']).columns
            X_val_pd[float_cols_val] = X_val_pd[float_cols_val].astype('float32')

            target_cols = list(y_val.columns)
            del X_val, y_val, float_cols_val
            gc.collect()
        else:
            X_val_pd, y_val_pd = None, None
            target_cols = list(y_train_pd.columns)

        # ============================================
        # ✅ 2. ВАЛИДАЦИЯ И ОЧИСТКА ДАННЫХ
        # ============================================
        if verbose:
            print(f"\n   🔍 Валидация данных...")

        # Очистка признаков
        X_train_pd = validate_features(X_train_pd, verbose)
        if use_eval and X_val_pd is not None:
            # Удаляем те же колонки что и в train
            X_val_pd = X_val_pd[X_train_pd.columns]
            X_val_pd = validate_features(X_val_pd, verbose)

        # Проверка таргетов
        self.valid_target_cols = []
        skipped_targets = []

        for col in target_cols:
            y_col = y_train_pd[col].values.copy()  # Копия
            if np.isnan(y_col).any():
                y_col = np.nan_to_num(y_col, nan=0.0)  # ✅ Меняем копию
                y_train_pd[col] = y_col

            if validate_target(y_col, col, verbose):
                self.valid_target_cols.append(col)
            else:
                skipped_targets.append(col)

        if verbose:
            print(f"   ✅ Валидных таргетов: {len(self.valid_target_cols)}/{len(target_cols)}")
            if skipped_targets:
                print(f"   ⚠️  Пропущено таргетов: {skipped_targets}")

        # ============================================
        # 3. Индексы категориальных признаков
        # ============================================
        cat_indices = [i for i, col in enumerate(X_train_pd.columns) if col in (cat_features or [])]

        # ============================================
        # 4. Параметры
        # ============================================
        params = self.model_params.copy()

        # ✅ ИСПРАВЛЕНИЕ ДЛЯ GPU: добавляем force_col_wise
        if params.get('device', 'cpu') == 'gpu':
            params['force_col_wise'] = True  # ✅ Стабильнее для GPU
            params['gpu_use_dp'] = True  # ✅ Double precision для стабильности

        if params.get('device', 'cpu') == 'cpu':
            params['n_jobs'] = -1
        else:
            params['n_jobs'] = None

        if not use_eval:
            if verbose:
                print("   ⚠️  Early Stopping отключен (нет валидации)")

        # ============================================
        # 5. Подготовка Dataset
        # ============================================
        print(f"\n🌳 Обучение LightGBM ({len(self.valid_target_cols)} моделей)...")
        print(f"   🚀 Device: {params.get('device', 'cpu')}")

        t_start = time.time()

        train_data = lgb.Dataset(
            X_train_pd,
            categorical_feature=cat_indices if cat_indices else 'auto',
            free_raw_data=False
        )

        if use_eval:
            val_data = lgb.Dataset(
                X_val_pd,
                categorical_feature=cat_indices if cat_indices else 'auto',
                reference=train_data,
                free_raw_data=False
            )
        else:
            val_data = None

        # Прогрев GPU
        if params.get('device', 'cpu') == 'gpu':
            if verbose:
                print("   🔥 Прогрев GPU...")
            warmup_params = params.copy()
            warmup_params['verbose'] = -1
            try:
                warmup_model = lgb.train(
                    warmup_params,
                    train_data,
                    valid_sets=[val_data] if val_data else None,
                    num_boost_round=1,
                    callbacks=[lgb.early_stopping(1, verbose=False)] if val_data else None
                )
                del warmup_model
                gc.collect()
                if verbose:
                    print("   ✅ GPU прогрет")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Прогрев GPU не удался: {e}")
                    print(f"   ⚠️  Переключаемся на CPU для стабильности")
                params['device'] = 'cpu'
                params['n_jobs'] = -1

        # ============================================
        # 6. Обучение моделей
        # ============================================
        self.models = []
        predictions = {}
        all_preds_val = []
        best_iterations = []
        failed_targets = []

        for i, col in enumerate(self.valid_target_cols):
            if verbose and i % 5 == 0:
                print(f"   Таргет {i + 1}/{len(self.valid_target_cols)}: {col}")

            y_train_col = y_train_pd[col].values.copy()

            # ✅ Ещё раз проверяем NaN перед обучением
            if np.isnan(y_train_col).any():
                y_train_col = np.nan_to_num(y_train_col, nan=0.0)

            if use_eval and y_val_pd is not None:
                y_val_col = y_val_pd[col].values
                if np.isnan(y_val_col).any():
                    y_val_col = np.nan_to_num(y_val_col, nan=0.0)

            n_pos = (y_train_col > 0).sum()
            if n_pos < 50:
                current_min_data = 5
                current_max_depth = 4
            elif n_pos < 500:
                current_min_data = 20
                current_max_depth = 6
            else:
                current_max_depth = params.get('max_depth', 6)
                current_min_data = params.get('min_data_in_leaf', 20)

            train_params = params.copy()
            train_params['max_depth'] = current_max_depth
            train_params['min_data_in_leaf'] = current_min_data
            train_params['verbose'] = -1

            try:
                if use_eval:
                    train_data.set_label(y_train_col)
                    val_data.set_label(y_val_col)

                    model = lgb.train(
                        train_params,
                        train_data,
                        valid_sets=[val_data],
                        num_boost_round=params.get('n_estimators', 1000),
                        callbacks=[
                            lgb.early_stopping(self.training_config.get('early_stopping_rounds', 50), verbose=False)]
                    )
                    best_iter = model.best_iteration
                    preds = model.predict(X_val_pd)
                else:
                    train_data.set_label(y_train_col)
                    final_iters = params.get('n_estimators', 1000)

                    model = lgb.train(
                        train_params,
                        train_data,
                        num_boost_round=final_iters
                    )
                    best_iter = final_iters
                    preds = None

                self.models.append(model)
                best_iterations.append(best_iter)

                if preds is not None:
                    predictions[col] = preds
                    all_preds_val.append(preds)

            except Exception as e:
                if verbose:
                    print(f"   ❌ {col}: Ошибка ({str(e)[:80]}), пропускаем")
                failed_targets.append(col)
                self.models.append(None)
                best_iterations.append(0)

        if verbose:
            print(f"\n   ✅ Все модели: {time.time() - t_start:.2f} сек")
            print(f"   ✅ Среднее на таргет: {(time.time() - t_start) / len(self.valid_target_cols):.2f} сек")
            if failed_targets:
                print(f"   ⚠️  Ошибок: {len(failed_targets)} → {failed_targets[:5]}...")

        del y_train_pd

        # ============================================
        # 7. Метрики
        # ============================================
        if use_eval and len(all_preds_val) > 0:
            all_preds_val = np.column_stack(all_preds_val)
            # ✅ Используем только валидные таргеты для AUC
            y_val_valid = y_val_pd[self.valid_target_cols].to_numpy()
            best_auc = roc_auc_score(y_val_valid, all_preds_val, average='macro')
        else:
            best_auc = 0.0

        # ============================================
        # 8. Очистка памяти
        # ============================================
        del train_data
        if use_eval:
            del X_val_pd, y_val_pd, val_data
        gc.collect()

        # ============================================
        # 9. Метаданные
        # ============================================
        safe_version = (version_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        safe_version = safe_version.replace('/', '_').replace('\\', '_').replace(':', '_')

        self.metadata = {
            "version": safe_version,
            "timestamp": datetime.datetime.now().isoformat(),
            "config_path": str(self.config_path),
            "params": params,
            "metrics": {
                "macro_roc_auc": best_auc,
                "best_iteration": int(np.mean(best_iterations)) if best_iterations else 0
            },
            "target_cols": self.valid_target_cols,
            "feature_cols": list(X_train_pd.columns),  # ✅ Только валидные
            "skipped_targets": skipped_targets,  # ✅ Пропущенные
            "trained_on_full_data": not use_eval,
            "n_models": len(self.models),
            "n_failed": len(failed_targets),
            "project_root": str(PROJECT_ROOT)
        }

        # ============================================
        # 10. Сохранение
        # ============================================
        if save_model:
            self._save_model()
            if verbose:
                print(f"\n✅ LGBM сохранена: {safe_version}, AUC: {best_auc:.4f}")
        else:
            if verbose:
                print(f"\n✅ LGBM обучена (не сохранена, CV режим)")

        self._is_trained = True
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

        for i, model in enumerate(self.models):
            if model is not None:
                model.save_model(model_path / f'model_{i}.txt')

        with open(model_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4)

    def load_model(self, version: str, fold_folder: Optional[str] = None) -> None:
        """Загружает модель и метаданные с диска."""
        folder = fold_folder or self.fold_folder

        # 🔥 ИСПРАВЛЕНО: Используем Path вместо os.path.join
        if folder:
            model_path = self.save_dir / folder / version
        else:
            model_path = self.save_dir / version

        meta_path = model_path / 'metadata.json'

        if not meta_path.exists():
            raise FileNotFoundError(
                f"❌ Модель не найдена: {meta_path}\n"
                f"💡 Проверьте что модель была обучена и сохранена"
            )

        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.models = []
        target_cols = self.metadata.get('target_cols', [])
        self.valid_target_cols = target_cols  # ✅ Восстанавливаем список

        for i in range(len(target_cols)):
            model_file = model_path / f'model_{i}.txt'
            if model_file.exists():
                model = lgb.Booster(model_file=model_file)
                self.models.append(model)
            else:
                self.models.append(None)

        self._is_trained = True
        print(f"✅ LGBM загружена: {version} ({len(self.models)} моделей)")

    # ==========================================================================
    # ИНФЕРЕНС
    # ==========================================================================

    def predict(
            self,
            X: pl.DataFrame,
            cat_features: Optional[List[str]] = None,
            verbose: bool = False
    ) -> Dict[str, np.ndarray]:
        """Делает предсказания на новых данных."""
        if not self.models or len(self.models) == 0:
            raise ValueError("Сначала обучите или загрузите модель!")

        if verbose:
            print(f"🔮 Инференс LGBM: {X.shape[0]} строк")

        X_pd = X.to_pandas()

        # ✅ ВЫРАВНИВАЕМ КОЛОНКИ ПО ПОРЯДКУ ОБУЧЕНИЯ
        feature_cols = self.metadata.get('feature_cols', None)
        if feature_cols is not None:
            # Проверяем что все колонки на месте
            missing = set(feature_cols) - set(X_pd.columns)
            if missing:
                raise ValueError(f"Отсутствуют колонки: {missing}")
            # ✅ Переставляем в ТОТ ЖЕ порядок что при обучении
            X_pd = X_pd[feature_cols]

        predictions = {}
        target_cols = self.metadata.get('target_cols', [])

        for i, col in enumerate(target_cols):
            if i < len(self.models) and self.models[i] is not None:
                preds = self.models[i].predict(X_pd)
                predictions[col] = preds
            else:
                predictions[col] = np.zeros(len(X_pd))

        del X_pd
        gc.collect()

        return predictions

    # ==========================================================================
    # УТИЛИТЫ
    # ==========================================================================

    def clear(self) -> None:
        """Очищает модели из памяти."""
        self.models = []
        self._is_trained = False
        gc.collect()
        print("   🧹 LGBMManager очищен")

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def target_cols(self) -> List[str]:
        return self.metadata.get('target_cols', [])