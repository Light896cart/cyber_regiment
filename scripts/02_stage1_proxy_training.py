# =============================================================================
# scripts/02_stage1_proxy_training.py
# Stage 1: Multi-Model OOF + Calibration + Stacking
# 🔧 ИСПРАВЛЕНО: Универсальные пути для команды и GitHub
# =============================================================================
# 🔥 ИЗМЕНЕНИЕ: Этот скрипт теперь сохраняет ОДИН объединенный файл для Stage 2
# =============================================================================

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import gc
import json
import os


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
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models_weight"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FOLDS_ROOT = PROJECT_ROOT / "folds"

# 🔥 Можно переопределить через env variables
CONFIGS_DIR = Path(os.getenv("CONFIGS_DIR", DEFAULT_CONFIGS_DIR))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
MODELS_DIR = Path(os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR))
DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
FOLDS_ROOT = Path(os.getenv("FOLDS_ROOT", DEFAULT_FOLDS_ROOT))

# Добавляем корень проекта в путь
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from models.catboost_model import CatBoostManager
from models.lgbm_model import LGBMManager
from models.nn_model import NNManager
from utils.oof_stacking import OOFStackingManager
from sklearn.isotonic import IsotonicRegression
from typing import Dict, List, Tuple


# =============================================================================
# 🔥 НОВОЕ: Менеджер проверок чекпоинтов (Исправлено под реальный API)
# =============================================================================

class ModelCheckpointManager:
    """
    Проверяет существование обученных моделей перед запуском.
    🔥 ИСПРАВЛЕНО: Проверяет наличие metadata.json, как это делают менеджеры.
    """

    def __init__(self, base_weights_dir: Path):
        self.base_weights_dir = base_weights_dir

    def check_model_exists(self, model_name: str, fold_idx: int, stage: str = 'stage1', n_splits: int = 2) -> Path:
        """
        Проверяет, существует ли модель для данного фолда.
        Возвращает имя версии (version_name), если найдена, иначе None.
        """
        # Формируем ожидаемое имя версии, как в manager.train()
        version_name = f"{model_name}_fold_{fold_idx}_{stage}"

        # Путь к папке фолдов (должен совпадать с fold_folder в менеджере)
        fold_folder = f"folds_{n_splits}"
        search_dir = self.base_weights_dir / fold_folder / version_name

        # Проверяем наличие metadata.json (стандарт для всех менеджеров)
        meta_path = search_dir / "metadata.json"

        if meta_path.exists():
            return version_name  # Возвращаем имя версии для load_model

        return None


# =============================================================================
# 🔥 НОВОЕ: Calibration OOF предсказаний
# =============================================================================

def calibrate_oof_predictions(
        oof_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        target_cols: List[str],
        method: str = 'isotonic'
) -> Dict[str, np.ndarray]:
    """
    Калибрует OOF предсказания для каждого таргета.
    """
    print(f"\n🔧 Калибровка OOF предсказаний ({method})...")

    calibrated = {}

    # Фильтруем ID-колонки
    id_keywords = ['id', 'customer', 'client', 'user']
    valid_target_cols = [
        col for col in target_cols
        if not any(keyword in col.lower() for keyword in id_keywords)
    ]

    print(f"   📊 Всего колонок в таргете: {len(target_cols)}")
    print(f"   📊 Колонок для калибровки: {len(valid_target_cols)}")

    for i, col in enumerate(valid_target_cols):
        if col not in oof_predictions:
            print(f"   ⚠️  Пропущено: {col} (нет в OOF)")
            continue

        # Безопасное получение индекса таргета в y_true
        if col in target_cols:
            true_idx = target_cols.index(col)
            y_col = y_true[:, true_idx]
        else:
            continue

        preds_col = oof_predictions[col]

        # Пропускаем если только один класс
        if len(np.unique(y_col)) < 2:
            calibrated[col] = preds_col
            continue

        if method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrated_preds = calibrator.fit_transform(preds_col, y_col)
        else:
            calibrated_preds = preds_col

        calibrated[col] = calibrated_preds.astype(np.float32)

    print(f"   ✅ Откалибровано {len(calibrated)} таргетов")
    return calibrated


# =============================================================================
# 🔥 НОВОЕ: Multi-Model OOF Generator
# =============================================================================

class MultiModelOOFGenerator:
    """
    Генерирует OOF предсказания от нескольких моделей для Stage 1.
    """

    def __init__(self, artifacts_dir: Path, weights_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.weights_dir = weights_dir
        self.oof_predictions = {}  # {model_name: {target_col: preds}}
        self.metadata = {}
        self.checkpoint_manager = ModelCheckpointManager(weights_dir)

    def generate_multi_model_oof(
            self,
            loader,
            n_splits: int = 5,
            models: List[str] = ['catboost', 'lgbm'],
            save_per_fold: bool = True,
            force_retrain: bool = False
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Генерирует OOF от нескольких моделей.
        """
        print(f"\n{'=' * 70}")
        print(f"🔮 MULTI-MODEL OOF GENERATION (Stage 1)")
        print(f"{'=' * 70}")
        print(f"   📊 Фолды: {n_splits}")
        print(f"   🤖 Модели: {models}")
        print(f"   🛡️  Проверка чекпоинтов: {'✅' if not force_retrain else '❌ (Force Retrain)'}")
        print(f"{'=' * 70}")

        n_samples = loader.get_metadata()['n_samples']
        target_cols = loader.target_cols

        # Инициализация OOF массивов для каждой модели
        for model_name in models:
            self.oof_predictions[model_name] = {
                col: np.full(n_samples, np.nan, dtype=np.float32)
                for col in target_cols
            }

        # Цикл по фолдам
        for fold_idx in range(n_splits):
            print(f"\n{'=' * 50}")
            print(f"🔁 ФОЛД {fold_idx + 1}/{n_splits}")
            print(f"{'=' * 50}")

            # Данные фолда
            X_train, y_train, X_val, y_val = loader.get_fold_data(fold_idx)
            train_idx, val_idx = loader.get_fold_idx(fold_idx)

            print(f"   📊 Train: {len(train_idx)}, Val: {len(val_idx)}")

            # Обучаем каждую модель
            for model_name in models:
                print(f"\n   🤖 {model_name.upper()}...")

                # 🔥 ПРОВЕРКА: Существует ли уже обученная модель?
                skip_training = False
                version_name = None

                if not force_retrain:
                    fold_folder_name = f"folds_{n_splits}"
                    version_name = self.checkpoint_manager.check_model_exists(
                        model_name, fold_idx, stage='stage1', n_splits=n_splits
                    )

                    if version_name:
                        print(f"   ⏩ Модель найдена: {version_name}")
                        skip_training = True
                    else:
                        print(f"   ⏳ Модель не найдена, будем обучать...")

                # Инициализация менеджера
                # 🔥 ИСПРАВЛЕНО: Пути через Path
                if model_name == 'catboost':
                    config_path = CONFIGS_DIR / "catboost" / "catboost_config.yaml"
                    if not config_path.exists():
                        raise FileNotFoundError(f"❌ Конфиг не найден: {config_path}")
                    manager = CatBoostManager(
                        config_path=str(config_path),
                        fold_folder=f"folds_{n_splits}"
                    )
                elif model_name == 'lgbm':
                    config_path = CONFIGS_DIR / "lightgbm" / "lgbm_config.yaml"
                    if not config_path.exists():
                        raise FileNotFoundError(f"❌ Конфиг не найден: {config_path}")
                    manager = LGBMManager(
                        config_path=str(config_path),
                        fold_folder=f"folds_{n_splits}"
                    )
                elif model_name == 'nn':
                    config_path = CONFIGS_DIR / "neural_network" / "nn_config.yaml"
                    if not config_path.exists():
                        raise FileNotFoundError(f"❌ Конфиг не найден: {config_path}")
                    manager = NNManager(
                        config_path=str(config_path),
                        fold_folder=f"folds_{n_splits}"
                    )
                else:
                    raise ValueError(f"Неизвестная модель: {model_name}")

                preds = None
                auc = 0.0

                if skip_training and version_name:
                    # 🔥 ЗАГРУЗКА И ПРЕДСКАЗАНИЕ (Исправлено под реальный API)
                    print(f"   📥 Загрузка весов и предсказание...")
                    try:
                        # Используем правильный метод load_model(version, fold_folder)
                        manager.load_model(version_name, fold_folder=f"folds_{n_splits}")
                        preds = manager.predict(X_val)
                        print(f"   ✅ Предсказание готово (без обучения)")

                    except Exception as e:
                        print(f"   ❌ Ошибка при загрузке/предсказании: {e}")
                        print(f"   ⚠️  Откат к обучению с нуля...")
                        skip_training = False

                if not skip_training:
                    # Обучение
                    preds, auc = manager.train(
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        cat_features=loader.cat_features,
                        version_name=f"{model_name}_fold_{fold_idx}_stage1",
                        save_model=save_per_fold,
                        verbose=True
                    )
                    print(f"   📈 {model_name} Fold AUC: {auc:.4f}")

                # Сохраняем OOF по индексам
                if preds:
                    for col in target_cols:
                        if col in preds:
                            self.oof_predictions[model_name][col][val_idx] = preds[col]
                else:
                    print(f"   ⚠️  Нет предсказаний для {model_name} Fold {fold_idx}!")

                # Очистка
                manager.clear()
                del manager
                gc.collect()

            # Очистка данных фолда
            del X_train, y_train, X_val, y_val
            gc.collect()

        # Проверка NaN
        for model_name in models:
            total_nan = sum(
                np.isnan(self.oof_predictions[model_name][col]).sum()
                for col in target_cols
            )
            if total_nan > 0:
                print(f"\n   ⚠️  {model_name}: {total_nan} NaN в OOF!")
            else:
                print(f"\n   ✅ {model_name}: Все строки покрыты")

        # 🔥 СОХРАНЕНИЕ (Один файл)
        self._save_multi_model_oof(models, target_cols)

        return self.oof_predictions

    def _save_multi_model_oof(self, models: List[str], target_cols: List[str]):
        """
        🔥 ИЗМЕНЕНИЕ: Сохраняет OOF от всех моделей в ОДИН объединенный файл.
        Формат колонок: {model_name}_{target_col}
        """
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 1. Создаем единый DataFrame
        combined_oof = {}

        for model_name in models:
            print(f"   🔄 Объединяем {model_name}...")
            model_preds = self.oof_predictions[model_name]

            # Добавляем префикс модели к каждой колонке таргета
            for col in target_cols:
                if col in model_preds:
                    # Ключевое изменение: имя колонки становится "model_target"
                    combined_oof[f"{model_name}_{col}"] = model_preds[col]

        # 2. Сохраняем один файл
        df_combined = pd.DataFrame(combined_oof)
        save_path = self.artifacts_dir / "oof_predictions_STACKED_stage1.parquet"
        df_combined.to_parquet(save_path, index=False)
        print(f"   💾 СОХРАНЕН ОБЩИЙ ФАЙЛ: {save_path}")
        print(f"   📊 Размер: {df_combined.shape}")

        # 3. (Опционально) Сохраняем и отдельные файлы для отладки
        for model_name in models:
            df_oof = pd.DataFrame(self.oof_predictions[model_name])
            save_path_single = self.artifacts_dir / f"oof_predictions_{model_name}_stage1.parquet"
            df_oof.to_parquet(save_path_single, index=False)
            print(f"   💾 {model_name} (отдельно): {save_path_single}")

        # 4. Метаданные
        self.metadata['models'] = models
        self.metadata['target_cols'] = target_cols
        self.metadata['combined_file'] = str(save_path)

        meta_path = self.artifacts_dir / "multi_model_oof_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)


# =============================================================================
# 🔥 НОВОЕ: Adversarial Validation Filter
# =============================================================================

def filter_adversarial_samples(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        threshold: float = 0.7,
        random_state: int = 42
) -> np.ndarray:
    """
    Находит и удаляет строки train которые слишком отличаются от test.
    """
    from lightgbm import LGBMClassifier

    print(f"\n🛡️  Adversarial Validation Filter...")

    n_train = len(X_train)
    n_test = len(X_test)

    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_adv = np.array([0] * n_train + [1] * n_test)

    model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state,
        verbose=-1
    )

    model.fit(X_combined, y_adv)
    train_proba = model.predict_proba(X_train)[:, 1]
    adversarial_idx = np.where(train_proba > threshold)[0]

    print(f"   📊 Всего train: {n_train}")
    print(f"   🗑️  Adversarial samples: {len(adversarial_idx)} ({len(adversarial_idx) / n_train * 100:.1f}%)")

    return adversarial_idx


# =============================================================================
# ОСНОВНОЙ СКРИПТ
# =============================================================================

def main():
    # =========================================================
    # 1. Инициализация
    # =========================================================
    print(f"\n{'=' * 70}")
    print(f"🚀 STAGE 1 V2: MULTI-MODEL OOF + CALIBRATION")
    print(f"{'=' * 70}")
    print(f"   📁 Проект: {PROJECT_ROOT}")
    print(f"   📁 Configs: {CONFIGS_DIR}")
    print(f"   📁 Artifacts: {ARTIFACTS_DIR}")
    print(f"   📁 Models: {MODELS_DIR}")
    print(f"{'=' * 70}")

    loader = DataLoader(cat_strategy="int")

    # =========================================================
    # 2. Загрузка данных
    # =========================================================
    loader.load_full_data()

    # 🔥 ВАЖНО: n_splits должен совпадать с тем, что использовался при генерации фолдов
    # 🔥 ИСПРАВЛЕНО: Используем FOLDS_ROOT из универсальных путей
    loader.load_folds_from_disk(n_splits=3)
    n_splits = loader.n_splits

    # =========================================================
    # 3. Adversarial Validation Filter (опционально)
    # =========================================================
    USE_ADVERSARIAL_FILTER = False

    if USE_ADVERSARIAL_FILTER:
        X_full, _ = loader.get_full_data()

        # 🔥 ИСПРАВЛЕНО: Path вместо хардкода
        test_path = DATA_DIR / "test_final.parquet"
        if not test_path.exists():
            raise FileNotFoundError(f"❌ Тестовые данные не найдены: {test_path}")

        X_test = pd.read_parquet(test_path)

        adversarial_idx = filter_adversarial_samples(
            X_full.to_pandas(),
            X_test,
            threshold=0.7
        )

        adv_path = ARTIFACTS_DIR / "adversarial_samples.json"
        with open(adv_path, 'w') as f:
            json.dump({'indices': adversarial_idx.tolist()}, f)
        print(f"   💾 Adversarial индексы сохранены")

    # =========================================================
    # 4. Multi-Model OOF
    # =========================================================
    oof_generator = MultiModelOOFGenerator(
        artifacts_dir=ARTIFACTS_DIR,
        weights_dir=MODELS_DIR
    )

    FORCE_RETRAIN = False

    oof_predictions = oof_generator.generate_multi_model_oof(
        loader=loader,
        n_splits=n_splits,
        models=['catboost', 'lgbm', 'nn'],
        save_per_fold=True,
        force_retrain=FORCE_RETRAIN
    )

    # =========================================================
    # 5. Calibration OOF предсказаний
    # =========================================================
    USE_CALIBRATION = True

    if USE_CALIBRATION:
        # 🔥 ИСПРАВЛЕНО: Path вместо хардкода
        target_path = DATA_DIR / "train_target.parquet"
        if not target_path.exists():
            raise FileNotFoundError(f"❌ Target данные не найдены: {target_path}")

        y_full = pd.read_parquet(target_path)

        id_cols = [col for col in y_full.columns if 'id' in col.lower() or 'customer' in col.lower()]
        y_full = y_full.drop(columns=id_cols)

        y_np = y_full.to_numpy()
        target_cols = list(y_full.columns)

        print(f"\n🔧 Калибровка: {len(target_cols)} таргетов (без ID)")

        for model_name in oof_predictions.keys():
            oof_predictions[model_name] = calibrate_oof_predictions(
                oof_predictions[model_name],
                y_np,
                target_cols,
                method='isotonic'
            )

    # =========================================================
    # 6. Генерация корреляционной матрицы
    # =========================================================
    oof_manager = OOFStackingManager(
        artifacts_dir=str(ARTIFACTS_DIR)
    )

    oof_manager.oof_predictions = oof_predictions['catboost']
    oof_manager.target_cols = loader.target_cols
    oof_manager.n_samples = loader.get_metadata()['n_samples']

    corr_matrix = oof_manager.generate_correlation_matrix(
        corr_threshold=0.05,
        n_best_corr=15
    )

    # =========================================================
    # 7. Финальный отчёт
    # =========================================================
    print(f"\n{'=' * 70}")
    print(f"✅ STAGE 1 V2 ЗАВЕРШЕН")
    print(f"{'=' * 70}")
    print(f"   📁 OOF файл (STACKED): {ARTIFACTS_DIR / 'oof_predictions_STACKED_stage1.parquet'}")
    print(f"   📁 Corr: {ARTIFACTS_DIR / 'corr_matrix_stage1.json'}")
    print(f"   📊 Фолдов: {n_splits}")
    print(f"   🤖 Моделей: {list(oof_predictions.keys())}")
    print(f"   🔧 Calibration: {'✅' if USE_CALIBRATION else '❌'}")
    print(f"{'=' * 70}")

    meta = {
        'n_splits': n_splits,
        'models': list(oof_predictions.keys()),
        'calibration': USE_CALIBRATION,
        'adversarial_filter': USE_ADVERSARIAL_FILTER,
        'timestamp': pd.Timestamp.now().isoformat(),
        'project_root': str(PROJECT_ROOT)
    }

    meta_path = ARTIFACTS_DIR / "stage1_v2_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()