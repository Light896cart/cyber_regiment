# =============================================================================
# scripts/tune_catboost_single.py
# Stage 2: ОДИН trials за запуск + сохранение лучшего в JSON
# 🔥 КАЖДЫЙ ЗАПУСК = НОВЫЙ ПРОЦЕСС = 100% очистка GPU памяти
# =============================================================================

import sys
import gc
import os
import json
import warnings
from pathlib import Path
import datetime
import numpy as np
import polars as pl
import pandas as pd
import torch

warnings.filterwarnings('ignore')

# =============================================================================
# 1. НАСТРОЙКА ПУТЕЙ
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

ROOT_DIR = get_project_root()
sys.path.append(str(ROOT_DIR))

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CONFIG_PATH = ROOT_DIR / "configs" / "catboost" / "catboost_config_stage2.yaml"
OPTUNA_RESULTS_DIR = ARTIFACTS_DIR / "optuna_results"
OPTUNA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 🔥 Файл для хранения ЛУЧШИХ результатов
BEST_PARAMS_PATH = OPTUNA_RESULTS_DIR / "best_params.json"
TRIALS_HISTORY_PATH = OPTUNA_RESULTS_DIR / "trials_history.json"


# =============================================================================
# 2. ЗАГРУЗКА ДАННЫХ
# =============================================================================

def load_stage2_data(sample_ratio=0.05):
    """Загружает данные с сэмплированием."""
    print(f"\n📂 Загрузка данных...")

    X_TRAIN_PATH = ARTIFACTS_DIR / "X_train_extended_val.parquet"
    Y_TRAIN_PATH = ARTIFACTS_DIR / "y_train_val.parquet"
    X_VAL_PATH = ARTIFACTS_DIR / "X_val_extended_val.parquet"
    Y_VAL_PATH = ARTIFACTS_DIR / "y_val_val.parquet"
    FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance_stage2.csv"

    X_train_pd = pd.read_parquet(X_TRAIN_PATH)
    y_train_pd = pd.read_parquet(Y_TRAIN_PATH)
    X_val_pd = pd.read_parquet(X_VAL_PATH)
    y_val_pd = pd.read_parquet(Y_VAL_PATH)

    cat_features_filtered = []
    if FEATURE_IMPORTANCE_PATH.exists():
        imp_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        selected_features = imp_df.head(1500)['feature'].tolist()
        common_cols = [c for c in selected_features if c in X_train_pd.columns]
        X_train_pd = X_train_pd[common_cols]
        X_val_pd = X_val_pd[common_cols]

        try:
            from src.data.loader import DataLoader
            loader = DataLoader(cat_strategy="int")
            loader.load_full_data()
            all_cat_features = getattr(loader, 'cat_features', [])
            cat_features_filtered = [f for f in all_cat_features if f in common_cols]
        except:
            pass

    # Сэмплирование
    if sample_ratio < 1.0:
        n_samples = int(len(X_train_pd) * sample_ratio)
        np.random.seed()  # Случайный seed каждый запуск
        sample_idx = np.random.choice(len(X_train_pd), n_samples, replace=False)
        X_train_pd = X_train_pd.iloc[sample_idx]
        y_train_pd = y_train_pd.iloc[sample_idx]
        print(f"   📊 Сэмплирование: {sample_ratio * 100:.0f}% ({n_samples:,} строк)")
    del sample_idx
    gc.collect()

    # Конвертация в Polars
    X_train_pl = pl.from_pandas(X_train_pd)
    y_train_pl = pl.from_pandas(y_train_pd)
    X_val_pl = pl.from_pandas(X_val_pd)
    y_val_pl = pl.from_pandas(y_val_pd)

    del X_train_pd, y_train_pd, X_val_pd, y_val_pd
    gc.collect()

    print(f"   ✅ Train: {X_train_pl.shape}")
    print(f"   ✅ Val: {X_val_pl.shape}")

    return X_train_pl, y_train_pl, X_val_pl, y_val_pl, cat_features_filtered


# =============================================================================
# 3. ГЕНЕРАЦИЯ ПАРАМЕТРОВ (ИСПРАВЛЕНО!)
# =============================================================================

def suggest_iterations_from_history(history, best_params):
    """
    🔥 УМНЫЙ подбор iterations на основе истории (эмуляция Optuna)
    Анализирует какие iterations давали лучший AUC
    """
    if len(history) < 3:
        # Мало данных — случайный выбор в широком диапазоне
        return np.random.randint(100, 2000)

    # Анализируем топ-3 trials по AUC
    history_sorted = sorted(history, key=lambda x: x['score'], reverse=True)[:5]
    top_iterations = [t['params'].get('iterations', 500) for t in history_sorted]

    # Среднее по лучшим + небольшой разброс
    base_iterations = int(np.mean(top_iterations))
    noise = np.random.randint(-100, 101)  # ±100
    iterations = int(np.clip(base_iterations + noise, 50, 5000))

    print(f"   📊 iterations: {iterations} (на основе топ-{len(history_sorted)} trials)")

    return iterations


def suggest_param_from_history(history, param_name, default_range, best_params=None):
    """
    🔥 УМНЫЙ подбор параметра на основе истории (эмуляция Optuna suggest)
    """
    if len(history) < 5 or best_params is None:
        # Мало данных — случайный выбор
        if isinstance(default_range, tuple):
            if param_name in ['learning_rate', 'l2_leaf_reg']:
                return 10 ** np.random.uniform(np.log10(default_range[0]), np.log10(default_range[1]))
            else:
                return np.random.uniform(default_range[0], default_range[1])
        else:
            return np.random.randint(default_range[0], default_range[1])

    # Берём лучшие параметры + небольшой шум
    base_value = best_params.get(param_name, np.mean(default_range))

    # Разный шум для разных параметров
    if param_name == 'depth':
        noise = np.random.randint(-1, 2)
        return int(np.clip(base_value + noise, default_range[0], default_range[1]))
    elif param_name in ['learning_rate', 'l2_leaf_reg']:
        noise = 10 ** np.random.uniform(-0.3, 0.3)
        return float(np.clip(base_value * noise, default_range[0], default_range[1]))
    elif param_name in ['subsample']:
        noise = np.random.uniform(-0.1, 0.1)
        return float(np.clip(base_value + noise, default_range[0], default_range[1]))
    elif param_name == 'border_count':
        noise = np.random.randint(-30, 31)
        return int(np.clip(base_value + noise, default_range[0], default_range[1]))
    elif param_name == 'min_data_in_leaf':
        noise = np.random.randint(-10, 11)
        return int(np.clip(base_value + noise, default_range[0], default_range[1]))
    else:
        return base_value


def generate_params(best_params=None, trial_number=0, history=None, best_score=0):
    """
    Генерирует новые параметры.
    🔥 УМНЫЙ подбор на основе истории (эмуляция Optuna)
    """

    history = history or []

    if best_params and trial_number > 5:
        # 🔥 Эксплуатация: варьируем вокруг лучших параметров
        print(f"   🎯 Эксплуатация (вариация вокруг лучших)")
        params = {
            'task_type': 'GPU',
            'verbose': 10,
            'random_seed': 42 + trial_number,

            # 🔥 ФИКСИРОВАННО: 2000 итераций (early stopping остановит раньше)
            'iterations': 2000,
            'early_stopping_rounds': 100,
            'use_best_model': True,

            # 🔥 ФИКСИРОВАННО: глубина 10 (как в production конфиге)
            'depth': 7,

            # Базовые настройки
            'bootstrap_type': 'Bernoulli',
            'loss_function': 'MultiLogloss',
            'eval_metric': 'MultiLogloss',

            # 🔥 НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ (12+)
            'subsample': suggest_param_from_history(history, 'subsample', (0.4, 1.0), best_params),
            'learning_rate': suggest_param_from_history(history, 'learning_rate', (0.01, 0.5), best_params),
            'l2_leaf_reg': suggest_param_from_history(history, 'l2_leaf_reg', (0.5, 20.0), best_params),
            'border_count': suggest_param_from_history(history, 'border_count', (64, 255), best_params),
            'min_data_in_leaf': suggest_param_from_history(history, 'min_data_in_leaf', (5, 100), best_params),

            # 🔥 НОВЫЕ ПАРАМЕТРЫ ДЛЯ ТЮНИНГА
        
            'random_strength': suggest_param_from_history(history, 'random_strength', (0.0, 10.0), best_params),
            
        }
    else:
        # 🔥 Исследование: случайный поиск
        print(f"   🔍 Исследование (случайный поиск)")
        params = {
            'task_type': 'GPU',
            'verbose': 10,
            'random_seed': 42 + trial_number,

            # 🔥 ФИКСИРОВАННО: 2000 итераций
            'iterations': 2000,
            'early_stopping_rounds': 100,
            'use_best_model': True,

            # 🔥 ФИКСИРОВАННО: глубина 10
            'depth': 7,

            # Базовые настройки
            'bootstrap_type': 'Bernoulli',
            'loss_function': 'MultiLogloss',
            'eval_metric': 'MultiLogloss',

            # 🔥 ВСЕ ПАРАМЕТРЫ СЛУЧАЙНЫЕ
            'subsample': np.random.uniform(0.4, 1.0),
            'learning_rate': 10 ** np.random.uniform(-2, -0.3),  # 0.01 to 0.5
            'l2_leaf_reg': 10 ** np.random.uniform(-0.3, 1.3),  # 0.5 to 20.0
            'border_count': np.random.randint(64, 255),
            'min_data_in_leaf': np.random.randint(5, 100),

            # 🔥 НОВЫЕ ПАРАМЕТРЫ
            
            'random_strength': np.random.uniform(0.0, 10.0),
        }

    return params


# =============================================================================
# 4. ЗАГРУЗКА/СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

def load_best_params():
    """Загружает лучшие параметры и считает trials из истории."""

    # 🔥 Считаем реальное количество trials из истории
    trial_count = 0
    history = []

    if TRIALS_HISTORY_PATH.exists():
        try:
            with open(TRIALS_HISTORY_PATH, 'r', encoding='utf-8') as f:
                history = json.load(f)
                trial_count = len(history)
                print(f"   📊 Найдено {trial_count} trials в истории")
        except Exception as e:
            print(f"   ⚠️  Не удалось прочитать историю: {e}")
            trial_count = 0
            history = []

    # Загружаем лучшие параметры отдельно
    best_params = None
    best_score = 0
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        best_params = data.get('best_params', None)
        best_score = data.get('best_score', 0)
        print(f"   ✅ Лучший AUC: {best_score:.4f} (trial {data.get('trial_number', 0)})")
    else:
        print(f"   📝 Нет сохранённых параметров (первый запуск)")

    return best_params, best_score, trial_count, history


def save_best_params(params, score, trial_number, improved=False):
    """Сохраняет лучшие параметры если score лучше предыдущего."""
    current_best = 0
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            current_best = data.get('best_score', 0)

    if score > current_best or improved:
        data = {
            'best_params': params,
            'best_score': score,
            'trial_number': trial_number,
            'datetime': datetime.datetime.now().isoformat(),
            'improved': improved,
            'note': 'iterations=2000, depth=10 (фиксированные)'
        }
        with open(BEST_PARAMS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"   💾 НОВЫЙ РЕКОРД! AUC: {score:.4f} (было: {current_best:.4f})")
        return True
    else:
        print(f"   ⏭️  Не лучше текущего рекорда ({current_best:.4f})")
        return False


def append_trial_history(params, score, trial_number):
    """Добавляет trial в историю."""
    history = []
    if TRIALS_HISTORY_PATH.exists():
        try:
            with open(TRIALS_HISTORY_PATH, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []

    history.append({
        'trial_number': trial_number,
        'score': score,
        'params': params,
        'datetime': datetime.datetime.now().isoformat()
    })

    with open(TRIALS_HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# =============================================================================
# 5. ОСНОВНОЙ ЗАПУСК (ОДИН TRIALS)
# =============================================================================

def main():
    print(f"\n{'=' * 70}")
    print(f"🔮 CATBOOST SINGLE TRIAL (GPU)")
    print(f"{'=' * 70}")
    print(f"🚀 Каждый запуск = новый процесс = 100% очистка GPU")
    print(f"📊 iterations: 2000 (фиксировано, early stopping остановит раньше)")
    print(f"🌲 depth: 10 (фиксировано)")
    print(f"🔧 Настраиваемых параметров: 12+")
    print(f"💾 Лучшее сохраняется в: {BEST_PARAMS_PATH}")
    print(f"{'=' * 70}\n")

    # Проверка GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} ГБ)")
            print(f"   📊 Свободно: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.1f} ГБ")
        else:
            print("⚠️  GPU не обнаружен, переключение на CPU")
    except:
        pass

    # Загрузка лучших параметров + ИСТОРИЯ
    best_params, best_score, last_trial, history = load_best_params()
    trial_number = last_trial + 1

    print(f"   🔢 Trial: {trial_number}")

    # Генерация параметров (с историей для умного подбора)
    params = generate_params(best_params, trial_number, history, best_score)

    print(f"\n⚙️  Параметры:")
    for k, v in params.items():
        if k not in ['task_type', 'devices', 'thread_count', 'gpu_ram_part', 'verbose', 'random_seed',
                     'early_stopping_rounds', 'use_best_model', 'bootstrap_type', 'loss_function', 'eval_metric']:
            print(f"   {k}: {v}")

    # Загрузка данных
    X_train, y_train, X_val, y_val, cat_features = load_stage2_data(sample_ratio=0.5)

    # Обучение
    print(f"\n🚀 Обучение...")
    from models.catboost_model import CatBoostManager

    manager = CatBoostManager(
        config_path=str(CONFIG_PATH),
        save_dir=str(ROOT_DIR / "models_weight"),
        fold_folder="optuna_trials"
    )

    manager.model_params.update(params)

    score = 0.0
    try:
        _, auc = manager.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cat_features=cat_features,
            version_name=f"trial_{trial_number}",
            save_model=False,
            verbose=10,
            optuna_trial=None
        )
        score = auc
        print(f"   ✅ AUC: {auc:.4f}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        score = 0.0
    finally:
        manager.clear()
        del manager

        # Очистка данных
        del X_train, y_train, X_val, y_val, cat_features
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   🧹 GPU память очищена")

    # Сохранение результатов
    print(f"\n💾 Сохранение результатов...")
    append_trial_history(params, score, trial_number)
    save_best_params(params, score, trial_number)

    # Итог
    print(f"\n{'=' * 70}")
    print(f"✅ TRIAL {trial_number} ЗАВЕРШЁН")
    print(f"{'=' * 70}")
    print(f"📊 AUC: {score:.4f}")
    print(f"🏆 Лучший AUC: {max(best_score, score):.4f}")
    print(f"\n📋 ЗАПУСТИ СКРИПТ СНОВА для следующего trials!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()