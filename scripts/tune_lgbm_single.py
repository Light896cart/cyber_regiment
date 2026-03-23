# =============================================================================
# scripts/tune_lgbm_single.py
# Stage 2: ОДИН trials за запуск + сохранение лучшего в JSON (LightGBM)
# 🔥 КАЖДЫЙ ЗАПУСК = НОВЫЙ ПРОЦЕСС = 100% очистка памяти
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

# LightGBM не использует torch, но оставим проверку для консистентности среды
try:
    import torch

    HAS_TORCH = True
except:
    HAS_TORCH = False

import lightgbm as lgb

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
# 🔥 Путь к конфигам LightGBM
CONFIG_PATH = ROOT_DIR / "configs" / "lightgbm" / "lgbm_config.yaml"
OPTUNA_RESULTS_DIR = ARTIFACTS_DIR / "optuna_results_lgbm"
OPTUNA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 🔥 Файл для хранения ЛУЧШИХ результатов
BEST_PARAMS_PATH = OPTUNA_RESULTS_DIR / "best_params.json"
TRIALS_HISTORY_PATH = OPTUNA_RESULTS_DIR / "trials_history.json"


# =============================================================================
# 2. ЗАГРУЗКА ДАННЫХ (Как в CatBoost скрипте)
# =============================================================================

def load_stage2_data(sample_ratio=0.05):
    """Загружает данные с сэмплированием."""
    print(f"\n📂 Загрузка данных...")

    X_TRAIN_PATH = ARTIFACTS_DIR / "X_train_extended_val.parquet"
    Y_TRAIN_PATH = ARTIFACTS_DIR / "y_train_val.parquet"
    X_VAL_PATH = ARTIFACTS_DIR / "X_val_extended_val.parquet"
    Y_VAL_PATH = ARTIFACTS_DIR / "y_val_val.parquet"
    FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance_stage2.csv"

    # Проверка существования файлов
    if not X_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {X_TRAIN_PATH}")

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

    # Конвертация в Polars (для совместимости с менеджером)
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
# 3. ГЕНЕРАЦИЯ ПАРАМЕТРОВ (АДАПТИРОВАНО ПОД LGBM)
# =============================================================================

def suggest_param_from_history(history, param_name, default_range, best_params=None):
    """
    🔥 УМНЫЙ подбор параметра на основе истории
    """
    if len(history) < 5 or best_params is None:
        # Мало данных — случайный выбор
        if isinstance(default_range, tuple):
            if param_name in ['learning_rate', 'lambda_l1', 'lambda_l2']:
                return 10 ** np.random.uniform(np.log10(default_range[0]), np.log10(default_range[1]))
            else:
                return np.random.uniform(default_range[0], default_range[1])
        else:
            return np.random.randint(default_range[0], default_range[1])

    # Берём лучшие параметры + небольшой шум
    base_value = best_params.get(param_name, np.mean(default_range))

    # Разный шум для разных параметров
    if param_name in ['num_leaves', 'max_depth']:
        noise = np.random.randint(-3, 4)  # ±1
        return int(np.clip(base_value + noise, default_range[0], default_range[1]))
    elif param_name == 'min_data_in_leaf':
        # 🔥 ±15 вместо ±1 (очень широкий поиск)
        noise = np.random.randint(-15, 16)
        return int(np.clip(base_value + noise, default_range[0], default_range[1]))
    elif param_name in ['learning_rate', 'lambda_l1', 'lambda_l2']:
        noise = 10 ** np.random.uniform(-0.5, 0.5)
        return float(np.clip(base_value * noise, default_range[0], default_range[1]))
    elif param_name in ['feature_fraction', 'bagging_fraction']:
        noise = np.random.uniform(-0.2, 0.2)
        return float(np.clip(base_value + noise, default_range[0], default_range[1]))
    elif param_name == 'max_bin':
        # GPU требует специфичные значения
        gpu_bins = [63, 127, 255]
        if best_params and param_name in best_params:
            # Выбираем ближайшее GPU-совместимое значение
            base = best_params[param_name]
            return min(gpu_bins, key=lambda x: abs(x - base))
        else:
            return np.random.choice(gpu_bins)
    else:
        return base_value

def convert_to_serializable(obj):
    """Конвертирует numpy типы в стандартные Python типы для JSON."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def append_trial_history(params, score, trial_number):
    """Добавляет trial в историю."""
    history = []
    if TRIALS_HISTORY_PATH.exists():
        try:
            with open(TRIALS_HISTORY_PATH, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []

    # 🔥 КОНВЕРТАЦИЯ numpy типов перед сохранением
    params_clean = convert_to_serializable(params)

    history.append({
        'trial_number': int(trial_number),
        'score': float(score),
        'params': params_clean,
        'datetime': datetime.datetime.now().isoformat()
    })

    with open(TRIALS_HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def generate_params(best_params=None, trial_number=0, history=None, best_score=0):
    """
    Генерирует новые параметры для LightGBM.
    """
    history = history or []

    # 🔥 Логика переключения: Исследование vs Эксплуатация
    if best_params and trial_number > 5:
        print(f"   🎯 Эксплуатация (вариация вокруг лучших)")
        params = {
            'device': 'gpu',  # Будет переопределено в main если нет GPU
            'verbose': 10,
            'seed': 42 + trial_number,
            'feature_pre_filter': False,
            # 🔥 ФИКСИРОВАНО: много итераций + early stopping
            'n_estimators': 450,
            'early_stopping_round': 100,

            # 🔥 НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ
            'num_leaves': suggest_param_from_history(history, 'num_leaves', (31, 127), best_params),
            'max_depth': suggest_param_from_history(history, 'max_depth', (6, 12), best_params),
            'learning_rate': suggest_param_from_history(history, 'learning_rate', (0.01, 0.3), best_params),
            'min_data_in_leaf': suggest_param_from_history(history, 'min_data_in_leaf', (5, 100), best_params),

            # Регуляризация и сэмплирование
            'feature_fraction': suggest_param_from_history(history, 'feature_fraction', (0.6, 1.0), best_params),
            'bagging_fraction': suggest_param_from_history(history, 'bagging_fraction', (0.6, 1.0), best_params),
            'bagging_freq': 1,  # Включаем bagging
            'lambda_l1': suggest_param_from_history(history, 'lambda_l1', (0.0, 10.0), best_params),
            'lambda_l2': suggest_param_from_history(history, 'lambda_l2', (0.0, 10.0), best_params),
            'max_bin': suggest_param_from_history(history, 'max_bin', (255, 511), best_params),
        }
    else:
        print(f"   🔍 Исследование (случайный поиск)")
        params = {
            'device': 'gpu',
            'verbose': 10,
            'seed': 42 + trial_number,

            'n_estimators': 450,
            'early_stopping_round': 100,

            # 🔥 ВСЕ ПАРАМЕТРЫ СЛУЧАЙНЫЕ
            'feature_pre_filter': False,
            'num_leaves': np.random.randint(31, 127),
            'max_depth': np.random.randint(6, 12),
            'learning_rate': 10 ** np.random.uniform(-2, -0.5),  # 0.01 to 0.3
            'min_data_in_leaf': np.random.randint(5, 100),

            'feature_fraction': np.random.uniform(0.6, 1.0),
            'bagging_fraction': np.random.uniform(0.6, 1.0),
            'bagging_freq': 1,
            'lambda_l1': 10 ** np.random.uniform(-3, 1),
            'lambda_l2': 10 ** np.random.uniform(-3, 1),
            'max_bin': np.random.choice([63, 127, 255]),
        }

    return params


# =============================================================================
# 4. ЗАГРУЗКА/СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

def load_best_params():
    """Загружает лучшие параметры и считает trials из истории."""
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
        # 🔥 КОНВЕРТАЦИЯ numpy типов
        params_clean = convert_to_serializable(params)
        data = {
            'best_params': params_clean,
            'best_score': float(score),
            'trial_number': int(trial_number),
            'datetime': datetime.datetime.now().isoformat(),
            'improved': improved,
            'note': 'n_estimators=3000, early_stopping=100'
        }
        with open(BEST_PARAMS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"   💾 НОВЫЙ РЕКОРД! AUC: {score:.4f} (было: {current_best:.4f})")
        return True
    else:
        print(f"   ⏭️  Не лучше текущего рекорда ({current_best:.4f})")
        return False



# =============================================================================
# 5. ОСНОВНОЙ ЗАПУСК (ОДИН TRIALS)
# =============================================================================

def main():
    print(f"\n{'=' * 70}")
    print(f"🔮 LIGHTGBM SINGLE TRIAL (GPU/CPU)")
    print(f"{'=' * 70}")
    print(f"🚀 Каждый запуск = новый процесс = 100% очистка памяти")
    print(f"📊 n_estimators: 3000 (фиксировано, early stopping остановит раньше)")
    print(f"🔧 Настраиваемых параметров: 10+")
    print(f"💾 Лучшее сохраняется в: {BEST_PARAMS_PATH}")
    print(f"{'=' * 70}\n")

    # Проверка GPU (LightGBM использует свой драйвер, не torch)
    use_gpu = False
    try:
        # Проверяем наличие GPU через torch (как в окружении пользователя)
        if HAS_TORCH and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"✅ GPU обнаружен: {gpu_name} ({gpu_memory:.1f} ГБ)")
            use_gpu = True
        else:
            print("⚠️  GPU не обнаружен (или torch не установлен), будет CPU")
    except:
        print("⚠️  Не удалось проверить GPU, будет CPU")

    # Загрузка лучших параметров + ИСТОРИЯ
    best_params, best_score, last_trial, history = load_best_params()
    trial_number = last_trial + 1

    print(f"   🔢 Trial: {trial_number}")

    # Генерация параметров
    params = generate_params(best_params, trial_number, history, best_score)

    # Принудительно ставим CPU если GPU нет
    if not use_gpu:
        params['device'] = 'cpu'
        params['n_jobs'] = -1
    else:
        params['device'] = 'gpu'
        # Важно для стабильности GPU в LGBM
        params['gpu_use_dp'] = True
        params['force_col_wise'] = True

    print(f"\n⚙️  Параметры:")
    for k, v in params.items():
        if k not in ['device', 'verbose', 'seed', 'early_stopping_round', 'n_jobs']:
            print(f"   {k}: {v}")

    # Загрузка данных
    # 🔥 sample_ratio=0.5 для баланса скорости и качества тюнинга
    X_train, y_train, X_val, y_val, cat_features = load_stage2_data(sample_ratio=0.5)

    # Обучение
    print(f"\n🚀 Обучение...")
    from models.lgbm_model import LGBMManager

    manager = LGBMManager(
        config_path=str(CONFIG_PATH),
        save_dir=str(ROOT_DIR / "models_weight"),
        fold_folder="optuna_trials_lgbm"
    )

    # 🔥 Обновляем параметры менеджера перед обучением
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
            save_model=False,  # Не сохраняем веса каждого trials, только лучшие
            verbose=10,
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

        # 🔥 LightGBM не использует torch.cuda, но если в окружении есть тензоры - чистим
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"   🧹 Память очищена (gc.collect)")

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