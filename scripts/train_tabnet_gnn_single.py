# =============================================================================
# scripts/train_tabnet_gnn_single.py
# Stage 2: Обучение TabNet + Target GNN (ОДИН ЗАПУСК)
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
CONFIG_PATH = ROOT_DIR / "configs" / "tabnet_gnn" / "tabnet_gnn_config.json"
RESULTS_DIR = ARTIFACTS_DIR / "tabnet_gnn_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 🔥 Файлы для сохранения
BEST_MODEL_PATH = RESULTS_DIR / "best_model_metadata.json"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"


# =============================================================================
# 2. ЗАГРУЗКА ДАННЫХ (точно как в Stage 2 validation)
# =============================================================================

def load_stage2_data(sample_ratio=1.0, n_select_features=1500):
    """
    Загружает данные для Stage 2 обучения.
    🔥 Повторяет логику из 03_stage2_validation.py
    """
    print(f"\n📂 Загрузка данных Stage 2...")

    X_TRAIN_PATH = ARTIFACTS_DIR / "X_train_extended_val.parquet"
    Y_TRAIN_PATH = ARTIFACTS_DIR / "y_train_val.parquet"
    X_VAL_PATH = ARTIFACTS_DIR / "X_val_extended_val.parquet"
    Y_VAL_PATH = ARTIFACTS_DIR / "y_val_val.parquet"
    FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance_stage2.csv"
    CORR_MATRIX_PATH = ARTIFACTS_DIR / "corr_matrix_stage1.json"

    # Загрузка parquet
    X_train_pd = pd.read_parquet(X_TRAIN_PATH)
    y_train_pd = pd.read_parquet(Y_TRAIN_PATH)
    X_val_pd = pd.read_parquet(X_VAL_PATH)
    y_val_pd = pd.read_parquet(Y_VAL_PATH)

    print(f"   📊 Исходные: Train {X_train_pd.shape}, Val {X_val_pd.shape}")

    # 🔥 Feature Selection (если есть)
    cat_features_filtered = []
    if FEATURE_IMPORTANCE_PATH.exists() and n_select_features > 0:
        imp_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        selected_features = imp_df.head(n_select_features)['feature'].tolist()

        # 🔥 Важно: не удаляем мета-признаки (они начинаются с meta_)
        meta_cols = [c for c in X_train_pd.columns if c.startswith('meta_')]
        feature_cols = [c for c in selected_features if c in X_train_pd.columns and not c.startswith('meta_')]
        all_cols = list(set(feature_cols + meta_cols))

        X_train_pd = X_train_pd[all_cols]
        X_val_pd = X_val_pd[all_cols]

        # Загружаем cat_features из DataLoader
        try:
            from src.data.loader import DataLoader
            loader = DataLoader(cat_strategy="int")
            loader.load_full_data()
            all_cat = getattr(loader, 'cat_features', [])
            cat_features_filtered = [f for f in all_cat if f in all_cols]
        except Exception as e:
            print(f"   ⚠️  Не удалось загрузить cat_features: {e}")
            cat_features_filtered = []

        print(f"   ✅ После Feature Selection: {len(all_cols)} признаков ({len(meta_cols)} мета)")

    # 🔥 Сэмплирование (для быстрых тестов)
    if sample_ratio < 1.0:
        n_samples = int(len(X_train_pd) * sample_ratio)
        np.random.seed()  # Случайный seed
        sample_idx = np.random.choice(len(X_train_pd), n_samples, replace=False)
        X_train_pd = X_train_pd.iloc[sample_idx]
        y_train_pd = y_train_pd.iloc[sample_idx]
        X_val_pd = X_val_pd.iloc[:int(len(X_val_pd) * sample_ratio)]
        y_val_pd = y_val_pd.iloc[:int(len(y_val_pd) * sample_ratio)]
        print(f"   📊 Сэмплирование: {sample_ratio * 100:.0f}%")
        del sample_idx

    # Конвертация в Polars
    X_train_pl = pl.from_pandas(X_train_pd)
    y_train_pl = pl.from_pandas(y_train_pd)
    X_val_pl = pl.from_pandas(X_val_pd)
    y_val_pl = pl.from_pandas(y_val_pd)

    # Загрузка корреляционной матрицы для GNN
    corr_matrix = None
    if CORR_MATRIX_PATH.exists():
        with open(CORR_MATRIX_PATH, 'r') as f:
            corr_data = json.load(f)
        corr_matrix = np.array(corr_data['matrix'])
        print(f"   ✅ Corr matrix: {corr_matrix.shape}")

    # Очистка
    del X_train_pd, y_train_pd, X_val_pd, y_val_pd
    gc.collect()

    print(f"   ✅ Train: {X_train_pl.shape}, Val: {X_val_pl.shape}")
    print(f"   🏷️  Cat features: {len(cat_features_filtered)}")

    return X_train_pl, y_train_pl, X_val_pl, y_val_pl, cat_features_filtered, corr_matrix


# =============================================================================
# 3. ГЕНЕРАЦИЯ КОНФИГА ДЛЯ МОДЕЛИ
# =============================================================================

def generate_tabnet_config(trial_number=0, use_best=False, best_config=None):
    """
    Генерирует конфигурацию для TabNetTargetGNN.
    🔥 Без Optuna — просто фиксированные/случайные параметры
    """

    # Базовые параметры (фиксированные для стабильности)
    base_config = {
        'model_params': {
            'tabnet': {
                'n_d': 64,
                'n_a': 64,
                'n_steps': 5,
                'gamma': 1.5,
                'virtual_batch_size': 256,
                'momentum': 0.02,
                'mask_type': 'sparsemax',
            },
            'gnn': {
                'hidden_dim': 128,
                'n_layers': 3,
                'dropout': 0.2,
                'corr_threshold': 0.3,  # 🔥 Критично: не 0.0!
            },
            # Фиксированные
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 2048,
            'epochs': 30,
        },
        'training': {
            'patience': 15,
            'random_seed': 42 + trial_number,
            'use_amp': torch.cuda.is_available(),  # Авто-детект AMP
        }
    }

    # 🔥 Если есть лучшие параметры — используем их с небольшим шумом
    if use_best and best_config:
        print(f"   🎯 Используем лучшие параметры + шум")
        # Можно добавить вариации вокруг лучших значений
        base_config['model_params']['learning_rate'] = best_config.get('learning_rate', 0.001)
        base_config['model_params']['tabnet']['n_d'] = best_config.get('n_d', 64)
    else:
        print(f"   🔍 Используем базовые параметры")

    # 🔥 Случайные вариации для исследования (если trial_number > 0)
    if trial_number > 0 and not use_best:
        # Легкий рандом для исследования пространства
        base_config['model_params']['tabnet']['n_d'] = np.random.choice([32, 64, 128])
        base_config['model_params']['gnn']['hidden_dim'] = np.random.choice([64, 128, 256])
        base_config['model_params']['learning_rate'] = 10 ** np.random.uniform(-3.5, -2.5)
        print(f"   🎲 Вариация: n_d={base_config['model_params']['tabnet']['n_d']}, "
              f"lr={base_config['model_params']['learning_rate']:.4f}")

    return base_config


# =============================================================================
# 4. ЗАГРУЗКА/СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

def load_best_config():
    """Загружает лучшую конфигурацию и историю."""
    best_config = None
    best_score = 0
    history = []

    if BEST_MODEL_PATH.exists():
        with open(BEST_MODEL_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        best_config = data.get('model_params', {}).get('tabnet', {})
        best_config.update(data.get('model_params', {}).get('gnn', {}))
        best_config.update(data.get('model_params', {}))
        best_score = data.get('best_score', 0)
        print(f"   ✅ Лучший AUC: {best_score:.4f}")

    if TRAINING_HISTORY_PATH.exists():
        with open(TRAINING_HISTORY_PATH, 'r', encoding='utf-8') as f:
            history = json.load(f)
        print(f"   📊 Найдено {len(history)} запусков в истории")

    return best_config, best_score, history


def save_results(params, score, trial_number, metadata=None):
    """Сохраняет результаты запуска."""
    # Обновляем лучший результат
    current_best = 0
    if BEST_MODEL_PATH.exists():
        with open(BEST_MODEL_PATH, 'r') as f:
            current_best = json.load(f).get('best_score', 0)

    is_new_best = score > current_best

    if is_new_best:
        result = {
            'best_params': params,
            'best_score': score,
            'trial_number': trial_number,
            'datetime': datetime.datetime.now().isoformat(),
            'model_path': str(RESULTS_DIR / f"model_trial_{trial_number}"),
        }
        if metadata:
            result.update(metadata)
        with open(BEST_MODEL_PATH, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"   💾 НОВЫЙ РЕКОРД! AUC: {score:.4f} (было: {current_best:.4f})")

    # Добавляем в историю
    history = []
    if TRAINING_HISTORY_PATH.exists():
        with open(TRAINING_HISTORY_PATH, 'r') as f:
            history = json.load(f)

    history.append({
        'trial_number': trial_number,
        'score': score,
        'params': params,
        'datetime': datetime.datetime.now().isoformat(),
        'is_new_best': is_new_best
    })

    with open(TRAINING_HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return is_new_best


# =============================================================================
# 5. ОСНОВНОЙ ЗАПУСК
# =============================================================================

def main():
    print(f"\n{'=' * 70}")
    print(f"🚀 TABNET + TARGET GNN: SINGLE TRAINING RUN")
    print(f"{'=' * 70}")
    print(f"🔥 Каждый запуск = новый процесс = 100% очистка GPU")
    print(f"📊 Данные: Stage 2 extended features")
    print(f"🧠 Модель: TabNet encoder + Target GNN refinement")
    print(f"💾 Результаты: {RESULTS_DIR}")
    print(f"{'=' * 70}\n")

    # 🔥 Проверка GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        free_mem = torch.cuda.mem_get_info()[0] / 1024 ** 3
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} ГБ всего, {free_mem:.1f} ГБ свободно)")
    else:
        print(f"⚠️  GPU не обнаружен, обучение на CPU (медленно!)")

    # 🔥 Загрузка лучших параметров + история
    best_config, best_score, history = load_best_config()
    trial_number = len(history) + 1
    print(f"   🔢 Запуск #{trial_number}")

    # 🔥 Генерация конфига
    config = generate_tabnet_config(
        trial_number=trial_number,
        use_best=(trial_number > 5),  # После 5 запусков используем лучшие
        best_config=best_config
    )

    print(f"\n⚙️  Конфигурация модели:")
    tabnet_p = config['model_params']['tabnet']
    gnn_p = config['model_params']['gnn']
    print(f"   📊 TabNet: n_d={tabnet_p['n_d']}, n_steps={tabnet_p['n_steps']}")
    print(
        f"   🕸️  GNN: hidden={gnn_p['hidden_dim']}, layers={gnn_p['n_layers']}, corr_thresh={gnn_p['corr_threshold']}")
    print(f"   🎯 LR: {config['model_params']['learning_rate']}, Epochs: {config['model_params']['epochs']}")
    print(f"   🔥 AMP: {config['training']['use_amp']}")

    # 🔥 Загрузка данных
    X_train, y_train, X_val, y_val, cat_features, corr_matrix = load_stage2_data(
        sample_ratio=0.8,  # 1.0 = все данные, 0.1 = 10% для теста
        n_select_features=1500
    )

    target_cols = list(y_train.columns)
    n_targets = len(target_cols)
    print(f"   🎯 Таргетов: {n_targets}")

    # 🔥 Инициализация менеджера и обучение
    print(f"\n🚀 Обучение модели...")
    from models.tabnet_gnn_model import TabNetGNNManager

    manager = TabNetGNNManager(
        config_path=str(CONFIG_PATH) if CONFIG_PATH.exists() else None,
        save_dir=str(RESULTS_DIR),
        fold_folder=f"tabnet_trial_{trial_number}"
    )

    # Обновляем параметры из нашего конфига
    if manager.model_params:
        manager.model_params.update(config['model_params'])
    if manager.training_config:
        manager.training_config.update(config['training'])

    score = 0.0
    predictions = {}

    try:
        predictions, score = manager.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cat_features=cat_features,
            version_name=f"tabnet_trial_{trial_number}",
            save_model=True,
            verbose=True,
            corr_matrix=corr_matrix  # 🔥 ПЕРЕДАЁМ НАПРЯМУЮ
        )
        print(f"   ✅ Обучение завершено | Val AUC: {score:.4f}")

    except Exception as e:
        print(f"   ❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()
        score = 0.0

    finally:
        # 🔥 Агрессивная очистка памяти
        manager.clear()
        del manager

        del X_train, y_train, X_val, y_val, cat_features, corr_matrix
        gc.collect()

        if device == 'cuda':
            torch.cuda.empty_cache()
            print(f"   🧹 GPU память очищена")

    # 🔥 Сохранение результатов
    print(f"\n💾 Сохранение результатов...")
    metadata = {
        'n_train': len(predictions[list(predictions.keys())[0]]) if predictions else 0,
        'n_targets': n_targets,
        'device': device,
        'corr_threshold': config['model_params']['gnn']['corr_threshold'],
    }

    is_new_best = save_results(config['model_params'], score, trial_number, metadata)

    # 🔥 Финальный отчёт
    print(f"\n{'=' * 70}")
    print(f"✅ ЗАПУСК #{trial_number} ЗАВЕРШЁН")
    print(f"{'=' * 70}")
    print(f"📊 Val Macro ROC-AUC: {score:.4f}")
    print(f"🏆 Лучший результат: {max(best_score, score):.4f}")
    print(f"🆕 Новый рекорд: {'✅ ДА' if is_new_best else '❌ нет'}")
    print(f"\n📋 Чтобы запустить следующий запуск — просто выполни скрипт снова!")
    print(f"{'=' * 70}\n")

    return score


if __name__ == "__main__":
    main()