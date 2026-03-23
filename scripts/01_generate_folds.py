# =============================================================================
# scripts/01_generate_folds.py
# 🔧 ИСПРАВЛЕНО: Универсальные пути для команды и GitHub
# =============================================================================

import polars as pl
import numpy as np
import json
import os
from pathlib import Path
from utils.fold_generator import get_cv_splits


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
# 🔧 ПУТИ (ИСПРАВЛЕНО!)
# =============================================================================

# 🔥 Пути относительно корня проекта (работают везде!)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FOLDS_ROOT = PROJECT_ROOT / "folds"

# 🔥 Можно переопределить через env variables
DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
FOLDS_ROOT = Path(os.getenv("FOLDS_ROOT", DEFAULT_FOLDS_ROOT))

# Входные файлы
TRAIN_PATH = DATA_DIR / "train_final.parquet"
TARGET_PATH = DATA_DIR / "train_target.parquet"


# =============================================================================
# 🔧 ФУНКЦИЯ ГЕНЕРАЦИИ ФОЛДОВ (ИСПРАВЛЕНО!)
# =============================================================================

def generate_and_save_folds(
        n_splits: int = 3,
        random_state: int = 42,
        force_regenerate: bool = False
):
    """
    Генерирует фолды и сохраняет данные на диск.
    Создаёт папку folds_5/ (или folds_3/) с файлами для каждого фолда.
    """
    fold_dir = FOLDS_ROOT / f"folds_{n_splits}"
    meta_path = fold_dir / "fold_metadata.json"

    # ✅ Проверка: если фолды уже есть, не генерируем заново
    if meta_path.exists() and not force_regenerate:
        print(f"✅ Фолды уже существуют в {fold_dir}")
        print(f"   Чтобы перегенерировать, установи force_regenerate=True")
        return

    # =====================================================================
    # 1. 🔧 ЗАГРУЗКА ДАННЫХ (С ПРОВЕРКОЙ СУЩЕСТВОВАНИЯ!)
    # =====================================================================
    print(f"🔄 Загрузка данных для генерации {n_splits} фолдов...")

    # 🔥 Проверка существования файлов
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"❌ Train данные не найдены: {TRAIN_PATH}\n"
            f"💡 Запусти сначала: python scripts/01_full_data_processing.py"
        )

    if not TARGET_PATH.exists():
        raise FileNotFoundError(
            f"❌ Target данные не найдены: {TARGET_PATH}"
        )

    df_train = pl.read_parquet(TRAIN_PATH)
    df_target = pl.read_parquet(TARGET_PATH)
    df = df_train.join(df_target, on="customer_id", how="inner")

    customer_ids = df["customer_id"].to_numpy()
    y_np = df_target.join(df_train.select("customer_id"), on="customer_id", how="inner")
    y_np = y_np.select([col for col in y_np.columns if col.startswith('target_')]).to_numpy()

    target_cols = [col for col in df.columns if col.startswith('target_')]
    cat_features = [col for col in df.columns if col.startswith('cat_')]

    print(f"   📊 Всего данных: {len(customer_ids)} строк")

    # =====================================================================
    # 2. СОЗДАНИЕ ДИРЕКТОРИИ
    # =====================================================================
    fold_dir.mkdir(parents=True, exist_ok=True)

    folds_info = []
    for fold_idx, (train_idx, val_idx) in enumerate(get_cv_splits(y_np, n_splits=n_splits, random_state=random_state)):
        print(f"   🔁 Фолд {fold_idx + 1}/{n_splits}: train={len(train_idx)}, val={len(val_idx)}")

        # ✅ СОХРАНЯЕМ ИНДЕКСЫ СТРОК (для OOF)
        # 🔥 Используем Path вместо os.path.join
        np.save(fold_dir / f"fold_{fold_idx}_train_idx.npy", train_idx)
        np.save(fold_dir / f"fold_{fold_idx}_val_idx.npy", val_idx)

        # ✅ СОХРАНЯЕМ CUSTOMER_ID (для сабмита)
        np.save(fold_dir / f"fold_{fold_idx}_train_ids.npy", customer_ids[train_idx])
        np.save(fold_dir / f"fold_{fold_idx}_val_ids.npy", customer_ids[val_idx])

        folds_info.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx))
        })

    # =====================================================================
    # 3. 🔧 СОХРАНЕНИЕ МЕТАДАННЫХ (С ПУТЯМИ ОТНОСИТЕЛЬНО КОРНЯ!)
    # =====================================================================
    metadata = {
        "n_splits": n_splits,
        "random_state": random_state,
        "n_samples": int(len(customer_ids)),
        "target_cols": target_cols,
        "cat_features": cat_features,
        # 🔥 Сохраняем пути относительно корня (не абсолютные!)
        "data_path": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
        "target_path": str(TARGET_PATH.relative_to(PROJECT_ROOT)),
        "project_root": str(PROJECT_ROOT),
        "folds": folds_info
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Индексы сохранены в {fold_dir}")
    print(f"   📁 Файлы: fold_0_train_ids.npy, fold_0_val_ids.npy, ...")
    print(f"   📄 Данные: {TRAIN_PATH} (не дублируются)")


# =============================================================================
# 🔧 ЗАПУСК (ИСПРАВЛЕНО!)
# =============================================================================

if __name__ == "__main__":
    print(f"🔍 Проект: {PROJECT_ROOT}")
    print(f"📁 Data: {DATA_DIR}")
    print(f"📁 Folds: {FOLDS_ROOT}")
    print(f"{'=' * 60}\n")

    generate_and_save_folds(n_splits=3, random_state=42, force_regenerate=True)