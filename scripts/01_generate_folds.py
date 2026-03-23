import polars as pl
import numpy as np
import json
import os
from pathlib import Path
from utils.fold_generator import get_cv_splits

# Пути
DATA_DIR = r"D:\Code\hackaton_cyberpolka_CV\data"
FOLDS_ROOT = r"D:\Code\hackaton_cyberpolka_CV\folds"

TRAIN_PATH = os.path.join(DATA_DIR, "train_final.parquet")
TARGET_PATH = os.path.join(DATA_DIR, "train_target.parquet")


def generate_and_save_folds(
        n_splits: int = 3,
        random_state: int = 42,
        force_regenerate: bool = False
):
    """
    Генерирует фолды и сохраняет данные на диск.
    Создаёт папку folds_5/ (или folds_3/) с файлами для каждого фолда.
    """
    fold_dir = os.path.join(FOLDS_ROOT, f"folds_{n_splits}")
    meta_path = os.path.join(fold_dir, "fold_metadata.json")

    # ✅ Проверка: если фолды уже есть, не генерируем заново
    if os.path.exists(meta_path) and not force_regenerate:
        print(f"✅ Фолды уже существуют в {fold_dir}")
        print(f"   Чтобы перегенерировать, установи force_regenerate=True")
        return

    # 1. Загрузка данных
    print(f"🔄 Загрузка данных для генерации {n_splits} фолдов...")
    df_train = pl.read_parquet(TRAIN_PATH)
    df_target = pl.read_parquet(TARGET_PATH)
    df = df_train.join(df_target, on="customer_id", how="inner")

    customer_ids = df["customer_id"].to_numpy()
    y_np = df_target.join(df_train.select("customer_id"), on="customer_id", how="inner")
    y_np = y_np.select([col for col in y_np.columns if col.startswith('target_')]).to_numpy()

    target_cols = [col for col in df.columns if col.startswith('target_')]
    cat_features = [col for col in df.columns if col.startswith('cat_')]

    print(f"   📊 Всего данных: {len(customer_ids)} строк")

    os.makedirs(fold_dir, exist_ok=True)

    folds_info = []
    for fold_idx, (train_idx, val_idx) in enumerate(get_cv_splits(y_np, n_splits=n_splits, random_state=random_state)):
        print(f"   🔁 Фолд {fold_idx + 1}/{n_splits}: train={len(train_idx)}, val={len(val_idx)}")

        # ✅ СОХРАНЯЕМ ИНДЕКСЫ СТРОК (для OOF)
        np.save(os.path.join(fold_dir, f"fold_{fold_idx}_train_idx.npy"), train_idx)
        np.save(os.path.join(fold_dir, f"fold_{fold_idx}_val_idx.npy"), val_idx)

        # ✅ СОХРАНЯЕМ CUSTOMER_ID (для сабмита)
        np.save(os.path.join(fold_dir, f"fold_{fold_idx}_train_ids.npy"), customer_ids[train_idx])
        np.save(os.path.join(fold_dir, f"fold_{fold_idx}_val_ids.npy"), customer_ids[val_idx])

        folds_info.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx))
        })

    metadata = {
        "n_splits": n_splits,
        "random_state": random_state,
        "n_samples": int(len(customer_ids)),
        "target_cols": target_cols,
        "cat_features": cat_features,
        "data_path": TRAIN_PATH,
        "target_path": TARGET_PATH,
        "folds": folds_info
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Индексы сохранены в {fold_dir}")
    print(f"   📁 Файлы: fold_0_train_ids.npy, fold_0_val_ids.npy, ...")
    print(f"   📄 Данные: {TRAIN_PATH} (не дублируются)")


if __name__ == "__main__":
    generate_and_save_folds(n_splits=3, random_state=42, force_regenerate=True)  # ← True чтобы перегенерировать