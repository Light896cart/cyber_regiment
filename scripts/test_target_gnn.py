# =============================================================================
# scripts/test_target_gnn.py
# ИЗОЛИРОВАННЫЙ ТЕСТ: TabNet + Target GNN + Meta-Features
# 🔥 ВЕРСИЯ: Максимальная производительность (CatIdxs + GNN + Meta)
# =============================================================================

import sys
import gc
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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

from src.data.loader import DataLoader
from models.tabnet_gnn_model import TabNetTargetGNN  # 🔥 ИМПОРТ ГОТОВОЙ МОДЕЛИ
from utils.meta_features import MetaFeaturesGenerator
from utils.feature_selector import select_features_catboost


# =============================================================================
# 🔥 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def convert_cat_features_to_indices(
        feature_names: List[str],
        cat_feature_names: List[str]
) -> Tuple[List[List[int]], List[int]]:
    """
    Конвертирует имена категориальных признаков в формат для TabNet.
    TabNet требует: cat_idxs = [[0], [5], ...], cat_emb_dim = [2, 2, ...]
    """
    cat_idxs = []
    cat_emb_dim = []

    for i, name in enumerate(feature_names):
        if name in cat_feature_names:
            cat_idxs.append([i])  # TabNet требует список списков
            cat_emb_dim.append(2)  # Размерность эмбеддинга (можно подобрать)

    return cat_idxs, cat_emb_dim


# =============================================================================
# 🔥 МЕНЕДЖЕР ОБУЧЕНИЯ
# =============================================================================

class TabNetGNNTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.best_auc = 0.0
        self.selected_features: Optional[List[str]] = None
        self.meta_generator = None
        self.final_feature_names: Optional[List[str]] = None
        self.cat_idxs_used = False  # Флаг: удалось ли использовать cat_idxs

    def train(
            self,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            X_val: pd.DataFrame,
            y_val: pd.DataFrame,
            corr_matrix: np.ndarray,
            cat_features: list,
            n_targets: int = 41,
            n_select_features: int = 750,
            use_meta_features: bool = True,
            epochs: int = 30,
            batch_size: int = 2048,
            verbose: bool = True
    ):
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"🚀 ОБУЧЕНИЕ TabNet + TARGET GNN")
            print(f"{'=' * 70}")
            print(f"   📊 Train: {X_train.shape[0]} строк, {X_train.shape[1]} признаков")
            print(f"   📊 Val: {X_val.shape[0]} строк")
            print(f"   🎯 Таргетов: {n_targets}")
            print(f"   🧠 Device: {self.device}")
            print(f"   🔮 Meta-Features: {'✅' if use_meta_features else '❌'}")

        # ============================================
        # 🔥 1. ГЕНЕРАЦИЯ МЕТА-ПРИЗНАКОВ
        # ============================================
        df_meta_train = None
        df_meta_val = None

        if use_meta_features:
            print(f"\n🔧 Генерация мета-признаков...")
            self.meta_generator = MetaFeaturesGenerator(
                artifacts_dir=str(ROOT_DIR / "artifacts")
            )
            self.meta_generator.load_correlation_matrix()

            oof_path = ROOT_DIR / "artifacts" / "oof_predictions_STACKED_stage1.parquet"
            if oof_path.exists():
                df_oof_stacked = pd.read_parquet(oof_path)
                print(f"   📊 Загружен STACKED OOF: {df_oof_stacked.shape}")

                df_oof = pd.DataFrame()
                for target in self.meta_generator.target_cols:
                    stacked_col = f"catboost_{target}"
                    if stacked_col in df_oof_stacked.columns:
                        df_oof[target] = df_oof_stacked[stacked_col].values

                if len(df_oof.columns) == 0:
                    raise ValueError("Не удалось извлечь таргеты из STACKED OOF!")

                print(f"   ✅ Извлечено {len(df_oof.columns)} таргетов для catboost")

                n_train, n_val = len(X_train), len(X_val)
                df_oof_train = df_oof.iloc[:n_train].reset_index(drop=True)
                df_oof_val = df_oof.iloc[n_train:n_train + n_val].reset_index(drop=True)

                print(f"   📊 OOF Train: {df_oof_train.shape}")
                print(f"   📊 OOF Val: {df_oof_val.shape}")

                df_meta_train = self.meta_generator.generate_from_dataframe(df_oof_train, n_best_corr=15)
                df_meta_val = self.meta_generator.generate_from_dataframe(df_oof_val, n_best_corr=15)

                print(f"   ✅ Мета-признаков: {df_meta_train.shape[1]}")
            else:
                print(f"   ⚠️  OOF файл не найден, пропускаем мета-признаки")
                use_meta_features = False

        # ============================================
        # 🔥 2. ОБЪЕДИНЕНИЕ + FEATURE SELECTION
        # ============================================
        X_train_combined = X_train.copy()
        X_val_combined = X_val.copy()

        if use_meta_features and df_meta_train is not None:
            X_train_combined = pd.concat([
                X_train_combined.reset_index(drop=True),
                df_meta_train.reset_index(drop=True)
            ], axis=1)
            X_val_combined = pd.concat([
                X_val_combined.reset_index(drop=True),
                df_meta_val.reset_index(drop=True)
            ], axis=1)

        print(f"   📊 Всего признаков (до FS): {X_train_combined.shape[1]}")

        print(f"\n🔍 Feature Selection (топ-{n_select_features})...")
        self.selected_features, importance_df = select_features_catboost(
            X=pl.from_pandas(X_train_combined),
            y=pl.from_pandas(y_train),
            cat_features=cat_features,
            n_select=n_select_features,
            verbose=verbose
        )

        X_train_selected = X_train_combined[self.selected_features]
        X_val_selected = X_val_combined[self.selected_features]

        print(f"   ✅ После Feature Selection: {X_train_selected.shape[1]} признаков")

        # 🔥 Сохраняем финальные имена признаков
        self.final_feature_names = list(X_train_selected.columns)

        # ============================================
        # 🔥 3. ПОДГОТОВКА CAT_IDXS ДЛЯ TABNET
        # ============================================
        # Фильтруем cat_features: оставляем только те, что попали в отбор
        cat_features_filtered = [f for f in cat_features if f in self.final_feature_names]

        cat_idxs = None
        cat_emb_dim = None

        if cat_features_filtered:
            try:
                cat_idxs, cat_emb_dim = convert_cat_features_to_indices(
                    self.final_feature_names,
                    cat_features_filtered
                )
                print(f"   🏷️  Cat features для TabNet: {len(cat_idxs)} (индексы подготовлены)")
            except Exception as e:
                print(f"   ⚠️  Ошибка подготовки cat_idxs: {e}. Будут использованы как continuous.")
                cat_idxs = None
                cat_emb_dim = None
        else:
            print(f"   ℹ️  Категориальных признаков после FS не осталось")

        # ============================================
        # 4. СТАНДАРТИЗАЦИЯ
        # ============================================
        mean_ = X_train_selected.values.mean(axis=0, dtype=np.float32)
        std_ = X_train_selected.values.std(axis=0, dtype=np.float32) + 1e-8

        X_train_scaled = ((X_train_selected.values - mean_) / std_).astype(np.float32)
        X_val_scaled = ((X_val_selected.values - mean_) / std_).astype(np.float32)

        y_train_np = y_train.values.astype(np.float32)
        y_val_np = y_val.values.astype(np.float32)

        # 🔥 Проверка на NaN/Inf
        assert not np.any(np.isnan(X_train_scaled)), "NaN в признаках!"
        assert not np.any(np.isinf(X_train_scaled)), "Inf в признаках!"

        # ============================================
        # 5. DATALOADER
        # ============================================
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_scaled).to(self.device),
            torch.from_numpy(y_train_np).to(self.device)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_scaled).to(self.device),
            torch.from_numpy(y_val_np).to(self.device)
        )

        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0
        )

        # ============================================
        # 🔥 6. МОДЕЛЬ (🔥 ИСПРАВЛЕНО: CatIdxs + Fallback)
        # ============================================
        print(f"\n🧠 Инициализация TabNetTargetGNN...")

        tabnet_params = {
            'n_d': 64,
            'n_a': 64,
            'n_steps': 3,
            'gamma': 1.5,
            'virtual_batch_size': 256,
        }

        # Добавляем cat_idxs если есть
        if cat_idxs is not None and cat_emb_dim is not None:
            tabnet_params['cat_idxs'] = cat_idxs
            tabnet_params['cat_emb_dim'] = cat_emb_dim
            # ❌ cat_emb_usage НЕ ДОБАВЛЯЕМ — его нет в TabNet
            print(f"   ✅ TabNet настроен с категориальными эмбеддингами")
        else:
            print(f"   ⚠️  TabNet будет работать без категориальных эмбеддингов (continuous)")

        # 🔥 ПОПЫТКА ИНИЦИАЛИЗАЦИИ С ЗАЩИТОЙ
        try:
            self.model = TabNetTargetGNN(
                input_dim=X_train_scaled.shape[1],
                n_targets=n_targets,
                tabnet_params=tabnet_params,
                gnn_params={
                    'hidden_dim': 128,
                    'n_layers': 3,
                    'dropout': 0.2,
                    'corr_threshold': 0.3  # 🔥 КРИТИЧНО: было 0.0
                },
                device=str(self.device)
            ).to(self.device)
            self.cat_idxs_used = (cat_idxs is not None)

        except AttributeError as e:
            # 🔥 FALLBACK: Если TabNet падает на cat_idxs (баг библиотеки)
            if "'list' object has no attribute 'to'" in str(e):
                print(f"\n   ❌ Ошибка TabNet с cat_idxs: {e}")
                print(f"   🔄 ПЕРЕЗАПУСК БЕЗ cat_idxs (continuous mode)...")

                # Убираем категориальные параметры
                tabnet_params.pop('cat_idxs', None)
                tabnet_params.pop('cat_emb_dim', None)
                tabnet_params.pop('cat_emb_usage', None)

                self.model = TabNetTargetGNN(
                    input_dim=X_train_scaled.shape[1],
                    n_targets=n_targets,
                    tabnet_params=tabnet_params,
                    gnn_params={
                        'hidden_dim': 128,
                        'n_layers': 3,
                        'dropout': 0.2,
                        'corr_threshold': 0.3
                    },
                    device=str(self.device)
                ).to(self.device)
                self.cat_idxs_used = False
                print(f"   ✅ Модель инициализирована в режиме continuous")
            else:
                raise

        self.model.set_feature_names(self.final_feature_names)
        self.model.set_correlation_matrix(corr_matrix, threshold=0.3)

        # ============================================
        # 7. ОБУЧЕНИЕ
        # ============================================
        pos_ratio = y_train_np.mean(axis=0)
        pos_weights = torch.tensor((1 - pos_ratio) / (pos_ratio + 1e-6), dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_model_state = None
        patience_counter = 0
        patience_limit = self.config.get('patience', 30)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            # Валидация
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(
                    torch.from_numpy(X_val_scaled).to(self.device)
                )
                val_preds = torch.sigmoid(val_outputs).cpu().numpy()

            valid_cols = [i for i in range(n_targets) if len(np.unique(y_val_np[:, i])) > 1]
            val_auc = roc_auc_score(y_val_np[:, valid_cols], val_preds[:, valid_cols], average='macro')
            scheduler.step(val_auc)

            if val_auc > self.best_auc:
                self.best_auc = val_auc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 5 == 0:
                print(
                    f"   [Epoch {epoch + 1:3d}] Loss: {total_loss / len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Best: {self.best_auc:.4f}")

            if patience_counter >= patience_limit:
                if verbose:
                    print(f"   ⏹️ Early Stop at epoch {epoch + 1}")
                break

            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        if verbose:
            print(f"\n   ✅ Обучение завершено | Best Val AUC: {self.best_auc:.4f}")
            if self.cat_idxs_used:
                print(f"   🏷️  Использованы категориальные эмбеддинги")
            else:
                print(f"   ℹ️  Категориальные признаки обработаны как continuous")

        return self.best_auc, val_preds


# =============================================================================
# 🔥 MAIN
# =============================================================================

def main():
    print(f"\n{'=' * 70}")
    print(f"🧪 ИЗОЛИРОВАННЫЙ ТЕСТ: TabNet + TARGET GNN + META-FEATURES")
    print(f"{'=' * 70}")

    loader = DataLoader(cat_strategy="int")
    loader.load_full_data()

    X_full, y_full = loader.get_full_data()
    X_np = X_full.to_numpy().astype(np.float32)
    y_np = y_full.to_numpy().astype(np.float32)
    n_targets = y_np.shape[1]
    cat_features = loader.cat_features

    print(f"   📊 Всего данных: {X_np.shape[0]} строк, {X_np.shape[1]} признаков")
    print(f"   🎯 Таргетов: {n_targets}")

    # Корреляционная матрица
    oof_path = ROOT_DIR / "artifacts" / "oof_predictions_STACKED_stage1.parquet"
    if oof_path.exists():
        df_oof = pd.read_parquet(oof_path)
        target_cols = [col for col in df_oof.columns if col.startswith('catboost_target_')]
        df_oof_catboost = df_oof[target_cols].copy()
        df_oof_catboost.columns = [col.replace('catboost_', '') for col in df_oof_catboost.columns]
        corr_matrix = df_oof_catboost.corr().values
        print(
            f"   ✅ Корреляционная матрица: {corr_matrix.shape}, min={corr_matrix.min():.4f}, max={corr_matrix.max():.4f}")
    else:
        corr_path = ROOT_DIR / "artifacts" / "corr_matrix_stage1.json"
        with open(corr_path, 'r') as f:
            corr_data = json.load(f)
        corr_matrix = np.array(corr_data['matrix'])

    # Train/Val split
    X_full_pd = X_full.to_pandas()
    y_full_pd = y_full.to_pandas()
    y_sum = y_full_pd.sum(axis=1)
    X_train_pd, X_val_pd, y_train_pd, y_val_pd, _, _ = train_test_split(
        X_full_pd, y_full_pd, y_sum, test_size=0.1, random_state=42, stratify=y_sum
    )

    print(f"   📊 Train: {X_train_pd.shape[0]} строк")
    print(f"   📊 Val: {X_val_pd.shape[0]} строк")

    # Конфиг
    config = {
        'gnn': {
            'hidden_dim': 128,
            'n_layers': 3,
            'dropout': 0.2,
            'corr_threshold': 0.3  # 🔥 КРИТИЧНО: не 0.0!
        },
        'dropout': 0.3,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'patience': 30
    }

    # Обучение
    trainer = TabNetGNNTrainer(config)

    best_auc, val_preds = trainer.train(
        X_train=X_train_pd,
        y_train=y_train_pd,
        X_val=X_val_pd,
        y_val=y_val_pd,
        corr_matrix=corr_matrix,
        cat_features=cat_features,
        n_targets=n_targets,
        n_select_features=750,
        use_meta_features=True,
        epochs=30,
        batch_size=2048,
        verbose=True
    )

    # Сравнение
    print(f"\n{'=' * 70}")
    print(f"📊 СРАВНЕНИЕ С BASELINE")
    print(f"{'=' * 70}")
    stage1_auc = 0.8166
    print(f"   📈 Stage 1 CatBoost CV AUC: {stage1_auc:.4f}")
    print(f"   📈 TabNet + Target GNN Val AUC: {best_auc:.4f}")
    print(f"   📊 Разница: {best_auc - stage1_auc:+.4f}")

    if best_auc > stage1_auc + 0.003:
        print(f"   ✅ ПРИРОСТ ЗНАЧИМЫЙ! (> 0.003) → Стоит внедрять")
    elif best_auc > stage1_auc:
        print(f"   ⚠️  Прирост есть, но маленький (< 0.003)")
    else:
        print(f"   ❌ Прироста нет → Не стоит внедрять")

    # Сохранение
    results = {
        'gnn_auc': float(best_auc),
        'stage1_baseline_auc': stage1_auc,
        'improvement': float(best_auc - stage1_auc),
        'config': config,
        'n_train': len(X_train_pd),
        'n_val': len(X_val_pd),
        'n_features_original': X_np.shape[1],
        'n_features_selected': len(trainer.selected_features) if trainer.selected_features else 0,
        'selected_features': trainer.selected_features if trainer.selected_features else [],
        'n_targets': n_targets,
        'used_meta_features': trainer.meta_generator is not None,
        'used_cat_idxs': trainer.cat_idxs_used,
        'final_feature_names': trainer.final_feature_names if trainer.final_feature_names else []
    }

    results_path = ROOT_DIR / "artifacts" / "test_target_gnn_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n   💾 Результаты сохранены: {results_path}")
    print(f"{'=' * 70}")
    print(f"✅ ТЕСТ ЗАВЕРШЁН")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()