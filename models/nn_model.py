# =============================================================================
# models/nn_model.py
# Менеджер для Neural Network с поддержкой MultiLabel классификации
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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, List, Any
import warnings
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
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "neural_network" / "nn_config.yaml"
DEFAULT_SAVE_DIR = MODELS_DIR


# ============================================
# OPTIMIZED DATASET
# ============================================
class LightweightDataset(Dataset):
    __slots__ = ['X', 'y']

    def __init__(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ============================================
# MODEL
# ============================================
class MultiLabelNN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 41,
            hidden_layers: List[int] = [256, 128, 64],
            dropout_rate: float = 0.2,
            use_batch_norm: bool = True,
            activation: str = 'relu'
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        activations = {
            'relu': nn.ReLU,
            'leaky_relu': lambda: nn.LeakyReLU(0.1),
            'elu': nn.ELU,
            'gelu': nn.GELU
        }

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activations.get(activation, nn.ReLU)())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================
# MANAGER CLASS
# ============================================
class NNManager:
    """
    Менеджер для обучения, сохранения и инференса Neural Network.

    Особенности:
    - Поддержка MultiLabel классификации (41 таргет)
    - Работа с Polars DataFrame
    - Поддержка OOF предсказаний для CV
    - Сохранение метаданных для воспроизводимости
    - Агрессивная очистка памяти
    - 🔧 Универсальные пути для Windows/Linux/Mac
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
            save_dir: Путь для сохранения моделей (переопределяет дефолт)
            fold_folder: Папка для фолдов (например, "folds_2" для Stage 1, "neural_network" для Stage 2)
        """
        # 🔥 ИСПРАВЛЕНО: Используем универсальные пути
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.save_dir = Path(save_dir) if save_dir else DEFAULT_SAVE_DIR
        self.fold_folder = fold_folder

        # Загрузка конфига
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"❌ Конфиг не найден: {self.config_path}\n"
                f"💡 Убедитесь что configs/neural_network/nn_config.yaml существует"
            )

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_params = self.config.get('model_params', {})
        self.training_config = self.config.get('training', {})

        # Состояние
        self.model: Optional[MultiLabelNN] = None
        self.mean_: Optional[np.ndarray] = None  # ✅ Единый scaler для всех признаков
        self.std_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None  # ✅ Для порядка колонок
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
            verbose: bool = True
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Обучает Neural Network модель.
        """
        if verbose:
            print(f"🚀 Начало обучения NN...")
            print(f"   📊 Train: {X_train.shape[0]} строк, {X_train.shape[1]} признаков")
            if X_val is not None:
                print(f"   📊 Val: {X_val.shape[0]} строк")

        target_cols = list(y_train.columns)
        n_samples = len(X_train)
        n_features = X_train.shape[1]

        # ✅ Сохраняем порядок колонок
        self.feature_names_ = list(X_train.columns)

        # ============================================
        # 1. Конвертация Polars → numpy (float32)
        # ============================================
        float_cols = [col for col in X_train.columns if col not in (cat_features or [])]
        X_train_f32 = X_train.with_columns([
            pl.col(col).cast(pl.Float32) for col in float_cols if col in X_train.columns
        ])

        X_train_np = X_train_f32.to_numpy().astype(np.float32, copy=False)
        y_train_np = y_train.to_numpy().astype(np.float32, copy=False)

        del X_train, y_train, X_train_f32, float_cols
        gc.collect()

        use_eval = X_val is not None and y_val is not None

        if use_eval:
            float_cols_val = [col for col in X_val.columns if col not in (cat_features or [])]
            X_val_f32 = X_val.with_columns([
                pl.col(col).cast(pl.Float32) for col in float_cols_val if col in X_val.columns
            ])
            X_val_np = X_val_f32.to_numpy().astype(np.float32, copy=False)
            y_val_np = y_val.to_numpy().astype(np.float32, copy=False)
            del X_val, y_val, X_val_f32, float_cols_val
            gc.collect()
        else:
            X_val_np, y_val_np = None, None

        # ============================================
        # 2. Стандартизация
        # ============================================
        nan_count = np.isnan(X_train_np).sum()
        if nan_count > 0:
            print(f"   ⚠️  Найдено {nan_count} NaN в X_train! Заполняем медианой...")
            col_medians = np.nanmedian(X_train_np, axis=0)
            for col in range(X_train_np.shape[1]):
                mask = np.isnan(X_train_np[:, col])
                if mask.any():
                    X_train_np[mask, col] = col_medians[col] if not np.isnan(col_medians[col]) else 0.0

        # ✅ ЕДИНАЯ СТАНДАРТИЗАЦИЯ (исправлено)
        self.mean_ = X_train_np.mean(axis=0, dtype=np.float32)
        self.std_ = X_train_np.std(axis=0, dtype=np.float32) + 1e-8

        if np.any(np.isnan(self.mean_)) or np.any(np.isnan(self.std_)):
            raise RuntimeError(f"❌ NaN в mean_ или std_! Проверь входные данные")

        X_train_scaled = (X_train_np - self.mean_) / self.std_
        X_train_scaled = X_train_scaled.astype(np.float32, copy=False)
        del X_train_np
        gc.collect()

        if use_eval:
            # ✅ Обработка NaN в валидации
            if X_val_np is not None:
                nan_count_val = np.isnan(X_val_np).sum()
                if nan_count_val > 0:
                    X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=0.0, neginf=0.0)

            X_val_scaled = (X_val_np - self.mean_) / self.std_
            X_val_scaled = X_val_scaled.astype(np.float32, copy=False)
            del X_val_np
            gc.collect()
        else:
            X_val_scaled = None

        input_dim = X_train_scaled.shape[1]

        # ============================================
        # 3. Веса для классов
        # ============================================
        device = self.training_config.get('device', 'cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        device = torch.device(device)

        pos_ratio = y_train_np.mean(axis=0)
        eps = 1e-4
        pos_ratio = np.clip(pos_ratio, eps, 1 - eps)
        raw_weights = (1 - pos_ratio) / pos_ratio

        def get_clipped_weight(ratio: float, raw_weight: float) -> float:
            if ratio < 0.001:
                return min(raw_weight, 50.0)
            elif ratio < 0.01:
                return min(raw_weight, 30.0)
            elif ratio < 0.05:
                return min(raw_weight, 20.0)
            else:
                return min(raw_weight, 10.0)

        pos_weights = np.array([get_clipped_weight(r, w) for r, w in zip(pos_ratio, raw_weights)])
        pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)

        # ============================================
        # 4. DataLoader
        # ============================================
        X_train_tensor = torch.from_numpy(X_train_scaled)

        if use_eval:
            y_train_tensor = torch.from_numpy(y_train_np.copy())
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            X_val_tensor = torch.from_numpy(X_val_scaled)
            y_val_tensor = torch.from_numpy(y_val_np.copy())
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        else:
            train_dataset = TensorDataset(X_train_tensor)
            val_dataset = None

        del X_train_scaled, y_train_np
        if use_eval:
            del X_val_scaled
        gc.collect()

        batch_size = min(self.model_params.get('batch_size', 2048), 2048 if n_samples > 500000 else 4096)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        val_loader = None
        if use_eval and val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )

        # ============================================
        # 5. Инициализация модели
        # ============================================
        torch.manual_seed(42)
        np.random.seed(42)

        self.model = MultiLabelNN(
            input_dim=input_dim,
            output_dim=len(target_cols),
            hidden_layers=self.model_params.get('hidden_layers', [256, 128, 64]),
            dropout_rate=self.model_params.get('dropout_rate', 0.2),
            use_batch_norm=self.model_params.get('use_batch_norm', True),
            activation=self.model_params.get('activation', 'relu'),
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.model_params.get('learning_rate', 0.001)),
            weight_decay=float(self.model_params.get('weight_decay', 0.00001)),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        # ============================================
        # 6. Обучение
        # ============================================
        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None
        best_epoch = 0

        epochs_to_run = self.model_params.get('epochs', 50) if use_eval else 5

        if verbose:
            print(f"\n   🧠 Обучение NN на {device} | Epochs: {epochs_to_run} | Batch: {batch_size}")

        for epoch in range(epochs_to_run):
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                if use_eval:
                    X_batch, y_batch = batch
                else:
                    X_batch = batch[0]
                    y_batch = None

                X_batch = X_batch.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = self.model(X_batch)

                if y_batch is not None:
                    y_batch = y_batch.to(device, non_blocking=True)
                    loss = criterion(outputs, y_batch)
                else:
                    loss = outputs.mean() * 0

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

                del X_batch, outputs, loss
                if y_batch is not None:
                    del y_batch

            avg_train_loss = total_loss / len(train_loader)

            if use_eval and val_loader is not None:
                self.model.eval()
                all_preds = []
                all_targets = []

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(device, non_blocking=True)
                        outputs = self.model(X_batch)
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        all_preds.append(probs)
                        all_targets.append(y_batch.numpy())

                        del X_batch, outputs, probs

                all_preds = np.vstack(all_preds)
                all_targets = np.vstack(all_targets)

                valid_cols = [i for i in range(all_targets.shape[1]) if len(np.unique(all_targets[:, i])) > 1]
                val_auc = roc_auc_score(all_targets[:, valid_cols], all_preds[:, valid_cols],
                                        average='macro') if valid_cols else 0.5

                scheduler.step(val_auc)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"   [Epoch {epoch + 1:3d}] Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}")

                if patience_counter >= self.model_params.get('patience', 30):
                    if verbose:
                        print(f"   ⏹️ Early Stop at epoch {epoch + 1}")
                    break

                del all_preds, all_targets
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"   [Epoch {epoch + 1:3d}] Loss: {avg_train_loss:.4f}")

            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            del best_model_state
            gc.collect()

        # ============================================
        # 7. Предсказания
        # ============================================
        predictions = {}
        if use_eval and val_loader is not None:
            self.model.eval()
            all_preds = []

            with torch.no_grad():
                for batch in val_loader:
                    X_batch = batch[0].to(device)
                    outputs = self.model(X_batch)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    all_preds.append(probs)
                    del X_batch, outputs, probs

            all_preds = np.vstack(all_preds)

            for i, col in enumerate(target_cols):
                predictions[col] = all_preds[:, i]

            del all_preds
            gc.collect()

        self.model.cpu()

        # ============================================
        # 8. Метаданные
        # ============================================
        safe_version = (version_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        safe_version = safe_version.replace('/', '_').replace('\\', '_').replace(':', '_')

        self.metadata = {
            "version": safe_version,
            "timestamp": datetime.datetime.now().isoformat(),
            "config_path": str(self.config_path),
            "params": self.model_params,
            "metrics": {
                "macro_roc_auc": best_val_auc,
                "best_iteration": best_epoch
            },
            "target_cols": target_cols,
            "trained_on_full_data": not use_eval,
            "input_dim": input_dim,
            "feature_names": self.feature_names_,  # ✅ Сохраняем порядок колонок
            "project_root": str(PROJECT_ROOT)
        }

        del train_loader, train_dataset
        if val_loader is not None:
            del val_loader, val_dataset
        gc.collect()

        if save_model:
            self._save_model()
            if verbose:
                print(f"✅ NN сохранена: {safe_version}, AUC: {best_val_auc:.4f}")
        else:
            if verbose:
                print(f"✅ NN обучена (не сохранена, CV режим)")

        return predictions, best_val_auc

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

        torch.save(self.model.state_dict(), model_path / 'model.pth')

        # ✅ Сохраняем ТОЛЬКО mean_ и std_ (единые)
        scaler_data = {
            'mean_': self.mean_,
            'std_': self.std_,
            'feature_names_': self.feature_names_
        }
        scaler_path = model_path / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_data, f)

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
        model_file = model_path / 'model.pth'
        scaler_file = model_path / 'scaler.pkl'

        if not meta_path.exists():
            raise FileNotFoundError(
                f"❌ metadata.json не найден: {meta_path}\n"
                f"💡 Проверьте что модель была обучена и сохранена"
            )
        if not model_file.exists():
            raise FileNotFoundError(
                f"❌ model.pth не найден: {model_file}\n"
                f"💡 Проверьте что model.pth существует"
            )
        if not scaler_file.exists():
            raise FileNotFoundError(
                f"❌ scaler.pkl не найден: {scaler_file}\n"
                f"💡 Проверьте что scaler.pkl существует"
            )

        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        input_dim = self.metadata.get('input_dim', 1553)

        output_dim = len(self.metadata.get('target_cols', []))
        if output_dim == 0:
            output_dim = self.metadata['params'].get('output_dim', 41)

        self.model = MultiLabelNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=self.metadata['params']['hidden_layers'],
            dropout_rate=self.metadata['params']['dropout_rate'],
            use_batch_norm=self.metadata['params']['use_batch_norm'],
            activation=self.metadata['params']['activation'],
        )

        state_dict = torch.load(model_file, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.cpu()
        self.model.eval()

        with open(scaler_file, 'rb') as f:
            scaler_data = pickle.load(f)

        # ✅ Загружаем ТОЛЬКО mean_ и std_ (единые)
        self.mean_ = scaler_data['mean_']
        self.std_ = scaler_data['std_']
        self.feature_names_ = scaler_data.get('feature_names_', None)

        self._is_trained = True
        print(f"✅ NN загружена: {version}")

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
        if self.model is None or self.mean_ is None or self.std_ is None:
            raise ValueError("Сначала обучите или загрузите модель!")

        device = torch.device('cpu')
        self.model.to(device)
        self.model.eval()

        if verbose:
            print(f"🔮 Инференс NN: {X.shape[0]} строк")

        # ✅ Приводим колонки к тому же порядку что при обучении
        if self.feature_names_ is not None:
            # Проверяем что все колонки на месте
            missing_cols = set(self.feature_names_) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Отсутствуют колонки: {missing_cols}")
            X = X.select(self.feature_names_)

        X_np = X.to_numpy().astype(np.float32, copy=False)

        # ✅ Обработка NaN
        nan_count = np.isnan(X_np).sum()
        if nan_count > 0:
            X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        # ✅ Та же стандартизация что при обучении
        X_scaled = (X_np - self.mean_) / self.std_
        X_scaled = X_scaled.astype(np.float32, copy=False)

        X_tensor = torch.from_numpy(X_scaled)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0)

        all_preds = []
        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0].to(device)
                outputs = self.model(X_batch)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
                del X_batch, outputs, probs

        all_preds = np.vstack(all_preds)
        target_cols = self.metadata.get('target_cols', [f"target_{i}" for i in range(all_preds.shape[1])])

        predictions = {col: all_preds[:, i] for i, col in enumerate(target_cols)}

        del X_np, X_scaled, X_tensor, all_preds
        gc.collect()

        return predictions

    # ==========================================================================
    # УТИЛИТЫ
    # ==========================================================================

    def clear(self) -> None:
        """Очищает модель из памяти."""
        if self.model is not None:
            del self.model
            self.model = None
        self._is_trained = False
        gc.collect()
        print("   🧹 NNManager очищен")

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def target_cols(self) -> List[str]:
        return self.metadata.get('target_cols', [])
