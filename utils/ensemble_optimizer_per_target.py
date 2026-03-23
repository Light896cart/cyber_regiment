# =============================================================================
# utils/ensemble_optimizer_per_target.py
# Оптимизатор весов для ансамбля — ОТДЕЛЬНО ДЛЯ КАЖДОГО ТАРГЕТА
# 🔥 ТОЛЬКО differential_evolution (никакого minimize/L-BFGS-B)
# =============================================================================
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional
from scipy.optimize import differential_evolution
import json
from pathlib import Path


class EnsembleWeightOptimizerPerTarget:
    """
    🔥 Оптимизирует веса ансамбля отдельно для каждого таргета.
    Использует ТОЛЬКО differential_evolution (эволюционный алгоритм).
    """

    def __init__(self, y_true: np.ndarray, target_cols: List[str]):
        self.y_true = y_true
        self.target_cols = target_cols
        self.n_targets = len(target_cols)
        self.n_models = 0
        self.predictions_dict = {}
        self.target_weights = {}  # 🔥 {target_col: {model_name: weight}}

        # Кэшируем матрицы предсказаний для скорости
        self._model_matrices = {}

    def add_model_predictions(self, model_name: str, predictions: Dict[str, np.ndarray]):
        """Добавляет предсказания модели"""
        if set(predictions.keys()) != set(self.target_cols):
            raise ValueError(
                f"Несовпадающие таргеты у модели '{model_name}'. "
                f"Ожидалось: {self.target_cols}, Получено: {list(predictions.keys())}"
            )

        self.predictions_dict[model_name] = predictions
        self._model_matrices[model_name] = np.column_stack([
            predictions[col] for col in self.target_cols
        ])
        self.n_models = len(self.predictions_dict)

    def _optimize_weights_for_target(
            self,
            y_target: np.ndarray,
            target_col: str,
            n_iterations: int = 70,
            verbose: bool = False
    ) -> Dict[str, float]:
        """
        🔥 Оптимизирует веса для ОДНОГО таргета через differential_evolution.
        """
        model_names = list(self.predictions_dict.keys())
        n_models = len(model_names)

        def loss(weights):
            """Loss функция: отрицательный AUC (минимизируем)"""
            weights = np.abs(weights)
            weights = weights / weights.sum()

            # Взвешенное усреднение для этого таргета
            blended = np.zeros(len(y_target))
            for model_idx, model_name in enumerate(model_names):
                if model_name in self.predictions_dict and target_col in self.predictions_dict[model_name]:
                    blended += weights[model_idx] * self.predictions_dict[model_name][target_col]

            # AUC loss (отрицательный, т.к. минимизируем)
            if len(np.unique(y_target)) > 1:
                return -roc_auc_score(y_target, blended)
            return 0.5

        # 🔥 Differential Evolution (ТОЛЬКО этот метод!)
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        result = differential_evolution(
            loss,
            bounds=bounds,
            maxiter=n_iterations,
            seed=42,
            disp=verbose,
            polish=True,
            workers=1,
            updating='deferred',
        )

        # Нормализуем веса
        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / optimal_weights.sum()

        return {
            model_names[i]: float(optimal_weights[i])
            for i in range(n_models)
        }

    def optimize_weights(
            self,
            n_iterations: int = 70,
            verbose: bool = True,
            n_jobs: int = 1  # ⚠️ differential_evolution не поддерживает параллелизм по таргетам
    ) -> Dict[str, Dict[str, float]]:
        """
        🔥 Оптимизирует веса для КАЖДОГО таргета отдельно.
        """
        if self.n_models < 2:
            raise ValueError("Нужно минимум 2 модели для оптимизации")

        if verbose:
            print(f"\n🔍 Per-Target оптимизация для {self.n_targets} таргетов...")
            print(f"   Метод: differential_evolution | Итераций: {n_iterations}")
            print(f"   Всего итераций: {self.n_targets * n_iterations:,}")

        # 🔥 Цикл по каждому таргету
        for t_idx, target_col in enumerate(self.target_cols):
            if verbose and (t_idx + 1) % 10 == 0:
                print(f"   🔄 Таргет {t_idx + 1}/{self.n_targets}: {target_col}")

            y_target = self.y_true[:, t_idx]

            # Проверка на валидность (если только один класс — равные веса)
            if len(np.unique(y_target)) < 2:
                self.target_weights[target_col] = {
                    m: 1.0 / self.n_models for m in self.predictions_dict.keys()
                }
                continue

            # 🔥 Оптимизация для этого таргета
            weights = self._optimize_weights_for_target(
                y_target,
                target_col,
                n_iterations=n_iterations,
                verbose=(verbose and t_idx < 3)  # Подробно только первые 3 таргета
            )
            self.target_weights[target_col] = weights

        if verbose:
            print(f"\n✅ Per-Target оптимизация завершена")
            self._print_summary()

        return self.target_weights

    def _print_summary(self):
        """Выводит сводку по весам"""
        print(f"\n📊 Средние веса по всем таргетам:")
        avg_weights = {m: 0.0 for m in self.predictions_dict.keys()}
        for target_col, weights in self.target_weights.items():
            for model_name, weight in weights.items():
                avg_weights[model_name] += weight
        for model_name in avg_weights:
            avg_weights[model_name] /= self.n_targets
            print(f"   {model_name}: {avg_weights[model_name]:.4f}")

        print(f"\n🏆 Где модели доминируют (вес > 0.5):")
        for model_name in self.predictions_dict.keys():
            dominant_targets = [
                target_col
                for target_col, weights in self.target_weights.items()
                if weights[model_name] > 0.5
            ]
            if dominant_targets:
                print(f"   {model_name}: {len(dominant_targets)} таргетов")

    def get_blended_predictions(self) -> Dict[str, np.ndarray]:
        """Возвращает финальные предсказания с per-target весами"""
        if not self.target_weights:
            raise ValueError("Сначала вызовите optimize_weights()")

        blended = {}
        for target_col in self.target_cols:
            weights = self.target_weights[target_col]
            blended[target_col] = np.zeros(
                len(list(self.predictions_dict.values())[0][target_col])
            )
            for model_name, weight in weights.items():
                blended[target_col] += weight * self.predictions_dict[model_name][target_col]

        return blended

    def evaluate(self, y_true: np.ndarray) -> float:
        """Оценивает Macro ROC-AUC с per-target весами"""
        blended = self.get_blended_predictions()
        blended_matrix = np.column_stack([blended[col] for col in self.target_cols])
        valid_cols = [i for i in range(self.n_targets) if len(np.unique(y_true[:, i])) > 1]
        if len(valid_cols) > 0:
            return roc_auc_score(y_true[:, valid_cols], blended_matrix[:, valid_cols], average='macro')
        return 0.5

    def save(self, path: str, metadata: Optional[Dict] = None):
        """Сохраняет веса в JSON"""
        data = {
            'method': 'per_target_differential_evolution',
            'target_weights': self.target_weights,
            'target_cols': self.target_cols,
            'model_names': list(self.predictions_dict.keys()),
            'metadata': metadata or {}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"   💾 Веса сохранены: {path}")

    @classmethod
    def load(cls, path: str) -> 'EnsembleWeightOptimizerPerTarget':
        """Загружает веса из JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instance = cls(data['target_cols'], data['model_names'])
        instance.target_weights = data['target_weights']
        instance.n_models = len(data['model_names'])

        print(f"   ✅ Веса загружены: {path}")
        return instance