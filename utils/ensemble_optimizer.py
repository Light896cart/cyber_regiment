# =============================================================================
# utils/ensemble_optimizer.py
# Оптимизатор весов для ансамбля моделей
# =============================================================================

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize, differential_evolution
import json
from pathlib import Path


class EnsembleWeightOptimizer:
    """
    Оптимизатор весов для ансамбля моделей.
    ⚡ ОПТИМИЗИРОВАННАЯ ВЕРСИЯ (в 10-50 раз быстрее)
    """

    def __init__(self, y_true: np.ndarray, target_cols: List[str]):
        self.y_true = y_true
        self.target_cols = target_cols
        self.n_targets = len(target_cols)
        self.n_models = 0
        self.predictions_dict = {}

        # ✅ ПРЕДВАРИТЕЛЬНЫЕ ВЫЧИСЛЕНИЯ (делаются 1 раз!)
        self._model_matrices = {}
        self._valid_cols = np.array([
            i for i in range(self.n_targets)
            if len(np.unique(self.y_true[:, i])) > 1
        ])
        self._y_true_valid = self.y_true[:, self._valid_cols] if len(self._valid_cols) > 0 else self.y_true

    def add_model_predictions(self, model_name: str, predictions: Dict[str, np.ndarray]):
        """Добавляет предсказания модели + кэширует матрицу"""
        if set(predictions.keys()) != set(self.target_cols):
            raise ValueError(
                f"Модель '{model_name}' имеет несовпадающие таргеты. "
                f"Ожидалось: {self.target_cols}, Получено: {list(predictions.keys())}"
            )

        self.predictions_dict[model_name] = predictions
        self._model_matrices[model_name] = np.column_stack([
            predictions[col] for col in self.target_cols
        ])
        self.n_models = len(self.predictions_dict)

    def _calculate_macro_auc(self, weights: np.ndarray) -> float:
        """⚡ Считает Macro ROC-AUC (ВЕКТОРИЗОВАННО)"""
        weights = np.abs(weights)
        weights = weights / weights.sum()

        all_preds = np.stack([
            self._model_matrices[name] for name in self.predictions_dict.keys()
        ])
        blended_preds = np.sum(all_preds * weights[:, np.newaxis, np.newaxis], axis=0)

        if len(self._valid_cols) > 0:
            blended_valid = blended_preds[:, self._valid_cols]
            auc = roc_auc_score(self._y_true_valid, blended_valid, average='macro')
        else:
            auc = 0.5

        return auc

    def _loss_function(self, weights: np.ndarray) -> float:
        """Функция потерь (минимизируем отрицательный AUC)"""
        return -self._calculate_macro_auc(weights)

    def optimize_weights(
            self,
            method: str = 'differential_evolution',
            n_iterations: int = 30,
            verbose: bool = True,
            n_jobs: int = 1
    ) -> Dict[str, float]:
        """Подбирает оптимальные веса."""
        if self.n_models < 2:
            raise ValueError("Нужно минимум 2 модели для оптимизации весов")

        model_names = list(self.predictions_dict.keys())

        if verbose:
            print(f"\n🔍 Оптимизация весов для {self.n_models} моделей: {model_names}")
            print(f"   Метод: {method} | Итераций: {n_iterations} | Ядер: {n_jobs}")

        bounds = [(0.0, 1.0) for _ in range(self.n_models)]

        if method == 'differential_evolution':
            result = differential_evolution(
                self._loss_function,
                bounds=bounds,
                maxiter=n_iterations,
                seed=42,
                disp=verbose,
                polish=True,
                workers=n_jobs,
                updating='deferred',
            )
            optimal_weights = result.x
        else:
            x0 = np.ones(self.n_models) / self.n_models
            result = minimize(
                self._loss_function,
                x0=x0,
                method='Nelder-Mead',
                options={'maxiter': n_iterations}
            )
            optimal_weights = result.x

        optimal_weights = np.abs(optimal_weights)
        optimal_weights = optimal_weights / optimal_weights.sum()

        weights_dict = {model_names[i]: optimal_weights[i] for i in range(self.n_models)}
        final_auc = self._calculate_macro_auc(optimal_weights)

        if verbose:
            print(f"\n✅ Оптимальные веса:")
            for model_name, weight in weights_dict.items():
                print(f"   {model_name}: {weight:.4f}")
            print(f"\n📊 Итоговый Macro ROC-AUC: {final_auc:.4f}")

            equal_weights = np.ones(self.n_models) / self.n_models
            equal_auc = self._calculate_macro_auc(equal_weights)
            print(f"   (Для сравнения: равные веса дают AUC = {equal_auc:.4f})")
            print(f"   Прирост от оптимизации: {final_auc - equal_auc:.4f}")

        return weights_dict

    def get_blended_predictions(
            self,
            weights: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """⚡ Возвращает финальные предсказания (ВЕКТОРИЗОВАННО)"""
        all_preds = np.stack([
            self._model_matrices[name] for name in self.predictions_dict.keys()
        ])
        weight_array = np.array([weights[name] for name in self.predictions_dict.keys()])
        blended = np.sum(all_preds * weight_array[:, np.newaxis, np.newaxis], axis=0)

        return {
            col: blended[:, i] for i, col in enumerate(self.target_cols)
        }

    def save_weights(self, path: str, metadata: Optional[Dict] = None):
        """Сохраняет веса в JSON."""
        model_names = list(self.predictions_dict.keys())
        optimal_weights = self.optimize_weights(verbose=False)

        data = {
            'model_names': model_names,
            'weights': {name: float(optimal_weights[name]) for name in model_names},
            'target_cols': self.target_cols,
            'metadata': metadata or {}
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"   💾 Веса сохранены: {path}")

    @staticmethod
    def load_weights(path: str) -> Dict:
        """Загружает веса из JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def optimize_ensemble_weights(
        y_true: np.ndarray,
        predictions_list: List[Dict[str, np.ndarray]],
        model_names: List[str],
        target_cols: List[str],
        verbose: bool = True,
        n_iterations: int = 30,
        n_jobs: int = 1
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], float]:
    """Удобная функция-обертка для оптимизации весов."""
    if len(predictions_list) != len(model_names):
        raise ValueError("Количество предсказаний должно совпадать с количеством имен моделей")

    optimizer = EnsembleWeightOptimizer(y_true, target_cols)

    for model_name, preds in zip(model_names, predictions_list):
        optimizer.add_model_predictions(model_name, preds)

    weights = optimizer.optimize_weights(
        method='differential_evolution',
        verbose=verbose,
        n_iterations=n_iterations,
        n_jobs=n_jobs
    )

    blended = optimizer.get_blended_predictions(weights)
    blended_matrix = np.column_stack([blended[col] for col in target_cols])
    final_auc = roc_auc_score(y_true, blended_matrix, average='macro')

    return weights, blended, final_auc