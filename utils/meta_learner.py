# =============================================================================
# utils/meta_learner.py
# Meta-Learner (Stacking) для ансамбля моделей
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
import json
from pathlib import Path


class MetaLearnerEnsemble:
    """
    Meta-Learner (Stacking) для объединения предсказаний моделей.

    Использование:
        meta = MetaLearnerEnsemble()
        meta.fit(y_val, predictions_dict)
        weights = meta.get_weights()
        blended = meta.predict(predictions_dict)
    """

    def __init__(
            self,
            meta_model_type: str = 'ridge',  # 'ridge', 'logistic', 'gbm'
            alpha: float = 1.0,
            non_negative: bool = True  # Ограничить веса >= 0
    ):
        self.meta_model_type = meta_model_type
        self.alpha = alpha
        self.non_negative = non_negative

        self.meta_models: Dict[str, object] = {}
        self.target_cols: List[str] = []
        self.model_names: List[str] = []
        self.weights_dict: Dict[str, float] = {}
        self.is_fitted = False

    def fit(
            self,
            y_true: np.ndarray,
            predictions_dict: Dict[str, Dict[str, np.ndarray]],
            target_cols: List[str]
    ) -> None:
        """
        Обучает meta-learner на валидационных данных.

        Args:
            y_true: Истинные таргеты (n_samples, n_targets)
            predictions_dict: {model_name: {target_col: predictions}}
            target_cols: Список таргетов
        """
        self.target_cols = target_cols
        self.model_names = list(predictions_dict.keys())
        n_samples = y_true.shape[0]
        n_targets = len(target_cols)
        n_models = len(self.model_names)

        print(f"\n📚 Обучение Meta-Learner ({self.meta_model_type})...")
        print(f"   📊 Samples: {n_samples}, Targets: {n_targets}, Models: {n_models}")

        # Создаём мета-признаки: для каждого таргета — предсказания всех моделей
        for t_idx, target_col in enumerate(target_cols):
            # X_meta: (n_samples, n_models) — предсказания всех моделей для этого таргета
            X_meta = np.column_stack([
                predictions_dict[model_name][target_col]
                for model_name in self.model_names
            ])

            # y_true для этого таргета
            y_target = y_true[:, t_idx]

            # Выбираем модель
            if self.meta_model_type == 'ridge':
                meta_model = Ridge(alpha=self.alpha, positive=self.non_negative)
            elif self.meta_model_type == 'logistic':
                meta_model = LogisticRegression(
                    C=1.0 / self.alpha,
                    solver='lbfgs',
                    max_iter=1000
                )
            elif self.meta_model_type == 'gbm':
                meta_model = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=3,
                    random_state=42
                )
            else:
                raise ValueError(f"Неизвестный тип meta_model: {self.meta_model_type}")

            # Обучаем
            meta_model.fit(X_meta, y_target)
            self.meta_models[target_col] = meta_model

            # Извлекаем веса
            if hasattr(meta_model, 'coef_'):
                weights = meta_model.coef_
            else:
                weights = np.ones(n_models) / n_models

            # Нормализуем веса
            weights = np.abs(weights)
            weights = weights / weights.sum()

            for i, model_name in enumerate(self.model_names):
                if model_name not in self.weights_dict:
                    self.weights_dict[model_name] = 0.0
                self.weights_dict[model_name] += weights[i] / n_targets

        # Нормализуем итоговые веса
        total = sum(self.weights_dict.values())
        self.weights_dict = {k: v / total for k, v in self.weights_dict.items()}

        self.is_fitted = True
        print(f"   ✅ Meta-Learner обучен")
        print(f"   📊 Веса моделей: {self.weights_dict}")

    def predict(
            self,
            predictions_dict: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Делает предсказания используя обученные meta-модели.

        Args:
            predictions_dict: {model_name: {target_col: predictions}}

        Returns:
            blended_predictions: {target_col: blended_predictions}
        """
        if not self.is_fitted:
            raise ValueError("Сначала обучите meta-learner!")

        blended = {}

        for target_col in self.target_cols:
            if target_col not in self.meta_models:
                continue

            meta_model = self.meta_models[target_col]

            # Создаём мета-признаки
            X_meta = np.column_stack([
                predictions_dict[model_name][target_col]
                for model_name in self.model_names
            ])

            # Предсказание
            blended[target_col] = meta_model.predict(X_meta)

        return blended

    def get_weights(self) -> Dict[str, float]:
        """Возвращает веса моделей."""
        return self.weights_dict.copy()

    def save(self, path: str) -> None:
        """Сохраняет meta-learner."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'meta_models': self.meta_models,
                'weights_dict': self.weights_dict,
                'target_cols': self.target_cols,
                'model_names': self.model_names,
                'meta_model_type': self.meta_model_type
            }, f)
        print(f"   💾 Meta-Learner сохранён: {path}")

    @classmethod
    def load(cls, path: str) -> 'MetaLearnerEnsemble':
        """Загружает meta-learner."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(meta_model_type=data['meta_model_type'])
        instance.meta_models = data['meta_models']
        instance.weights_dict = data['weights_dict']
        instance.target_cols = data['target_cols']
        instance.model_names = data['model_names']
        instance.is_fitted = True

        print(f"   ✅ Meta-Learner загружен: {path}")
        return instance


def compare_ensemble_methods(
        y_true: np.ndarray,
        predictions_dict: Dict[str, Dict[str, np.ndarray]],
        weighted_weights: Dict[str, float],
        meta_model_type: str = 'ridge'
) -> Tuple[Dict, Dict, str]:
    """
    Сравнивает Weighted Average и Meta-Learner, выбирает лучший.

    Returns:
        best_method: 'weighted' или 'meta'
        best_predictions: Предсказания лучшего метода
        comparison_results: Результаты сравнения
    """
    target_cols = list(list(predictions_dict.values())[0].keys())

    # =========================================
    # 1. Weighted Average
    # =========================================
    print(f"\n{'=' * 60}")
    print(f"📊 СРАВНЕНИЕ МЕТОДОВ АНСАМБЛИРОВАНИЯ")
    print(f"{'=' * 60}\n")

    weighted_preds = {}
    for col in target_cols:
        weighted_preds[col] = np.average(
            [predictions_dict[model][col] for model in predictions_dict.keys()],
            axis=0,
            weights=[weighted_weights.get(model, 1.0) for model in predictions_dict.keys()]
        )

    weighted_auc = roc_auc_score(y_true, np.column_stack([weighted_preds[col] for col in target_cols]), average='macro')
    print(f"   🏆 Weighted Average AUC: {weighted_auc:.4f}")

    # =========================================
    # 2. Meta-Learner
    # =========================================
    meta_learner = MetaLearnerEnsemble(meta_model_type=meta_model_type)
    meta_learner.fit(y_true, predictions_dict, target_cols)
    meta_preds = meta_learner.predict(predictions_dict)

    meta_auc = roc_auc_score(y_true, np.column_stack([meta_preds[col] for col in target_cols]), average='macro')
    print(f"   🧠 Meta-Learner AUC: {meta_auc:.4f}")

    # =========================================
    # 3. Сравнение
    # =========================================
    improvement = meta_auc - weighted_auc

    print(f"\n   📈 Прирост Meta-Learner: {improvement:+.4f}")

    if improvement > 0.001:
        best_method = 'meta'
        best_predictions = meta_preds
        print(f"   ✅ Meta-Learner лучше на {improvement:.4f}")
    else:
        best_method = 'weighted'
        best_predictions = weighted_preds
        print(f"   ✅ Weighted Average лучше (или разница < 0.001)")

    comparison_results = {
        'weighted_auc': weighted_auc,
        'meta_auc': meta_auc,
        'improvement': improvement,
        'best_method': best_method,
        'weighted_weights': weighted_weights,
        'meta_weights': meta_learner.get_weights(),
        'meta_model_type': meta_model_type
    }

    return best_method, best_predictions, comparison_results