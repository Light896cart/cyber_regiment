# =============================================================================
# utils/meta_learner_per_target.py
# Meta-Learner с оптимизацией весов для КАЖДОГО таргета отдельно
# Версия 2.0: Исправлены ошибки, оптимизация через AUC
# =============================================================================

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize, differential_evolution
import json
from pathlib import Path

class MetaLearnerPerTarget:
    """
    Meta-Learner который обучает отдельные веса для каждого таргета.
    С shrinkage к глобальным весам для защиты от переобучения.

    🔧 ИСПРАВЛЕНИЯ v2.0:
    - Оптимизация через AUC (не MSE)
    - Проверка что per-target улучшает global
    - Защита для редких таргетов
    - Убран несуществующий параметр Ridge(positive=True)
    """

    def __init__(
            self,
            target_cols: List[str],
            model_names: List[str],
            shrinkage_factor: float = 0.5,
            min_improvement: float = 0.0002  # Минимальный прирост AUC для использования per-target
    ):
        self.target_cols = target_cols
        self.model_names = model_names
        self.n_targets = len(target_cols)
        self.n_models = len(model_names)
        self.shrinkage_factor = shrinkage_factor
        self.min_improvement = min_improvement

        # Веса для каждого таргета: {target_col: {model_name: weight}}
        self.target_weights: Dict[str, Dict[str, float]] = {}
        self.global_weights: Dict[str, float] = {}
        self.is_fitted = False

        # Статистика для отладки
        self.stats = {
            'targets_using_global': 0,
            'targets_using_per_target': 0,
            'avg_improvement': 0.0
        }

    def fit(
            self,
            y_true: np.ndarray,
            predictions_dict: Dict[str, Dict[str, np.ndarray]],
            verbose: bool = True
    ) -> None:
        """
        Обучает отдельные веса для каждого таргета с shrinkage.
        🔧 ИСПРАВЛЕНО: Оптимизация через AUC, не через Ridge/MSE
        """
        if verbose:
            print(f"\n📚 Обучение Meta-Learner Per-Target (v2.0)...")
            print(f"   📊 Таргетов: {self.n_targets}, Моделей: {self.n_models}")
            print(f"   🔒 Shrinkage factor: {self.shrinkage_factor} ({self.shrinkage_factor * 100:.0f}% global)")

        # =========================================
        # 1. Сначала считаем глобальные веса (для shrinkage)
        # =========================================
        if verbose:
            print(f"   🔄 Расчет глобальных весов...")

        self.global_weights = self._optimize_global_weights(y_true, predictions_dict, verbose=False)

        if verbose:
            print(f"   ✅ Глобальные веса: {self.global_weights}")

        # =========================================
        # 2. Теперь per-target веса с shrinkage
        # =========================================
        improvements = []

        for t_idx, target_col in enumerate(self.target_cols):
            if verbose and (t_idx + 1) % 10 == 0:
                print(f"   🔄 Таргет {t_idx + 1}/{self.n_targets}: {target_col}")

            # Проверяем количество positive samples
            n_positive = int(y_true[:, t_idx].sum())
            y_target = y_true[:, t_idx]

            # 🔧 ЗАЩИТА 1: Для редких таргетов используем global веса
            if n_positive < 50:
                if verbose and t_idx < 5:
                    print(f"      ⚠️  Редкий таргет ({n_positive} positive) → используем global веса")
                self.target_weights[target_col] = self.global_weights.copy()
                self.stats['targets_using_global'] += 1
                continue

            # Проверяем на валидность
            if len(np.unique(y_target)) < 2:
                self.target_weights[target_col] = self.global_weights.copy()
                self.stats['targets_using_global'] += 1
                continue

            # 🔧 ЗАЩИТА 2: Оптимизируем per-target веса через AUC
            per_target_weights = self._optimize_per_target_weights(
                y_target,
                predictions_dict,
                target_col,  # ← ✅ Убедись что передаётся
                shrinkage=self.shrinkage_factor
            )

            # 🔧 ЗАЩИТА 3: Проверяем что per-target улучшает global
            global_auc = self._calculate_target_auc(y_target, predictions_dict, self.global_weights, target_col)
            per_target_auc = self._calculate_target_auc(y_target, predictions_dict, per_target_weights, target_col)
            improvement = per_target_auc - global_auc

            improvements.append(improvement)

            if improvement > self.min_improvement:
                # Per-target лучше → используем его
                self.target_weights[target_col] = per_target_weights
                self.stats['targets_using_per_target'] += 1
                if verbose and t_idx < 5:
                    print(f"      ✅ Per-target лучше на {improvement:.4f} (AUC: {per_target_auc:.4f})")
            else:
                # Global лучше или разница незначительна → используем global
                self.target_weights[target_col] = self.global_weights.copy()
                self.stats['targets_using_global'] += 1
                if verbose and t_idx < 5:
                    print(f"      ℹ️  Global лучше (improvement: {improvement:.4f})")

        self.is_fitted = True
        self.stats['avg_improvement'] = np.mean(improvements) if improvements else 0.0

        if verbose:
            print(f"\n   ✅ Meta-Learner обучен")
            self._print_summary()

    def _optimize_global_weights(
            self,
            y_true: np.ndarray,
            predictions_dict: Dict[str, Dict[str, np.ndarray]],
            verbose: bool = False
    ) -> Dict[str, float]:
        """Оптимизирует глобальные веса через AUC."""

        def loss(weights):
            weights = np.abs(weights)
            weights = weights / weights.sum()

            # Взвешенное усреднение для всех таргетов
            all_preds = []
            for col_idx in range(self.n_targets):
                blended = np.zeros(len(y_true))
                for model_idx, model_name in enumerate(self.model_names):
                    blended += weights[model_idx] * predictions_dict[model_name][self.target_cols[col_idx]]
                all_preds.append(blended)

            all_preds = np.column_stack(all_preds)
            valid_cols = [i for i in range(self.n_targets) if len(np.unique(y_true[:, i])) > 1]

            if len(valid_cols) > 0:
                return -roc_auc_score(y_true[:, valid_cols], all_preds[:, valid_cols], average='macro')
            return 0.5

        # Оптимизация
        result = minimize(
            loss,
            np.ones(self.n_models) / self.n_models,
            method='L-BFGS-B',
            bounds=[(0.05, 0.90)] * self.n_models
        )

        optimal = np.abs(result.x)
        optimal = optimal / optimal.sum()

        return {
            self.model_names[i]: float(optimal[i])
            for i in range(self.n_models)
        }

    def _optimize_per_target_weights(
            self,
            y_target: np.ndarray,
            predictions_dict: Dict[str, Dict[str, np.ndarray]],
            target_col: str,  # ← ✅ Добавили параметр
            shrinkage: float = 0.5
    ) -> Dict[str, float]:
        """
        Оптимизирует веса для одного таргета.
        🔧 ИСПРАВЛЕНО: Shrinkage применяется только после оптимизации
        """

        def loss(weights):
            weights = np.abs(weights)
            weights = weights / weights.sum()

            # ✅ Без shrinkage внутри loss — только чистая оптимизация
            blended = np.zeros(len(y_target))
            for model_idx, model_name in enumerate(self.model_names):
                if model_name in predictions_dict and target_col in predictions_dict[model_name]:
                    blended += weights[model_idx] * predictions_dict[model_name][target_col]

            # AUC loss
            if len(np.unique(y_target)) > 1:
                return -roc_auc_score(y_target, blended)
            return 0.5

        # Оптимизация
        result = minimize(
            loss,
            np.ones(self.n_models) / self.n_models,
            method='L-BFGS-B',
            bounds=[(0.01, 1.0)] * self.n_models
        )

        optimal = np.abs(result.x)
        optimal = optimal / optimal.sum()

        # ✅ Применяем shrinkage ТОЛЬКО ЗДЕСЬ (один раз!)
        global_array = np.array([self.global_weights[m] for m in self.model_names])
        final_weights = shrinkage * global_array + (1 - shrinkage) * optimal
        final_weights = final_weights / final_weights.sum()

        return {
            self.model_names[i]: float(final_weights[i])
            for i in range(self.n_models)
        }

    def _calculate_target_auc(
            self,
            y_target: np.ndarray,
            predictions_dict: Dict[str, Dict[str, np.ndarray]],
            weights: Dict[str, float],
            target_col: str  # ← ✅ Передаём target_col явно
    ) -> float:
        """Считает AUC для одного таргета с заданными весами."""
        if len(np.unique(y_target)) < 2:
            return 0.5

        # ✅ Взвешенное усреднение
        blended = np.zeros(len(y_target))
        for model_name, weight in weights.items():
            if model_name in predictions_dict and target_col in predictions_dict[model_name]:
                blended += weight * predictions_dict[model_name][target_col]

        return roc_auc_score(y_target, blended)

    def _print_summary(self) -> None:
        """Выводит сводку по весам."""
        print(f"\n   📊 Статистика:")
        print(f"      Таргетов с global весами: {self.stats['targets_using_global']}")
        print(f"      Таргетов с per-target весами: {self.stats['targets_using_per_target']}")
        print(f"      Средний прирост AUC: {self.stats['avg_improvement']:+.4f}")

        print(f"\n   📊 Средние веса по таргетам:")
        avg_weights = {model_name: 0.0 for model_name in self.model_names}
        for target_col, weights in self.target_weights.items():
            for model_name, weight in weights.items():
                avg_weights[model_name] += weight

        for model_name in self.model_names:
            avg_weights[model_name] /= self.n_targets
            print(f"      {model_name}: {avg_weights[model_name]:.4f}")

        # Показываем где модели доминируют
        print(f"\n   🏆 Где модели доминируют (вес > 0.5):")
        for model_name in self.model_names:
            dominant_targets = [
                target_col
                for target_col, weights in self.target_weights.items()
                if weights[model_name] > 0.5
            ]
            if dominant_targets:
                print(f"      {model_name}: {len(dominant_targets)} таргетов")

    def predict(
            self,
            predictions_dict: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Делает предсказания с per-target весами."""
        if not self.is_fitted:
            raise ValueError("Сначала обучите meta-learner!")

        blended = {}

        for target_col in self.target_cols:
            weights = self.target_weights[target_col]

            # Взвешенная сумма для этого таргета
            blended[target_col] = np.zeros(
                len(list(predictions_dict.values())[0][target_col])
            )

            for model_name, weight in weights.items():
                blended[target_col] += weight * predictions_dict[model_name][target_col]

        return blended

    def evaluate(
            self,
            y_true: np.ndarray,
            predictions_dict: Dict[str, Dict[str, np.ndarray]]
    ) -> float:
        """Оценивает Macro ROC-AUC с per-target весами."""
        blended = self.predict(predictions_dict)
        blended_matrix = np.column_stack([
            blended[col] for col in self.target_cols
        ])

        valid_cols = [
            i for i in range(self.n_targets)
            if len(np.unique(y_true[:, i])) > 1
        ]

        if len(valid_cols) > 0:
            auc = roc_auc_score(y_true[:, valid_cols], blended_matrix[:, valid_cols], average='macro')
        else:
            auc = 0.5

        return auc

    def save(self, path: str) -> None:
        """Сохраняет meta-learner."""
        data = {
            'target_cols': self.target_cols,
            'model_names': self.model_names,
            'target_weights': self.target_weights,
            'global_weights': self.global_weights,
            'shrinkage_factor': self.shrinkage_factor,
            'min_improvement': self.min_improvement,
            'stats': self.stats,
            'is_fitted': self.is_fitted
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"   💾 Meta-Learner сохранён: {path}")

    @classmethod
    def load(cls, path: str) -> 'MetaLearnerPerTarget':
        """Загружает meta-learner."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instance = cls(
            target_cols=data['target_cols'],
            model_names=data['model_names'],
            shrinkage_factor=data.get('shrinkage_factor', 0.5),
            min_improvement=data.get('min_improvement', 0.0005)
        )
        instance.target_weights = data['target_weights']
        instance.global_weights = data['global_weights']
        instance.stats = data.get('stats',
                                  {'targets_using_global': 0, 'targets_using_per_target': 0, 'avg_improvement': 0.0})
        instance.is_fitted = data['is_fitted']

        print(f"   ✅ Meta-Learner загружен: {path}")
        return instance


def compare_ensemble_methods_per_target(
        y_true: np.ndarray,
        predictions_dict: Dict[str, Dict[str, np.ndarray]],
        target_cols: List[str],
        model_names: List[str],
        weighted_weights: Dict[str, float],
        shrinkage_factor: float = 0.5,
        min_improvement: float = 0.0005,
        verbose: bool = True
) -> Tuple[str, Dict[str, np.ndarray], Dict]:
    """
    Сравнивает Weighted Average vs Meta-Learner Per-Target с shrinkage.
    🔧 ИСПРАВЛЕНО: Проверка что per-target действительно улучшает
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"📊 СРАВНЕНИЕ МЕТОДОВ АНСАМБЛИРОВАНИЯ")
        print(f"{'=' * 60}\n")

    # =========================================
    # 1. Weighted Average (глобальные веса)
    # =========================================
    weighted_preds = {}
    for target_col in target_cols:
        weighted_preds[target_col] = np.average(
            [predictions_dict[model_name][target_col] for model_name in model_names],
            axis=0,
            weights=[weighted_weights.get(model_name, 1.0) for model_name in model_names]
        )

    weighted_matrix = np.column_stack([weighted_preds[col] for col in target_cols])
    valid_cols = [i for i in range(len(target_cols)) if len(np.unique(y_true[:, i])) > 1]
    weighted_auc = roc_auc_score(y_true[:, valid_cols], weighted_matrix[:, valid_cols], average='macro')

    if verbose:
        print(f"   🏆 Weighted Average AUC: {weighted_auc:.4f}")

    # =========================================
    # 2. Meta-Learner Per-Target с shrinkage
    # =========================================
    meta_learner = MetaLearnerPerTarget(
        target_cols=target_cols,
        model_names=model_names,
        shrinkage_factor=shrinkage_factor,
        min_improvement=min_improvement
    )
    meta_learner.fit(y_true, predictions_dict, verbose=False)
    meta_preds = meta_learner.predict(predictions_dict)
    meta_auc = meta_learner.evaluate(y_true, predictions_dict)

    if verbose:
        print(f"   🧠 Meta-Learner Per-Target AUC: {meta_auc:.4f}")
        print(f"   📊 Таргетов с per-target: {meta_learner.stats['targets_using_per_target']}/{len(target_cols)}")

    # =========================================
    # 3. Сравнение
    # =========================================
    improvement = meta_auc - weighted_auc

    if verbose:
        print(f"\n   📈 Прирост Meta-Learner: {improvement:+.4f}")

    # 🔧 ЗАЩИТА: Используем per-target только если реально лучше
    if improvement > min_improvement:
        best_method = 'meta_per_target'
        best_predictions = meta_preds
        if verbose:
            print(f"   ✅ Meta-Learner Per-Target лучше на {improvement:.4f}")
    else:
        best_method = 'weighted'
        best_predictions = weighted_preds
        if verbose:
            print(f"   ✅ Weighted Average лучше (или разница < {min_improvement})")

    comparison_results = {
        'weighted_auc': float(weighted_auc),
        'meta_per_target_auc': float(meta_auc),
        'improvement': float(improvement),
        'best_method': best_method,
        'weighted_weights': weighted_weights,
        'per_target_weights': meta_learner.target_weights if best_method == 'meta_per_target' else {},
        'global_weights': meta_learner.global_weights,
        'shrinkage_factor': shrinkage_factor,
        'min_improvement': min_improvement,
        'n_targets': len(target_cols),
        'n_models': len(model_names),
        'stats': meta_learner.stats
    }

    return best_method, best_predictions, comparison_results