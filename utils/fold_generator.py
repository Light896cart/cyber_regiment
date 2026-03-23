import numpy as np
from typing import Generator, Tuple, Optional
from sklearn.model_selection import KFold, StratifiedKFold


def get_cv_splits(
        y: np.ndarray,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        use_stratified: bool = True
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Генератор индексов для Cross-Validation.

    Args:
        y: Массив таргетов (n_samples,) или (n_samples, n_targets)
        n_splits: Количество фолдов (1 = Hold-out)
        shuffle: Перемешивать ли данные
        random_state: Seed для воспроизводимости
        use_stratified: Использовать стратификацию (для multi-label — по сумме таргетов)

    Yields:
        (train_idx, val_idx) — индексы для train и validation
    """
    if n_splits == 1:
        # Режим Hold-out (быстрый тест)
        n_samples = len(y) if y.ndim == 1 else y.shape[0]
        indices = np.arange(n_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        split_point = int(n_samples * 0.8)
        yield indices[:split_point], indices[split_point:]
        return

    # Для multi-label стратифицируем по сумме активных таргетов
    if use_stratified and y.ndim == 2:
        y_strat = y.sum(axis=1)  # Сумма таргетов на клиента
    elif use_stratified and y.ndim == 1:
        y_strat = y
    else:
        y_strat = None

    if y_strat is not None:
        kf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        for train_idx, val_idx in kf.split(np.zeros(len(y)), y_strat):
            yield train_idx, val_idx
    else:
        kf = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        for train_idx, val_idx in kf.split(np.zeros(len(y))):
            yield train_idx, val_idx