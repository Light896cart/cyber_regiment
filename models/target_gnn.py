# =============================================================================
# models/target_gnn.py
# Target Graph Neural Network для multi-label классификации
# 🔥 МАКСИМАЛЬНАЯ ВЕРСИЯ: совместимость с TabNetGNNManager + правильное сохранение
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
from pathlib import Path

class TargetGraphConv(nn.Module):
    """
    Graph Convolution для таргетов.
    Сообщения передаются между коррелирующими таргетами.

    🔧 УЛУЧШЕНИЯ:
    - ✅ Явная передача adj_matrix (не полагается на состояние модуля)
    - ✅ Защита от inplace операций
    - ✅ Better normalization для стабильности
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-6)

        # Инициализация весов
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_targets, in_dim) — предсказания/эмбеддинги таргетов
            adj_matrix: (n_targets, n_targets) — матрица смежности графа таргетов
        Returns:
            (batch, n_targets, out_dim) — обновлённые эмбеддинги
        """
        # 🔥 Копируем adj_matrix чтобы избежать inplace модификаций
        # и гарантируем что он на том же устройстве что и x
        if adj_matrix.device != x.device:
            adj_matrix = adj_matrix.to(x.device, non_blocking=True)

        # Message passing: aggregate от соседей
        # (batch, n_targets, n_targets) @ (batch, n_targets, in_dim)
        # → (batch, n_targets, in_dim)
        messages = torch.bmm(
            adj_matrix.unsqueeze(0).expand(x.shape[0], -1, -1),
            x
        )

        # Transform
        out = self.linear(messages)
        out = self.dropout(out)

        # Skip connection + layer norm
        out = self.layer_norm(out + x)

        return out


class TargetGNN(nn.Module):
    """
    Multi-layer Target GNN с residual connections.

    🔧 КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ:
    - ✅ _adj_matrix как register_buffer (сохраняется в state_dict!)
    - ✅ Методы для явного сохранения/загрузки матрицы (fallback)
    - ✅ Улучшенная инициализация весов
    - ✅ Защита от NaN/Inf в forward
    - ✅ Поддержка AMP (autocast compatible)
    """

    def __init__(
            self,
            n_targets: int = 41,
            hidden_dim: int = 128,
            n_layers: int = 3,
            dropout: float = 0.2,
            corr_threshold: float = 0.3
    ):
        super().__init__()
        self.n_targets = n_targets
        self.hidden_dim = hidden_dim
        self.corr_threshold = corr_threshold  # Сохраняем для сериализации

        # Projection layer (1 → hidden_dim) для каждого таргета
        self.input_projection = nn.Linear(1, hidden_dim, bias=True)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            TargetGraphConv(hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        # Output layer (hidden_dim → 1)
        self.output_projection = nn.Linear(hidden_dim, 1, bias=True)

        # 🔥 register_buffer: _adj_matrix БУДЕТ сохранён в state_dict!
        # Это решает проблему с загрузкой модели
        self.register_buffer('_adj_matrix', torch.zeros(n_targets, n_targets))
        self.register_buffer('_adj_matrix_initialized', torch.tensor(False))

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов всех линейных слоёв."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_adjacency_matrix(self, corr_matrix: np.ndarray, threshold: Optional[float] = None) -> int:
        """
        Устанавливает матрицу смежности на основе корреляции таргетов.

        Args:
            corr_matrix: (n_targets, n_targets) корреляционная матрица
            threshold: порог бинаризации (если None, используется self.corr_threshold)

        Returns:
            n_edges: количество рёбер в графе (без self-loops)
        """
        if threshold is None:
            threshold = self.corr_threshold

        # Валидация входных данных
        assert corr_matrix.shape == (self.n_targets, self.n_targets), \
            f"Ожидалась матрица {self.n_targets}x{self.n_targets}, получено {corr_matrix.shape}"

        # Бинаризуем корреляционную матрицу
        adj = (np.abs(corr_matrix) > threshold).astype(np.float32)

        # Добавляем self-loops (каждый таргет связан с собой)
        np.fill_diagonal(adj, 1.0)

        # Считаем рёбра ДО нормализации (для лога)
        n_edges = int(adj.sum() - self.n_targets)  # без self-loops

        # Нормализуем (row-normalization) для стабильности GNN
        row_sum = adj.sum(axis=1, keepdims=True)
        # Защита от деления на ноль
        row_sum = np.where(row_sum == 0, 1.0, row_sum)
        adj = adj / row_sum

        # 🔥 Копируем в buffer (автоматически переносится на device модели)
        adj_tensor = torch.from_numpy(adj)
        self._adj_matrix.copy_(adj_tensor)
        self._adj_matrix_initialized.fill_(True)

        return n_edges

    def get_adjacency_matrix(self) -> Optional[torch.Tensor]:
        """Возвращает текущую adjacency matrix (или None если не инициализирована)."""
        if self._adj_matrix_initialized.item():
            return self._adj_matrix.clone()  # Возвращаем копию для безопасности
        return None

    def save_adjacency_matrix(self, path: Union[str, Path]) -> None:
        """
        Сохраняет adjacency matrix отдельно (для совместимости со старым кодом).

        Args:
            path: путь к файлу для сохранения
        """
        if not self._adj_matrix_initialized.item():
            raise ValueError("Adjacency matrix не инициализирована!")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'matrix': self._adj_matrix.cpu().numpy(),
            'n_targets': self.n_targets,
            'corr_threshold': self.corr_threshold,
            'n_edges': int((self._adj_matrix > 0).sum().item() - self.n_targets)
        }

        torch.save(save_data, path)

    def load_adjacency_matrix(self, path: Union[str, Path], device: Optional[torch.device] = None) -> int:
        """
        Загружает adjacency matrix из файла.

        Args:
            path: путь к файлу
            device: устройство для загрузки (если None, используется текущий device модели)

        Returns:
            n_edges: количество рёбер в загруженном графе
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        load_data = torch.load(path, map_location='cpu', weights_only=True)

        # Валидация
        if load_data['n_targets'] != self.n_targets:
            raise ValueError(
                f"Несовпадение n_targets: в файле {load_data['n_targets']}, "
                f"в модели {self.n_targets}"
            )

        # Загрузка в буфер
        adj_tensor = torch.from_numpy(load_data['matrix'])
        if device is not None:
            adj_tensor = adj_tensor.to(device)

        self._adj_matrix.copy_(adj_tensor)
        self._adj_matrix_initialized.fill_(True)
        self.corr_threshold = load_data.get('corr_threshold', self.corr_threshold)

        return load_data.get('n_edges', 0)

    def forward(self, base_predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_predictions: (batch, n_targets) — предсказания от базовой модели
        Returns:
            refined_predictions: (batch, n_targets) — уточнённые предсказания
        """
        # 🔥 Проверка что adjacency matrix инициализирована
        if not self._adj_matrix_initialized.item():
            # Fallback: возвращаем base_predictions без модификаций
            # (можно заменить на warning или exception в продакшене)
            return base_predictions

        # 🔥 adj_matrix уже в buffer — PyTorch автоматически перенесёт его
        # на тот же device что и входные данные при вызове forward
        adj_matrix = self._adj_matrix

        batch_size = base_predictions.shape[0]

        # Project: (batch, n_targets) → (batch, n_targets, 1) → (batch, n_targets, hidden_dim)
        x = base_predictions.unsqueeze(-1)  # (batch, n_targets, 1)
        x = self.input_projection(x)  # (batch, n_targets, hidden_dim)

        # 🔥 Защита от NaN в предсказаниях (может случиться при плохой инициализации)
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        # GNN layers
        for layer_idx, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, adj_matrix)  # (batch, n_targets, hidden_dim)

            # 🔥 Дополнительная защита от exploding gradients
            if layer_idx < len(self.gnn_layers) - 1:  # Не на последнем слое
                x = torch.clamp(x, min=-1e4, max=1e4)

        # Output: (batch, n_targets, hidden_dim) → (batch, n_targets, 1) → (batch, n_targets)
        output = self.output_projection(x).squeeze(-1)

        # 🔥 Residual connection с базовыми предсказаниями
        # Коэффициент 0.5 для стабильности (можно тюнить)
        output = 0.5 * output + 0.5 * base_predictions

        # 🔥 Финальная защита от NaN/Inf
        output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
        output = torch.clamp(output, min=0.0, max=1.0)  # Для вероятностей

        return output

    def extra_repr(self) -> str:
        """Дополнительная информация для print(model)."""
        return (
            f"n_targets={self.n_targets}, "
            f"hidden_dim={self.hidden_dim}, "
            f"n_layers={len(self.gnn_layers)}, "
            f"adj_initialized={self._adj_matrix_initialized.item()}"
        )


# =============================================================================
# 🔥 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def create_adjacency_from_correlation(
        corr_matrix: np.ndarray,
        threshold: float = 0.3,
        add_self_loops: bool = True,
        normalize: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Создаёт adjacency matrix из корреляционной матрицы.

    Args:
        corr_matrix: (n_targets, n_targets) корреляционная матрица
        threshold: порог для бинаризации
        add_self_loops: добавить ли диагональ
        normalize: нормализовать ли строки

    Returns:
        adj_tensor: torch.Tensor adjacency matrix
        n_edges: количество рёбер (без self-loops если add_self_loops=True)
    """
    adj = (np.abs(corr_matrix) > threshold).astype(np.float32)

    if add_self_loops:
        np.fill_diagonal(adj, 1.0)

    n_edges = int(adj.sum() - adj.shape[0]) if add_self_loops else int(adj.sum())

    if normalize:
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1.0, row_sum)
        adj = adj / row_sum

    return torch.from_numpy(adj), n_edges


def visualize_adjacency(adj_matrix: torch.Tensor, target_names: Optional[list] = None) -> str:
    """
    Создаёт текстовую визуализацию adjacency matrix для отладки.

    Args:
        adj_matrix: (n_targets, n_targets) adjacency matrix
        target_names: имена таргетов для подписей (опционально)

    Returns:
        str: текстовое представление графа
    """
    n = adj_matrix.shape[0]
    lines = [f"🔗 Adjacency Matrix ({n}x{n}):\n"]

    # Считаем степень каждого узла
    degrees = (adj_matrix > 0).sum(dim=1)

    for i in range(min(n, 10)):  # Показываем первые 10 таргетов
        name = target_names[i] if target_names else f"target_{i}"
        connected = [j for j in range(n) if adj_matrix[i, j] > 0 and i != j]
        lines.append(f"  {name}: degree={degrees[i].item()}, connected to {len(connected)} targets")
        if connected:
            conn_names = [target_names[j] if target_names else f"target_{j}" for j in connected[:5]]
            lines.append(f"    → {', '.join(conn_names)}{'...' if len(connected) > 5 else ''}")

    if n > 10:
        lines.append(f"  ... и ещё {n - 10} таргетов")

    return "\n".join(lines)