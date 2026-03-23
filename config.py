"""
config.py - Централизованные настройки пайплайна
"""
from pathlib import Path

# =============================================================================
# ПУТИ К ДАННЫМ
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

BASE_DIR = get_project_root()
DATA_DIR = BASE_DIR / "data_new"
OUTPUT_DIR = BASE_DIR / "data_new"  # Сохраняем туда же

# Входные файлы
TRAIN_MAIN = DATA_DIR / "train_main_features.parquet"
TEST_MAIN = DATA_DIR / "test_main_features.parquet"
TRAIN_EXTRA = DATA_DIR / "train_extra_features.parquet"
TEST_EXTRA = DATA_DIR / "test_extra_features.parquet"
TRAIN_TARGET = DATA_DIR / "train_target.parquet"  # Если есть отдельно

# Выходные файлы
TRAIN_FINAL = OUTPUT_DIR / "train_final_processed.parquet"
TEST_FINAL = OUTPUT_DIR / "test_final_processed.parquet"
METADATA_FILE = OUTPUT_DIR / "processing_metadata.json"

# =============================================================================
# ПАРАМЕТРЫ ОБРАБОТКИ
# =============================================================================
ID_COLUMN = "customer_id"

# Порог удаления колонок (% пропусков)
NULL_THRESHOLD = 0.80  # Удаляем колонки где >80% null

# Null PCA
NULL_PCA_COMPONENTS = 20  # Количество компонент SVD

# Флаги пропусков
CREATE_NULL_FLAGS = True  # Создавать ли флаги was_null
MIN_NULL_RATIO_FOR_FLAG = 0.01  # Минимум 1% пропусков для создания флага

# Ratio features (подбираются автоматически)
CREATE_RATIO_FEATURES = True
MAX_RATIO_PAIRS = 50  # Максимум пар отношений

# Row statistics
CREATE_ROW_STATS = True

# Импутирование
IMPUTE_STRATEGY = "median"  # "median" или "mean"

# Оптимизация памяти
TARGET_FLOAT_TYPE = "float32"
TARGET_INT_TYPE = "int32"

# Логирование
VERBOSE = True
SAVE_INTERMEDIATE = False  # Сохранять промежуточные этапы