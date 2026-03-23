# =============================================================================
# scripts/tune_catboost_loop.py
# Запускает multiple trials с полным перезапуском процесса
# =============================================================================

import subprocess
import sys
import time
import json
from pathlib import Path


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
PYTHON_EXE = sys.executable
SINGLE_SCRIPT = ROOT_DIR / "scripts" / "tune_catboost.py"
BEST_PARAMS_PATH = ROOT_DIR / "artifacts" / "optuna_results" / "best_params.json"

N_TRIALS = 100
PAUSE_BETWEEN = 5  # секунд


def get_best_score():
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH, 'r') as f:
            data = json.load(f)
        return data.get('best_score', 0)
    return 0


def main():
    print(f"\n{'=' * 70}")
    print(f"🔮 CATBOOST GPU TUNING - {N_TRIALS} TRIALS")
    print(f"{'=' * 70}")
    print(f"Каждый trials = новый процесс = 100% очистка GPU памяти\n")

    for i in range(1, N_TRIALS + 1):
        print(f"\n{'=' * 70}")
        print(f"TRIAL {i} из {N_TRIALS}")
        print(f"Текущий лучший AUC: {get_best_score():.4f}")
        print(f"{'=' * 70}\n")

        result = subprocess.run([PYTHON_EXE, str(SINGLE_SCRIPT)])

        if result.returncode != 0:
            print(f"\n⚠️  Trial {i} завершился с ошибкой")

        if i < N_TRIALS:
            print(f"\n⏳ Пауза {PAUSE_BETWEEN} секунд...")
            time.sleep(PAUSE_BETWEEN)

    print(f"\n{'=' * 70}")
    print(f"✅ ВСЕ TRIALS ЗАВЕРШЕНЫ")
    print(f"Финальный лучший AUC: {get_best_score():.4f}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()