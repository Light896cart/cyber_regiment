# =============================================================================
# scripts/tune_lgbm_loop.py
# Запускает multiple trials с полным перезапуском процесса (LightGBM)
# =============================================================================

import subprocess
import sys
import time
import json
from pathlib import Path

# =============================================================================
# 1. НАСТРОЙКА ПУТЕЙ
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

ROOT_DIR = get_project_root()
PYTHON_EXE = sys.executable

# 🔥 Путь к скрипту одиночного запуска LightGBM
SINGLE_SCRIPT = ROOT_DIR / "scripts" / "tune_lgbm_test.py"

# 🔥 Путь к результатам LightGBM (отдельно от CatBoost!)
BEST_PARAMS_PATH = ROOT_DIR / "artifacts" / "optuna_results_lgbm" / "best_params.json"

# =============================================================================
# 2. НАСТРОЙКИ ЗАПУСКА
# =============================================================================

N_TRIALS = 100
PAUSE_BETWEEN = 5  # секунд между trials


# =============================================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def get_best_score():
    """Загружает текущий лучший AUC из файла."""
    if BEST_PARAMS_PATH.exists():
        try:
            with open(BEST_PARAMS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('best_score', 0)
        except:
            return 0
    return 0


def get_trial_count():
    """Считает количество завершённых trials из истории."""
    history_path = ROOT_DIR / "artifacts" / "optuna_results_lgbm" / "trials_history.json"
    if history_path.exists():
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return len(history)
        except:
            return 0
    return 0


# =============================================================================
# 4. ОСНОВНОЙ ЦИКЛ
# =============================================================================

def main():
    print(f"\n{'=' * 70}")
    print(f"🔮 LIGHTGBM GPU TUNING - {N_TRIALS} TRIALS")
    print(f"{'=' * 70}")
    print(f"📊 Скрипт: {SINGLE_SCRIPT.name}")
    print(f"💾 Результаты: {BEST_PARAMS_PATH.parent}")
    print(f"🔄 Каждый trials = новый процесс = 100% очистка памяти")
    print(f"⏸️  Пауза между trials: {PAUSE_BETWEEN} сек")
    print(f"{'=' * 70}\n")

    # Проверка существования скрипта
    if not SINGLE_SCRIPT.exists():
        print(f"❌ Ошибка: Скрипт не найден: {SINGLE_SCRIPT}")
        sys.exit(1)

    # Стартовые показатели
    start_trial = get_trial_count()
    start_auc = get_best_score()

    print(f"📈 Старт: Trial #{start_trial + 1}, Лучший AUC: {start_auc:.4f}\n")

    for i in range(1, N_TRIALS + 1):
        current_trial = start_trial + i

        print(f"\n{'=' * 70}")
        print(f"🚀 TRIAL {current_trial} из {start_trial + N_TRIALS}")
        print(f"🏆 Текущий лучший AUC: {get_best_score():.4f}")
        print(f"{'=' * 70}\n")

        # Запуск одиночного скрипта
        result = subprocess.run([PYTHON_EXE, str(SINGLE_SCRIPT)])

        if result.returncode != 0:
            print(f"\n⚠️  Trial {current_trial} завершился с ошибкой (код {result.returncode})")
        else:
            print(f"\n✅ Trial {current_trial} завершён успешно")

        # Пауза перед следующим запуском (кроме последнего)
        if i < N_TRIALS:
            print(f"\n⏳ Пауза {PAUSE_BETWEEN} секунд...")
            time.sleep(PAUSE_BETWEEN)

    # Финальные показатели
    final_auc = get_best_score()
    final_trials = get_trial_count()

    print(f"\n{'=' * 70}")
    print(f"✅ ВСЕ TRIALS ЗАВЕРШЕНЫ")
    print(f"{'=' * 70}")
    print(f"📊 Всего trials выполнено: {final_trials}")
    print(f"📈 Стартовый AUC: {start_auc:.4f}")
    print(f"🏆 Финальный лучший AUC: {final_auc:.4f}")
    print(f"📉 Улучшение: {final_auc - start_auc:.4f}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()