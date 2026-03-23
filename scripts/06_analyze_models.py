# =============================================================================
# scripts/06_analyze_models.py
# Детальный анализ моделей по каждому таргету + статистика таргетов
# =============================================================================

import sys
import gc
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(r"D:\Code\hackaton_cyberpolka_CV")
sys.path.append(str(ROOT_DIR))

from src.data.loader import DataLoader
from models.catboost_model import CatBoostManager
from models.nn_model import NNManager
from models.lgbm_model import LGBMManager


def main():
    # =========================================================
    # 1. Инициализация
    # =========================================================
    print(f"\n{'=' * 80}")
    print(f"🔬 ДЕТАЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ ПО КАЖДОМУ ТАРГЕТУ")
    print(f"{'=' * 80}\n")

    loader = DataLoader(cat_strategy="int")
    loader.load_full_data()
    target_cols = loader.target_cols
    cat_features = loader.cat_features

    # =========================================================
    # 2. Загрузка валидационных данных
    # =========================================================
    print(f"📂 Загрузка валидационных данных...")

    X_val_extended = pd.read_parquet(ROOT_DIR / "artifacts" / "X_val_extended_val.parquet")
    y_val = pd.read_parquet(ROOT_DIR / "artifacts" / "y_val_val.parquet")

    print(f"   📊 Validation: {X_val_extended.shape[0]} строк, {X_val_extended.shape[1]} признаков")
    print(f"   📊 Таргетов: {len(target_cols)}\n")

    # Конвертируем в Polars для моделей
    X_val_pl = pl.from_pandas(X_val_extended)
    y_val_np = y_val.to_numpy()

    # =========================================================
    # 3. Загрузка моделей
    # =========================================================
    print(f"📂 Загрузка моделей...\n")

    # CatBoost
    print(f"   🌲 Загрузка CatBoost...")
    model_cb = CatBoostManager()
    model_cb.load_model("stage2_catboost_validation_v1", fold_folder="catboost")
    preds_cb = model_cb.predict(X_val_pl, cat_features=cat_features)
    print(f"      ✅ CatBoost готов\n")

    # Neural Network
    print(f"   🧠 Загрузка NN...")
    model_nn = NNManager()
    model_nn.load_model("stage2_nn_validation_v1", fold_folder="neural_network")
    preds_nn = model_nn.predict(X_val_pl, cat_features=cat_features)
    print(f"      ✅ NN готов\n")

    # LightGBM
    print(f"   🌳 Загрузка LGBM...")
    model_lgbm = LGBMManager()
    model_lgbm.load_model("stage2_lgbm_validation_v1", fold_folder="lightgbm")
    preds_lgbm = model_lgbm.predict(X_val_pl, cat_features=cat_features)
    print(f"      ✅ LGBM готов\n")

    # =========================================================
    # 4. Статистика таргетов (частота, баланс)
    # =========================================================
    print(f"{'=' * 80}")
    print(f"📊 СТАТИСТИКА ТАРГЕТОВ (частота и баланс)")
    print(f"{'=' * 80}\n")

    target_stats = []
    for i, col in enumerate(target_cols):
        y_true_col = y_val_np[:, i]
        n_positive = int(y_true_col.sum())
        n_negative = len(y_true_col) - n_positive
        positive_ratio = n_positive / len(y_true_col) * 100

        target_stats.append({
            'target': col,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'positive_ratio_%': round(positive_ratio, 3),
            'balance': 'rare' if positive_ratio < 1 else ('imbalanced' if positive_ratio < 5 else 'balanced')
        })

    df_target_stats = pd.DataFrame(target_stats)

    # Сортировка по частоте
    df_target_stats_sorted = df_target_stats.sort_values('positive_ratio_%', ascending=True)

    print(f"   📈 Распределение таргетов по частоте:\n")
    print(f"   {'Таргет':<15} {'Positive':>10} {'Negative':>10} {'Ratio %':>10} {'Balance':>12}")
    print(f"   {'-' * 60}")

    for _, row in df_target_stats_sorted.iterrows():
        print(
            f"   {row['target']:<15} {row['n_positive']:>10} {row['n_negative']:>10} {row['positive_ratio_%']:>10.3f} {row['balance']:>12}")

    # Сводка по балансу
    n_rare = len(df_target_stats[df_target_stats['balance'] == 'rare'])
    n_imbalanced = len(df_target_stats[df_target_stats['balance'] == 'imbalanced'])
    n_balanced = len(df_target_stats[df_target_stats['balance'] == 'balanced'])

    print(f"\n   📊 Сводка по балансу:")
    print(f"      🔴 Rare (<1%): {n_rare} таргетов")
    print(f"      🟡 Imbalanced (1-5%): {n_imbalanced} таргетов")
    print(f"      🟢 Balanced (>5%): {n_balanced} таргетов\n")

    # =========================================================
    # 5. Расчет AUC по каждому таргету
    # =========================================================
    print(f"{'=' * 80}")
    print(f"📊 РАСЧЕТ AUC ПО КАЖДОМУ ТАРГЕТУ")
    print(f"{'=' * 80}\n")

    results = []

    for i, col in enumerate(target_cols):
        y_true_col = y_val_np[:, i]

        # Проверка на валидность таргета
        if len(np.unique(y_true_col)) < 2:
            auc_cb = auc_nn = auc_lgbm = auc_ensemble = np.nan
        else:
            pred_cb_col = preds_cb[col]
            pred_nn_col = preds_nn[col]
            pred_lgbm_col = preds_lgbm[col]

            # Ансамбль (взвешенный)
            pred_ensemble_col = (
                    0.55 * pred_cb_col +
                    0.01 * pred_nn_col +
                    0.44 * pred_lgbm_col
            )

            auc_cb = roc_auc_score(y_true_col, pred_cb_col)
            auc_nn = roc_auc_score(y_true_col, pred_nn_col)
            auc_lgbm = roc_auc_score(y_true_col, pred_lgbm_col)
            auc_ensemble = roc_auc_score(y_true_col, pred_ensemble_col)

        # Определяем лучшую модель для этого таргета
        aucs = {'CatBoost': auc_cb, 'NN': auc_nn, 'LGBM': auc_lgbm}
        best_model = max(aucs, key=aucs.get) if not np.isnan(auc_cb) else 'N/A'

        # Насколько ансамбль лучше лучшей модели
        if not np.isnan(auc_ensemble):
            best_single_auc = max(auc_cb, auc_nn, auc_lgbm)
            ensemble_gain = auc_ensemble - best_single_auc
        else:
            ensemble_gain = np.nan

        # Получаем статистику таргета
        stats_row = df_target_stats[df_target_stats['target'] == col].iloc[0]

        results.append({
            'target': col,
            'n_positive': stats_row['n_positive'],
            'positive_ratio_%': stats_row['positive_ratio_%'],
            'balance': stats_row['balance'],
            'auc_catboost': auc_cb,
            'auc_nn': auc_nn,
            'auc_lgbm': auc_lgbm,
            'auc_ensemble': auc_ensemble,
            'best_model': best_model,
            'ensemble_gain': ensemble_gain
        })

    df_results = pd.DataFrame(results)

    # =========================================================
    # 6. Корреляция между частотой таргета и AUC
    # =========================================================
    print(f"{'=' * 80}")
    print(f"🔗 КОРРЕЛЯЦИЯ: ЧАСТОТА ТАРГЕТА vs AUC")
    print(f"{'=' * 80}\n")

    valid_df = df_results.dropna(subset=['auc_ensemble', 'positive_ratio_%'])

    corr_ensemble = np.corrcoef(valid_df['positive_ratio_%'], valid_df['auc_ensemble'])[0, 1]
    corr_cb = np.corrcoef(valid_df['positive_ratio_%'], valid_df['auc_catboost'])[0, 1]
    corr_nn = np.corrcoef(valid_df['positive_ratio_%'], valid_df['auc_nn'])[0, 1]
    corr_lgbm = np.corrcoef(valid_df['positive_ratio_%'], valid_df['auc_lgbm'])[0, 1]

    print(f"   📊 Корреляция (частота vs AUC):")
    print(f"      Ensemble: {corr_ensemble:+.4f}")
    print(f"      CatBoost: {corr_cb:+.4f}")
    print(f"      NN:       {corr_nn:+.4f}")
    print(f"      LGBM:     {corr_lgbm:+.4f}\n")

    if corr_ensemble > 0.3:
        print(f"   ⚠️  СИЛЬНАЯ корреляция! Редкие таргеты предсказываются хуже.")
        print(f"   💡  Рекомендация: Добавь class weights для редких таргетов\n")
    elif corr_ensemble > 0.1:
        print(f"   ⚠️  СЛАБАЯ корреляция. Частота влияет на качество.\n")
    else:
        print(f"   ✅ Корреляции нет. Модели устойчивы к дисбалансу.\n")

    # =========================================================
    # 7. AUC по группам баланса
    # =========================================================
    print(f"{'=' * 80}")
    print(f"📈 AUC ПО ГРУППАМ БАЛАНСА")
    print(f"{'=' * 80}\n")

    for balance_group in ['rare', 'imbalanced', 'balanced']:
        group_df = df_results[df_results['balance'] == balance_group]
        if len(group_df) > 0:
            print(f"   {balance_group.upper()} ({len(group_df)} таргетов):")
            print(
                f"      Ensemble AUC: {group_df['auc_ensemble'].mean():.4f} (+/- {group_df['auc_ensemble'].std():.4f})")
            print(f"      CatBoost AUC: {group_df['auc_catboost'].mean():.4f}")
            print(f"      NN AUC:       {group_df['auc_nn'].mean():.4f}")
            print(f"      LGBM AUC:     {group_df['auc_lgbm'].mean():.4f}")
            print(
                f"      Min AUC:      {group_df['auc_ensemble'].min():.4f} ({group_df.loc[group_df['auc_ensemble'].idxmin(), 'target']})")
            print(
                f"      Max AUC:      {group_df['auc_ensemble'].max():.4f} ({group_df.loc[group_df['auc_ensemble'].idxmax(), 'target']})\n")

    # =========================================================
    # 8. Сводная статистика
    # =========================================================
    print(f"{'=' * 80}")
    print(f"📈 СВОДНАЯ СТАТИСТИКА")
    print(f"{'=' * 80}\n")

    summary = {
        'Metric': ['Mean AUC', 'Std AUC', 'Min AUC', 'Max AUC', 'Median AUC'],
        'CatBoost': [
            df_results['auc_catboost'].mean(),
            df_results['auc_catboost'].std(),
            df_results['auc_catboost'].min(),
            df_results['auc_catboost'].max(),
            df_results['auc_catboost'].median()
        ],
        'NN': [
            df_results['auc_nn'].mean(),
            df_results['auc_nn'].std(),
            df_results['auc_nn'].min(),
            df_results['auc_nn'].max(),
            df_results['auc_nn'].median()
        ],
        'LGBM': [
            df_results['auc_lgbm'].mean(),
            df_results['auc_lgbm'].std(),
            df_results['auc_lgbm'].min(),
            df_results['auc_lgbm'].max(),
            df_results['auc_lgbm'].median()
        ],
        'Ensemble': [
            df_results['auc_ensemble'].mean(),
            df_results['auc_ensemble'].std(),
            df_results['auc_ensemble'].min(),
            df_results['auc_ensemble'].max(),
            df_results['auc_ensemble'].median()
        ]
    }

    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    # =========================================================
    # 9. Худшие таргеты (Top 10)
    # =========================================================
    print(f"\n{'=' * 80}")
    print(f"❌ ТОП-10 ХУДШИХ ТАРГЕТОВ (по AUC ансамбля)")
    print(f"{'=' * 80}\n")

    df_worst = df_results.nsmallest(10, 'auc_ensemble')

    print(df_worst[['target', 'n_positive', 'positive_ratio_%', 'auc_catboost', 'auc_nn', 'auc_lgbm', 'auc_ensemble',
                    'best_model']].to_string(index=False))

    # =========================================================
    # 10. Лучшие таргеты (Top 10)
    # =========================================================
    print(f"\n{'=' * 80}")
    print(f"✅ ТОП-10 ЛУЧШИХ ТАРГЕТОВ (по AUC ансамбля)")
    print(f"{'=' * 80}\n")

    df_best = df_results.nlargest(10, 'auc_ensemble')

    print(df_best[['target', 'n_positive', 'positive_ratio_%', 'auc_catboost', 'auc_nn', 'auc_lgbm', 'auc_ensemble',
                   'best_model']].to_string(index=False))

    # =========================================================
    # 11. Какая модель лучше для каких таргетов
    # =========================================================
    print(f"\n{'=' * 80}")
    print(f"🏆 КАКАЯ МОДЕЛЬ ЛУЧШЕ ДЛЯ КАКИХ ТАРГЕТОВ")
    print(f"{'=' * 80}\n")

    model_wins = df_results['best_model'].value_counts()
    print(model_wins)
    print(f"\n   CatBoost лучший в: {model_wins.get('CatBoost', 0)} из {len(target_cols)} таргетов")
    print(f"   NN лучший в: {model_wins.get('NN', 0)} из {len(target_cols)} таргетов")
    print(f"   LGBM лучший в: {model_wins.get('LGBM', 0)} из {len(target_cols)} таргетов")

    # =========================================================
    # 12. Детальный анализ проблемных таргетов
    # =========================================================
    print(f"\n{'=' * 80}")
    print(f"🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПРОБЛЕМНЫХ ТАРГЕТОВ (AUC < 0.70)")
    print(f"{'=' * 80}\n")

    problematic = df_results[df_results['auc_ensemble'] < 0.70].sort_values('auc_ensemble')

    if len(problematic) > 0:
        print(f"   Найдено {len(problematic)} проблемных таргетов:\n")

        for _, row in problematic.head(5).iterrows():
            print(f"   ┌─ {row['target']}")
            print(f"   │  Positive samples: {row['n_positive']} ({row['positive_ratio_%']:.3f}%)")
            print(f"   │  Balance: {row['balance']}")
            print(f"   │  Ensemble AUC: {row['auc_ensemble']:.4f}")
            print(f"   │  CatBoost:     {row['auc_catboost']:.4f}")
            print(f"   │  NN:           {row['auc_nn']:.4f}")
            print(f"   │  LGBM:         {row['auc_lgbm']:.4f}")
            print(f"   │  Лучшая модель: {row['best_model']}")

            # Анализ: какая модель больше всего ухудшает
            aucs = [row['auc_catboost'], row['auc_nn'], row['auc_lgbm']]
            model_names = ['CatBoost', 'NN', 'LGBM']
            worst_idx = np.argmin(aucs)
            print(f"   │  ⚠️  Худшая модель: {model_names[worst_idx]} ({aucs[worst_idx]:.4f})")
            print(f"   └─\n")
    else:
        print(f"   ✅ Нет таргетов с AUC < 0.70!\n")

    # =========================================================
    # 13. Влияние NN на ансамбль
    # =========================================================
    print(f"{'=' * 80}")
    print(f"🧠 ВЛИЯНИЕ NEURAL NETWORK НА АНСАМБЛЬ")
    print(f"{'=' * 80}\n")

    # NN имеет вес ~0.01 в ансамбле - проверить, стоит ли её включать
    df_no_nn = df_results.copy()
    df_no_nn['auc_no_nn'] = (
            0.555 * df_no_nn['auc_catboost'] +
            0.445 * df_no_nn['auc_lgbm']
    )

    nn_impact = df_no_nn['auc_ensemble'] - df_no_nn['auc_no_nn']
    print(f"   Среднее влияние NN на ансамбль: {nn_impact.mean():+.6f}")
    print(f"   Максимальное положительное влияние: {nn_impact.max():+.6f}")
    print(f"   Максимальное отрицательное влияние: {nn_impact.min():+.6f}")

    positive_impact = (nn_impact > 0).sum()
    negative_impact = (nn_impact < 0).sum()
    print(f"\n   NN улучшает ансамбль в {positive_impact} таргетах")
    print(f"   NN ухудшает ансамбль в {negative_impact} таргетах")

    if negative_impact > positive_impact:
        print(f"\n   ⚠️  РЕКОМЕНДАЦИЯ: NN имеет вес 0.01 и чаще ухудшает результат.")
        print(f"   💡  Попробуй убрать NN из ансамбля или уменьшить её вес до 0.005")
    else:
        print(f"\n   ✅ NN приносит пользу, текущий вес оптимален")

    # =========================================================
    # 14. Сохранение результатов
    # =========================================================
    print(f"\n{'=' * 80}")
    print(f"💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print(f"{'=' * 80}\n")

    # Детальные результаты по таргетам
    output_path = ROOT_DIR / "artifacts" / "model_analysis_per_target.csv"
    df_results.to_csv(output_path, index=False, float_format='%.6f')
    print(f"   ✅ Сохранено: {output_path}")

    # Статистика таргетов
    stats_path = ROOT_DIR / "artifacts" / "target_statistics.csv"
    df_target_stats.to_csv(stats_path, index=False, float_format='%.3f')
    print(f"   ✅ Сохранено: {stats_path}")

    # Сводная статистика
    summary_path = ROOT_DIR / "artifacts" / "model_analysis_summary.csv"
    df_summary.to_csv(summary_path, index=False, float_format='%.6f')
    print(f"   ✅ Сохранено: {summary_path}")

    # JSON с полным анализом
    json_path = ROOT_DIR / "artifacts" / "model_analysis_full.json"
    analysis_data = {
        'target_statistics': df_target_stats.to_dict(),
        'correlation': {
            'ensemble': float(corr_ensemble),
            'catboost': float(corr_cb),
            'nn': float(corr_nn),
            'lgbm': float(corr_lgbm)
        },
        'summary': df_summary.to_dict(),
        'worst_10_targets': df_worst.to_dict(),
        'best_10_targets': df_best.to_dict(),
        'model_wins': model_wins.to_dict(),
        'problematic_targets_count': len(problematic),
        'nn_impact': {
            'mean': float(nn_impact.mean()),
            'max': float(nn_impact.max()),
            'min': float(nn_impact.min()),
            'positive_count': int(positive_impact),
            'negative_count': int(negative_impact)
        },
        'balance_summary': {
            'rare': int(n_rare),
            'imbalanced': int(n_imbalanced),
            'balanced': int(n_balanced)
        }
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    print(f"   ✅ Сохранено: {json_path}")

    # =========================================================
    # 15. Финальные рекомендации
    # =========================================================
    print(f"\n{'=' * 80}")
    print(f"💡 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ")
    print(f"{'=' * 80}\n")

    # 1. Проблема с редкими таргетами
    if n_rare > 0:
        rare_auc = df_results[df_results['balance'] == 'rare']['auc_ensemble'].mean()
        balanced_auc = df_results[df_results['balance'] == 'balanced']['auc_ensemble'].mean()
        gap = balanced_auc - rare_auc

        print(f"   ⚠️  PROBLEM 1: {n_rare} редких таргетов (<1% positive)")
        print(f"      AUC редких: {rare_auc:.4f} vs AUC сбалансированных: {balanced_auc:.4f}")
        print(f"      Разрыв: {gap:.4f}")
        print(f"      Решение:")
        print(f"      - Добавь class weights для редких таргетов")
        print(f"      - Попробуй focal loss для NN")
        print(f"      - Увеличь min_data_in_leaf для LGBM на редких таргетах")
        print(f"      Ожидаемый прирост: +0.005-0.015 AUC\n")

    # 2. Корреляция частоты и AUC
    if corr_ensemble > 0.3:
        print(f"   ⚠️  PROBLEM 2: Сильная корреляция частоты и AUC ({corr_ensemble:+.4f})")
        print(f"      Решение: Добавь таргет-специфичные гиперпараметры\n")

    # 3. CatBoost доминирует
    if model_wins.get('CatBoost', 0) > 25:
        print(f"   ✅ GOOD: CatBoost лучший в {model_wins.get('CatBoost', 0)} таргетах")
        print(f"      Рекомендация: Увеличь вес CatBoost в ансамбле до 0.60\n")

    # 4. LGBM стабилен
    if model_wins.get('LGBM', 0) > 10:
        print(f"   ✅ GOOD: LGBM лучший в {model_wins.get('LGBM', 0)} таргетах")
        print(f"      Рекомендация: Текущий вес 0.44 оптимален\n")

    # 5. NN приносит пользу
    if positive_impact > negative_impact:
        print(f"   ✅ GOOD: NN улучшает {positive_impact} из {len(target_cols)} таргетов")
        print(f"      Рекомендация: Оставь вес NN ~0.01\n")

    print(f"{'=' * 80}")
    print(f"✅ АНАЛИЗ ЗАВЕРШЕН")
    print(f"{'=' * 80}\n")

    # =============================================================================
    # 16. АНАЛИЗ: Какие признаки важны для худших таргетов?
    # =============================================================================
    print(f"\n{'=' * 80}")
    print(f"🔍 АНАЛИЗ: Признаки для проблемных таргетов")
    print(f"{'=' * 80}\n")

    # Загружаем feature importance
    importance_path = ROOT_DIR / "artifacts" / "feature_importance_stage2.csv"
    if importance_path.exists():
        imp_df = pd.read_csv(importance_path)

        # Берём 5 худших таргетов
        worst_5 = df_results.nsmallest(5, 'auc_ensemble')['target'].tolist()

        print(f"   Топ-10 признаков для худших таргетов:\n")
        for target in worst_5:
            # Здесь можно добавить логику извлечения важности по таргету,
            # если она хранится отдельно. Пока заглушка:
            print(f"   • {target}: (требуется таргет-специфичная importance)")

        print(f"\n   💡 Рекомендация: сохрани importance отдельно для каждого таргета")

    # =============================================================================
    # 17. АНАЛИЗ: Калибровка предсказаний (реальная частота vs предсказанная)
    # =============================================================================
    print(f"\n{'=' * 80}")
    print(f"📐 АНАЛИЗ: Калибровка предсказаний")
    print(f"{'=' * 80}\n")

    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    # Проверяем 3 случайных таргета
    sample_targets = np.random.choice(target_cols, 3, replace=False)

    for target in sample_targets:
        y_true = y_val[target].values
        y_pred = preds_cb[target]  # или ансамбль

        # Биннинг предсказаний
        fractions, means = calibration_curve(y_true, y_pred, n_bins=10)

        # Идеальная калибровка: fractions == means
        calibration_error = np.mean(np.abs(fractions - means))

        print(f"   {target}:")
        print(f"      Calibration Error: {calibration_error:.4f} (чем меньше, тем лучше)")
        print(f"      Mean prediction: {y_pred.mean():.4f}, Actual positive rate: {y_true.mean():.4f}")

        if calibration_error > 0.1:
            print(f"      ⚠️  Плохая калибровка! Попробуй Platt scaling или isotonic regression")
        print()

    # =============================================================================
    # 18. АНАЛИЗ: Паттерны ошибок (для 3 худших таргетов)
    # =============================================================================
    print(f"\n{'=' * 80}")
    print(f"❌ АНАЛИЗ: Паттерны ошибок")
    print(f"{'=' * 80}\n")

    worst_3 = df_results.nsmallest(3, 'auc_ensemble')['target'].tolist()

    for target in worst_3:
        y_true = y_val[target].values
        y_pred = preds_cb[target]  # или ансамбль

        # Порог 0.5 для бинаризации
        y_pred_bin = (y_pred >= 0.5).astype(int)

        # Матрица ошибок
        tp = ((y_true == 1) & (y_pred_bin == 1)).sum()
        fp = ((y_true == 0) & (y_pred_bin == 1)).sum()
        fn = ((y_true == 1) & (y_pred_bin == 0)).sum()
        tn = ((y_true == 0) & (y_pred_bin == 0)).sum()

        print(f"   {target}:")
        print(f"      TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"      Precision: {tp / (tp + fp):.4f}" if (tp + fp) > 0 else "      Precision: N/A")
        print(f"      Recall:    {tp / (tp + fn):.4f}" if (tp + fn) > 0 else "      Recall: N/A")

        # Если много FP → модель слишком "оптимистична"
        # Если много FN → модель слишком "пессимистична"
        if fp > fn * 2:
            print(f"      ⚠️  Много ложных срабатываний → повысь порог классификации")
        elif fn > fp * 2:
            print(f"      ⚠️  Много пропусков → понизь порог или добавь чувствительности")
        print()

    # =============================================================================
    # 19. АНАЛИЗ: Насколько лучше рандома?
    # =============================================================================
    print(f"\n{'=' * 80}")
    print(f"🎲 АНАЛИЗ: Прирост над случайным угадыванием")
    print(f"{'=' * 80}\n")

    # Случайный бейзлайн: предсказываем частоту позитивов
    random_gains = []
    for target in target_cols:
        y_true = y_val[target].values
        if len(np.unique(y_true)) < 2:
            continue

        # Random baseline: предсказываем P(positive) для всех
        base_rate = y_true.mean()
        y_random = np.full(len(y_true), base_rate)

        auc_random = roc_auc_score(y_true, y_random)  # всегда ~0.5
        auc_model = df_results[df_results['target'] == target]['auc_ensemble'].values[0]

        if not np.isnan(auc_model):
            gain = auc_model - 0.5  # прирост над рандомом
            random_gains.append({'target': target, 'gain': gain})

    df_gains = pd.DataFrame(random_gains)
    print(f"   Средний прирост над рандомом: {df_gains['gain'].mean():+.4f}")
    print(
        f"   Минимальный прирост: {df_gains['gain'].min():+.4f} ({df_gains.nsmallest(1, 'gain')['target'].values[0]})")
    print(
        f"   Максимальный прирост: {df_gains['gain'].max():+.4f} ({df_gains.nlargest(1, 'gain')['target'].values[0]})")

    # Сколько таргетов реально лучше рандома?
    better_than_random = (df_gains['gain'] > 0.02).sum()  # порог 0.02 для значимости
    print(f"\n   ✅ {better_than_random} из {len(target_cols)} таргетов значимо лучше рандома (+0.02 AUC)")

    # =============================================================================
    # 20. АНАЛИЗ: Оптимизация весов ансамбля (грубый поиск)
    # =============================================================================
    print(f"\n{'=' * 80}")
    print(f"⚖️  АНАЛИЗ: Подбор оптимальных весов ансамбля")
    print(f"{'=' * 80}\n")

    from itertools import product

    # Грубый поиск по сетке весов
    best_weights = None
    best_mean_auc = 0

    for w_cb in np.arange(0.4, 0.8, 0.1):
        for w_lgbm in np.arange(0.2, 0.5, 0.1):
            w_nn = 1 - w_cb - w_lgbm
            if w_nn < 0 or w_nn > 0.2:  # NN не больше 20%
                continue

            aucs = []
            for _, row in df_results.iterrows():
                if np.isnan(row['auc_catboost']):
                    continue
                ensemble_auc = (
                        w_cb * row['auc_catboost'] +
                        w_nn * row['auc_nn'] +
                        w_lgbm * row['auc_lgbm']
                )
                aucs.append(ensemble_auc)

            mean_auc = np.mean(aucs)
            if mean_auc > best_mean_auc:
                best_mean_auc = mean_auc
                best_weights = (w_cb, w_nn, w_lgbm)

    if best_weights:
        print(f"   🔍 Найденные оптимальные веса:")
        print(f"      CatBoost: {best_weights[0]:.2f} (было 0.55)")
        print(f"      NN:       {best_weights[1]:.2f} (было 0.01)")
        print(f"      LGBM:     {best_weights[2]:.2f} (было 0.44)")
        print(f"   📈 Ожидаемый Mean AUC: {best_mean_auc:.4f} (было {df_results['auc_ensemble'].mean():.4f})")
        print(f"\n   💡 Примени эти веса в продакшене!")


if __name__ == "__main__":
    main()