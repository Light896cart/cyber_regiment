# =============================================================================
# scripts/check_data_leakage.py
# 🔍 АНАЛИЗ ДАННЫХ НА DATA LEAKAGE (Stage 2, топ-1500 признаков)
# =============================================================================

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.metrics import roc_auc_score, mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

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
sys.path.append(str(ROOT_DIR))

ARTIFACTS_DIR = ROOT_DIR / "artifacts"


# =============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =============================================================================

def load_data_for_analysis():
    """Загружает данные для анализа утечек."""
    print(f"\n📂 Загрузка данных для анализа...")

    # Основные данные
    X_train = pd.read_parquet(ARTIFACTS_DIR / "X_train_extended_val.parquet")
    y_train = pd.read_parquet(ARTIFACTS_DIR / "y_train_val.parquet")
    X_val = pd.read_parquet(ARTIFACTS_DIR / "X_val_extended_val.parquet")
    y_val = pd.read_parquet(ARTIFACTS_DIR / "y_val_val.parquet")

    # Feature importance
    importance_df = pd.read_csv(ARTIFACTS_DIR / "feature_importance_stage2.csv")
    top_1500 = importance_df.head(1500)['feature'].tolist()

    # OOF predictions Stage 1
    oof_path = ARTIFACTS_DIR / "oof_predictions_STACKED_stage1.parquet"
    df_oof = pd.read_parquet(oof_path) if oof_path.exists() else None

    print(f"   ✅ Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"   ✅ Таргетов: {y_train.shape[1]}")
    print(f"   ✅ Топ-1500 признаков отобрано")

    return X_train, y_train, X_val, y_val, top_1500, df_oof, importance_df


# =============================================================================
# 2. ПРОВЕРКА #1: TARGET LEAKAGE (прямая корреляция с таргетом)
# =============================================================================

def check_target_leakage(X_train, y_train, X_val, y_val, features, threshold=0.95):
    """
    Проверяет признаки на чрезмерную корреляцию с таргетом.
    ⚠️ Если признак коррелирует с таргетом > 0.95 — это красный флаг!
    """
    print(f"\n🔍 ПРОВЕРКА #1: Target Leakage (корреляция с таргетом)")
    print(f"   📊 Порог корреляции: {threshold}")

    suspicious = []

    for target_idx in range(y_train.shape[1]):
        y_target = y_train.iloc[:, target_idx]
        target_name = y_train.columns[target_idx]

        for feat in features[:100]:  # Проверяем топ-100 для скорости
            if feat in X_train.columns:
                x_feat = X_train[feat].dropna()
                y_clean = y_target.loc[x_feat.index]

                if len(x_feat) > 100:
                    # Pearson correlation
                    corr, p_val = stats.pearsonr(x_feat, y_clean)

                    if abs(corr) > threshold and p_val < 0.01:
                        suspicious.append({
                            'feature': feat,
                            'target': target_name,
                            'correlation': corr,
                            'p_value': p_val,
                            'type': 'pearson'
                        })

    if suspicious:
        print(f"   ⚠️  НАЙДЕНО {len(suspicious)} подозрительных признаков!")
        for item in suspicious[:10]:
            print(f"      • {item['feature']} → {item['target']}: r={item['correlation']:.3f}")
    else:
        print(f"   ✅ Прямых утечек не обнаружено (в проверенных признаках)")

    return suspicious


# =============================================================================
# 3. ПРОВЕРКА #2: TRAIN/VAL DISTRIBUTION SHIFT
# =============================================================================

def check_distribution_shift(X_train, X_val, features, threshold=0.3):
    """
    Проверяет, не "слишком хорошо" ли признак разделяет train/val.
    Если признак может предсказать, из какого датасета строка — это утечка!
    """
    print(f"\n🔍 ПРОВЕРКА #2: Distribution Shift (train vs val)")

    suspicious = []

    for feat in features[:200]:  # Топ-200 для скорости
        if feat in X_train.columns and feat in X_val.columns:
            train_vals = X_train[feat].dropna()
            val_vals = X_val[feat].dropna()

            if len(train_vals) > 100 and len(val_vals) > 100:
                # KS test: проверяем, одинаковое ли распределение
                ks_stat, p_val = stats.ks_2samp(train_vals, val_vals)

                # Если p-value очень маленький — распределения разные
                # Это НЕ всегда утечка, но стоит проверить
                if p_val < 0.001 and ks_stat > threshold:
                    # Дополнительная проверка: может ли признак предсказать источник?
                    from sklearn.ensemble import RandomForestClassifier

                    X_src = pd.concat([
                        pd.DataFrame({feat: train_vals, 'source': 0}),
                        pd.DataFrame({feat: val_vals, 'source': 1})
                    ]).sample(frac=1, random_state=42)

                    y_src = X_src['source']
                    X_src_feat = X_src[[feat]]

                    # Простая модель: если легко предсказать источник — подозрительно
                    try:
                        clf = RandomForestClassifier(n_estimators=10, random_state=42)
                        clf.fit(X_src_feat, y_src)
                        score = clf.score(X_src_feat, y_src)

                        if score > 0.7:  # Слишком легко отличить train от val
                            suspicious.append({
                                'feature': feat,
                                'ks_statistic': ks_stat,
                                'p_value': p_val,
                                'source_auc': score,
                                'train_mean': train_vals.mean(),
                                'val_mean': val_vals.mean()
                            })
                    except:
                        pass

    if suspicious:
        print(f"   ⚠️  НАЙДЕНО {len(suspicious)} признаков с сильным shift!")
        for item in suspicious[:10]:
            print(f"      • {item['feature']}: KS={item['ks_statistic']:.3f}, "
                  f"source_AUC={item['source_auc']:.2f}, "
                  f"train_mean={item['train_mean']:.2f}, val_mean={item['val_mean']:.2f}")
    else:
        print(f"   ✅ Сильных распределительных сдвигов не обнаружено")

    return suspicious


# =============================================================================
# 4. ПРОВЕРКА #3: OOF CONSISTENCY (Stage 1 predictions)
# =============================================================================

def check_oof_consistency(y_train, y_val, df_oof, target_cols):
    """
    Проверяет, что OOF predictions из Stage 1 не "слишком хорошие".
    Если OOF AUC близок к 1.0 — возможно, утечка в Stage 1.
    """
    print(f"\n🔍 ПРОВЕРКА #3: OOF Consistency (Stage 1 predictions)")

    if df_oof is None:
        print(f"   ⚠️  OOF файл не найден, пропускаем проверку")
        return []

    suspicious = []

    for target in target_cols:
        oof_col = f"catboost_{target}"  # Или другая модель

        if oof_col in df_oof.columns and target in y_train.columns:
            # Проверяем на train
            y_true = y_train[target]
            y_pred = df_oof.loc[y_true.index, oof_col]

            mask = ~y_true.isna() & ~y_pred.isna()
            if mask.sum() > 100:
                try:
                    auc = roc_auc_score(y_true[mask], y_pred[mask])

                    # AUC > 0.99 на OOF — очень подозрительно
                    if auc > 0.99:
                        suspicious.append({
                            'target': target,
                            'oof_auc': auc,
                            'issue': 'OOF AUC слишком высокий (возможная утечка в Stage 1)'
                        })
                    elif auc < 0.5:
                        suspicious.append({
                            'target': target,
                            'oof_auc': auc,
                            'issue': 'OOF AUC < 0.5 (возможно, инверсия или ошибка)'
                        })
                    else:
                        print(f"   ✅ {target}: OOF AUC = {auc:.4f}")
                except:
                    pass

    if suspicious:
        print(f"   ⚠️  НАЙДЕНО {len(suspicious)} проблем с OOF!")
        for item in suspicious:
            print(f"      • {item['target']}: AUC={item['oof_auc']:.4f} — {item['issue']}")
    else:
        print(f"   ✅ OOF predictions выглядят консистентно")

    return suspicious


# =============================================================================
# 5. ПРОВЕРКА #4: META-FEATURE LEAKAGE
# =============================================================================

def check_meta_feature_leakage(X_train, y_train, features, meta_prefix='meta_'):
    """
    Проверяет мета-признаки на утечки.
    Мета-признаки не должны иметь чрезмерную корреляцию с таргетом.
    """
    print(f"\n🔍 ПРОВЕРКА #4: Meta-Feature Leakage")

    meta_features = [f for f in features if meta_prefix in f]
    if not meta_features:
        print(f"   ℹ️  Мета-признаки с префиксом '{meta_prefix}' не найдены")
        return []

    print(f"   📊 Проверка {len(meta_features)} мета-признаков...")

    suspicious = []

    for feat in meta_features[:50]:  # Проверяем часть для скорости
        if feat in X_train.columns:
            for target_idx in range(min(5, y_train.shape[1])):  # Первые 5 таргетов
                y_target = y_train.iloc[:, target_idx]
                x_feat = X_train[feat].dropna()
                y_clean = y_target.loc[x_feat.index]

                if len(x_feat) > 100:
                    corr, p_val = stats.spearmanr(x_feat, y_clean)

                    # Мета-признаки могут быть информативными, но > 0.8 — подозрительно
                    if abs(corr) > 0.8 and p_val < 0.01:
                        suspicious.append({
                            'feature': feat,
                            'target': y_train.columns[target_idx],
                            'correlation': corr,
                            'p_value': p_val,
                            'type': 'meta_spearman'
                        })

    if suspicious:
        print(f"   ⚠️  НАЙДЕНО {len(suspicious)} подозрительных мета-признаков!")
        for item in suspicious[:10]:
            print(f"      • {item['feature']} → {item['target']}: ρ={item['correlation']:.3f}")
    else:
        print(f"   ✅ Мета-признаки выглядят безопасно")

    return suspicious


# =============================================================================
# 6. ПРОВЕРКА #5: FEATURE IMPORTANCE VALIDATION
# =============================================================================

def check_feature_importance_validity(X_train, y_train, features, importance_df):
    """
    Пересчитывает важность признаков на чистом train и сравнивает с сохранённой.
    Если расхождения большие — возможно, importance считалась с утечкой.
    """
    print(f"\n🔍 ПРОВЕРКА #5: Feature Importance Validity")

    # Берём первые 3 таргета для скорости
    results = []

    for target_idx in range(min(3, y_train.shape[1])):
        y_target = y_train.iloc[:, target_idx]
        target_name = y_train.columns[target_idx]

        # Простая модель для оценки важности
        from lightgbm import LGBMClassifier

        X_subset = X_train[features[:100]].fillna(-999)  # Топ-100 признаков

        try:
            model = LGBMClassifier(
                n_estimators=50,
                learning_rate=0.1,
                verbose=-1,
                random_state=42
            )
            model.fit(X_subset, y_target)

            # Получаем важность
            fi_new = pd.DataFrame({
                'feature': features[:100],
                'importance_recalculated': model.feature_importances_
            })

            # Сравниваем с сохранённой
            fi_saved = importance_df[importance_df['feature'].isin(features[:100])][['feature', 'importance']]

            if len(fi_saved) > 0:
                merged = fi_new.merge(fi_saved, on='feature', how='inner')
                if len(merged) > 10:
                    # Корреляция между старой и новой важностью
                    corr = merged['importance'].corr(merged['importance_recalculated'])

                    results.append({
                        'target': target_name,
                        'importance_correlation': corr,
                        'n_features_compared': len(merged)
                    })

                    if corr < 0.5:
                        print(f"   ⚠️  {target_name}: низкая корреляция важности (r={corr:.3f})")
                    else:
                        print(f"   ✅ {target_name}: важность консистентна (r={corr:.3f})")
        except Exception as e:
            print(f"   ⚠️  {target_name}: ошибка пересчёта — {e}")

    return results


# =============================================================================
# 7. ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    print(f"\n{'=' * 70}")
    print(f"🔍 DATA LEAKAGE ANALYSIS — Stage 2, топ-1500 признаков")
    print(f"{'=' * 70}")

    # Загрузка
    X_train, y_train, X_val, y_val, features, df_oof, importance_df = load_data_for_analysis()

    # Запуск проверок
    all_issues = []

    # 1. Target leakage
    issues_1 = check_target_leakage(X_train, y_train, X_val, y_val, features)
    all_issues.extend(issues_1)

    # 2. Distribution shift
    issues_2 = check_distribution_shift(X_train, X_val, features)
    all_issues.extend(issues_2)

    # 3. OOF consistency
    from src.data.loader import DataLoader
    loader = DataLoader()
    loader.load_full_data()
    target_cols = loader.target_cols
    issues_3 = check_oof_consistency(y_train, y_val, df_oof, target_cols)
    all_issues.extend(issues_3)

    # 4. Meta-feature leakage
    issues_4 = check_meta_feature_leakage(X_train, y_train, features)
    all_issues.extend(issues_4)

    # 5. Feature importance validity
    issues_5 = check_feature_importance_validity(X_train, y_train, features, importance_df)

    # =============================================================================
    # ИТОГОВЫЙ ОТЧЁТ
    # =============================================================================
    print(f"\n{'=' * 70}")
    print(f"📋 ИТОГОВЫЙ ОТЧЁТ")
    print(f"{'=' * 70}")

    if all_issues:
        print(f"\n⚠️  НАЙДЕНО {len(all_issues)} ПОТЕНЦИАЛЬНЫХ ПРОБЛЕМ:")
        for i, issue in enumerate(all_issues, 1):
            print(f"\n   {i}. {issue.get('feature', issue.get('target', 'N/A'))}")
            for k, v in issue.items():
                if k not in ['feature', 'target']:
                    print(f"      • {k}: {v}")

        print(f"\n🔧 РЕКОМЕНДАЦИИ:")
        print(f"   1. Исключи признаки с correlation > 0.95 с таргетом")
        print(f"   2. Проверь, как генерировались OOF predictions в Stage 1")
        print(f"   3. Убедись, что feature importance считалась только на train")
        print(f"   4. Для мета-признаков: пересчитай корреляционную матрицу на train")
    else:
        print(f"\n✅ ЯВНЫХ УТЕЧЕК НЕ ОБНАРУЖЕНО!")
        print(f"   📝 Но это не гарантия: проверяй вручную подозрительные признаки")

    # Сохранение отчёта
    report_path = ARTIFACTS_DIR / "leakage_analysis_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        import json, datetime
        json.dump({
            'timestamp': datetime.datetime.now().isoformat(),
            'n_features_analyzed': len(features),
            'issues_found': len(all_issues),
            'issues': all_issues,
            'importance_validation': issues_5
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Отчёт сохранён: {report_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()