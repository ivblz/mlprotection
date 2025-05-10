import pandas as pd
import numpy as np
import os
import datetime
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



def create_backup(df):
    """
    Создаём резервную копию датасета в директорию backups
    """
    if not os.path.exists('backups'):
        os.makedirs('backups')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"dataset_backup_{timestamp}.csv"
    backup_path = os.path.join('backups', backup_filename)

    df.to_csv(backup_path, index=False)
    print(f"Резервная копия создана: {backup_path}")
    return backup_path


def detection_methods(row):
    """
    Определяет методы, которыми была обнаружена аномалия в строке
    """
    methods = []
    if row.get('isolation_forest_anomaly_flag', 0) == 1:
        methods.append('isolation_forest')
    if row.get('zscore_anomaly_flag', 0) == 1:
        methods.append('zscore')
    if row.get('dbscan_anomaly_flag', 0) == 1:
        methods.append('dbscan')
    return ','.join(methods)


def check_isolation_forest(df_input, contamination=0.05, visualize=False):
    """
    Обнаруживает аномалии в датасете с использованием Isolation Forest
    """
    features = df_input.select_dtypes(include=[np.number]).columns.tolist()
    if not features:
        print("Isolation Forest: Нет числовых признаков для анализа.")
        return pd.Series(0, index=df_input.index), 0.0

    df_features_only = df_input[features]

    model = IsolationForest(contamination=contamination, random_state=42)
    predictions_raw = model.fit_predict(df_features_only)

    anomaly_flags_for_features = pd.Series(
        np.where(predictions_raw == -1, 1, 0),
        index=df_features_only.index)

    final_anomaly_flags = pd.Series(0, index=df_input.index)
    final_anomaly_flags.update(
        anomaly_flags_for_features[anomaly_flags_for_features == 1])

    n_anomalies = final_anomaly_flags.sum()
    percent = (
        round(n_anomalies / len(df_input) * 100, 2) if len(df_input) > 0 else 0.0)
    
    print(f"Isolation Forest обнаружил {n_anomalies} аномалий ({percent}%)")

    if visualize and len(features) >= 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df_input[features[0]],
            df_input[features[1]],
            c=final_anomaly_flags,
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        plt.colorbar(label='Аномалия')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title('Результаты Isolation Forest')
        plt.show()

    return final_anomaly_flags, percent


def check_dbscan(df_input, eps=0.5, min_samples=5, visualize=False):
    """
    Обнаруживает аномалии в датасете с использованием DBSCAN
    """
    features = df_input.select_dtypes(include=[np.number]).columns.tolist()
    if not features:
        print("DBSCAN: Нет числовых признаков для анализа.")
        return pd.Series(0, index=df_input.index), 0.0

    df_features_only = df_input[features]

    scaler = StandardScaler()
    if df_features_only.empty:
        print("DBSCAN: DataFrame с признаками пуст после выбора числовых колонок.")
        return pd.Series(0, index=df_input.index), 0.0

    scaled_data = scaler.fit_transform(df_features_only)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)

    anomaly_flags_for_features = pd.Series(
        np.where(clusters == -1, 1, 0),
        index=df_features_only.index)

    final_anomaly_flags = pd.Series(0, index=df_input.index)
    final_anomaly_flags.update(
        anomaly_flags_for_features[anomaly_flags_for_features == 1])

    n_anomalies = final_anomaly_flags.sum()
    percent = (
        round(n_anomalies / len(df_input) * 100, 2) if len(df_input) > 0 else 0.0)
    
    print(f"DBSCAN обнаружил {n_anomalies} аномалий ({percent}%)")

    if visualize and len(features) >= 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df_input[features[0]],
            df_input[features[1]],
            c=final_anomaly_flags,
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        plt.colorbar(label='Аномалия (DBSCAN)')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title('Результаты DBSCAN')
        plt.show()

    return final_anomaly_flags, percent


def check_zscore(df_input, threshold=3.0, visualize=False):
    """
    Обнаруживает аномалии в датасете на основе Z-score
    """
    numerical_columns = df_input.select_dtypes(include=[np.number]).columns.tolist()
    if not numerical_columns:
        print("Z-score: Нет числовых признаков для анализа.")
        return pd.Series(0, index=df_input.index), 0.0

    overall_anomaly_flags_np = np.zeros(len(df_input), dtype=int)
    zscore_values_for_viz = pd.DataFrame(index=df_input.index)

    for col in numerical_columns:
        col_mean = df_input[col].mean()
        col_std = df_input[col].std()

        if col_std == 0 or pd.isna(col_std):
            col_zscores = pd.Series(0.0, index=df_input.index)
        else:
            col_zscores = np.abs((df_input[col] - col_mean) / col_std)

        zscore_values_for_viz[col + '_zscore'] = col_zscores.fillna(0)
        overall_anomaly_flags_np = np.where(
            col_zscores.fillna(False) > threshold,
            1,
            overall_anomaly_flags_np)

    final_anomaly_flags = pd.Series(overall_anomaly_flags_np, index=df_input.index)

    n_anomalies = final_anomaly_flags.sum()
    percent = (
        round(n_anomalies / len(df_input) * 100, 2) if len(df_input) > 0 else 0.0)

    print(f"Z-score обнаружил {n_anomalies} аномалий ({percent}%)")

    if visualize and len(numerical_columns) > 0:
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(numerical_columns[:min(4, len(numerical_columns))]):
            plt.subplot(2, 2, i + 1)
            sns.histplot(
                zscore_values_for_viz[col + '_zscore'], bins=30, kde=True
            )
            plt.axvline(
                x=threshold,
                color='r',
                linestyle='--',
                label=f'Порог ({threshold})'
            )
            plt.title(f'Распределение Z-score для {col}')
            plt.legend()
        plt.tight_layout()
        plt.show()

    return final_anomaly_flags, percent


def start(df, save=False, classification=False, visualize=False, backups=False):
    """
    Главнаая функция для анализа датасета

    Применяем различные методы аанализа (Isolation Forest, Z-score, DBSCAN)
    выводим сводную информацию и сохраняет результаты
    """
    df_original_columns = df.columns.tolist()
    df_copy = df.copy()

    if backups:
        create_backup(df_copy)

    if save:
        if os.path.exists('warning_data.csv'): os.remove('warning_data.csv')
        if os.path.exists('treat_data.csv'): os.remove('treat_data.csv')

    original_size = len(df_copy)
    df_copy.dropna(inplace=True)
    removed_rows = original_size - len(df_copy)
    if removed_rows > 0:
        print(f"Не учитываются {removed_rows} строк с None/NaN значениями ({removed_rows/original_size*100:.2f}% от исходного датасета)\n")

    df_copy['isolation_forest_anomaly_flag'] = 0
    df_copy['zscore_anomaly_flag'] = 0
    df_copy['dbscan_anomaly_flag'] = 0

    isolation_flags, isolation_percent = check_isolation_forest(df_copy, visualize=visualize)
    df_copy.loc[isolation_flags.index, 'isolation_forest_anomaly_flag'] = isolation_flags

    zscore_flags, zscore_percent = check_zscore(df_copy, visualize=visualize)
    df_copy.loc[zscore_flags.index, 'zscore_anomaly_flag'] = zscore_flags

    dbscan_percent = 0.0
    if classification and isinstance(classification, int) and classification > 0:
        dbscan_flags, dbscan_percent_val = check_dbscan(df_copy,
            min_samples=classification,
            visualize=visualize)
        df_copy.loc[dbscan_flags.index, 'dbscan_anomaly_flag'] = dbscan_flags
        dbscan_percent = dbscan_percent_val
        avg_pct = round(
            (isolation_percent + dbscan_percent + zscore_percent) / 3, 2)
    else:
        avg_pct = round((isolation_percent + zscore_percent) / 2, 2)

    print(f"Средний процент аномалий: {avg_pct}%\n")

    if avg_pct > 15:
        print(f"ВНИМАНИЕ: Обнаружены аномалии ({avg_pct}%). Рекомендуется проверить датасет.")
    else:
        print(f"Предположительно с датасетом все хорошо. Процент аномалий: {avg_pct}%\n")

    if save:
        df_copy['is_warning'] = 0
        df_copy.loc[df_copy['isolation_forest_anomaly_flag'] == 1, 'is_warning'] = 1
        df_copy.loc[df_copy['zscore_anomaly_flag'] == 1, 'is_warning'] = 1
        if classification and isinstance(classification, int) and classification > 0:
            df_copy.loc[df_copy['dbscan_anomaly_flag'] == 1, 'is_warning'] = 1

        warning_df = df_copy[df_copy['is_warning'] == 1].copy()
        
        columns_for_warning_csv = df_original_columns + ['detection_methods']

        if not warning_df.empty:
            warning_df['detection_methods'] = warning_df.apply(detection_methods, axis=1)
            warning_df_to_save = warning_df[columns_for_warning_csv]
        else:
            warning_df_to_save = pd.DataFrame(columns=columns_for_warning_csv)

        warning_df_to_save.to_csv('warning_data.csv', index=False)
        print("Подозрительные данные сохранены в warning_data.csv")

        treated_df = df_copy[df_copy['is_warning'] == 0].copy()
        treated_df_to_save = treated_df[df_original_columns]
        treated_df_to_save.to_csv('treat_data.csv', index=False)
        print("Обработанный датасет (без предупреждений) сохранен в treat_data.csv")

    print("Проверка завершена")