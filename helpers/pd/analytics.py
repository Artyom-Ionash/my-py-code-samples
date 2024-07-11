import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
from typing import Union

pd.set_option("display.width", 1000)
pd.set_option("display.float_format", "{:.2f}".format)


def check_correlation(features: Union[pd.DataFrame, np.ndarray]):
    features = pd.DataFrame(features) if isinstance(features, np.ndarray) else features

    # Вычисление матрицы корреляции
    correlation_matrix = features.corr()

    if "target" in correlation_matrix.columns:
        # Корреляция Пирсона (для линейной связи)
        pearson_corr = correlation_matrix["target"]

        # Корреляция Спирмена (для монотонной связи)
        spearman_corr = features.corr(method="spearman")["target"]

        # Вычисление p-значений для корреляции Пирсона
        pearson_p_values = features.corr().loc[:, "target"].apply(lambda x: x**2)
        pearson_p_values = 2 * (1 - pearson_p_values)

        shapiro_p_value = np.array([*map(lambda key: stats.shapiro(features[key])[1], features)])

        # Создание таблицы с корреляциями и p-значениями
        correlation_table = pd.DataFrame(
            {
                "ID": range(0, len(correlation_matrix.columns)),
                "Корреляция Пирсона": pearson_corr,
                "Корреляция Спирмена": spearman_corr,
                "p-значение (Пирсон)": pearson_p_values,
                "Распределение": np.where(shapiro_p_value > 0.05, "Гауссово", "Ненормальное"),
            },
            index=correlation_matrix.columns,
        )

        # Вывод таблицы
        print(correlation_table)

    feature_dim = features.shape[1]
    print(f"Количество признаков: {feature_dim}")

    abs_correlation_sum: float = round(correlation_matrix.abs().values.sum(), 2)
    if feature_dim == abs_correlation_sum:
        print(f"Признаки не коррелируют между собой!")
        return

    block_size = 30
    if correlation_matrix.shape[0] > block_size:
        print(f"Отобразим только первые {block_size}.")
        correlation_matrix = correlation_matrix.iloc[:block_size, :block_size]

    # Визуализация матрицы корреляции
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title("Матрица корреляции")
    plt.show()


def check_importance(model, X: Union[pd.DataFrame, np.ndarray], verbosity=False):
    X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X

    explainer = shap.explainers.Linear(model, X)
    shap_values_scaled: shap.Explanation = explainer(X)

    # Вычисление среднего абсолютного значения SHAP для каждой фичи
    feature_importance = np.abs(shap_values_scaled.values).mean(axis=0)

    # Получение индексов фичей в порядке важности
    feature_indices = feature_importance.argsort()[::-1]

    # Создание фрейма с данными
    data = {
        "Feature Index": feature_indices,
        "Feature Name": X.columns[feature_indices],
        "Feature Importance": np.round(np.array(feature_importance)[feature_indices], decimals=4),
    }
    df = pd.DataFrame(data)

    if verbosity:
        # Отображение фрейма
        print(df)
        shap.summary_plot(shap_values_scaled, X)

    return df


def plot_feature_grid(X: Union[pd.DataFrame, np.ndarray], y, width=12, hight=12):
    X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X

    num_features = X.shape[1]
    grid_size = (math.ceil(math.sqrt(num_features)), math.ceil(math.sqrt(num_features)))

    fig, axes = plt.subplots(*grid_size, figsize=(width, hight))
    feature_names = X.columns

    for i, ax in enumerate(axes.flatten()):
        if i < num_features:
            feature = feature_names[i]
            ax.scatter(X[feature], y)
            ax.set_xlabel(feature)
            ax.set_ylabel("Целевой признак" if hight > 6 else "t")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
