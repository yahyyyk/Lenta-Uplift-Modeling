"""Все графики проекта; используется неинтерактивный backend для серверного запуска."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklift.metrics import qini_curve

sns.set_theme(style="whitegrid", font="DejaVu Sans")


def save_figure(path: Path) -> None:
    """Сохранить текущую фигуру и освободить память."""
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_qini(y: np.ndarray, t: np.ndarray, predictions: dict[str, np.ndarray],
              path: Path) -> None:
    """Построить общую Qini-кривую: накопленный отклик treatment сверх контроля."""
    plt.figure(figsize=(10, 6))
    for name, pred in predictions.items():
        n, gain = qini_curve(y, pred, t)
        plt.plot(n / len(y), gain, label=name)
    endpoint = y[t == 1].sum() - y[t == 0].sum() * t.sum() / (1-t).sum()
    plt.plot([0, 1], [0, endpoint], "k--", label="Случайный таргетинг")
    plt.title("Lenta · Qini на test")
    plt.xlabel("Доля клиентов с наибольшим прогнозом uplift")
    plt.ylabel("Qini gain (масштаб treatment)")
    plt.legend(fontsize=9)
    save_figure(path)


def plot_deciles(deciles: pd.DataFrame, path: Path) -> None:
    """Показать несуммарный uplift по децилям отдельно для каждой модели."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, (name, frame) in zip(axes.flat, deciles.groupby("model", sort=False)):
        ax.bar(frame["decile"], frame["uplift"] * 100, color="#247c91")
        ax.errorbar(frame["decile"], frame["uplift"] * 100,
                    yerr=(frame["high"] - frame["uplift"]) * 100,
                    fmt="none", color="#243747", capsize=2)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Дециль: 1 — наибольший прогноз uplift")
        ax.set_ylabel("Разница долей визитов, п.п.")
    fig.suptitle("Test · uplift по децилям и приближённые 95%-интервалы")
    save_figure(path)


def plot_matrix(p0: np.ndarray, p1: np.ndarray, threshold: float,
                path: Path) -> pd.DataFrame:
    """Показать четыре прокси-сегмента по p0/p1 T-Learner, не истинные типы клиентов."""
    labels = np.array([["Lost causes", "Persuadables"], ["Sleeping dogs", "Sure things"]])
    counts = np.zeros((2, 2), dtype=int)
    np.add.at(counts, ((p0 >= threshold).astype(int), (p1 >= threshold).astype(int)), 1)
    annotation = np.array([[f"{labels[i,j]}\n{counts[i,j]:,} ({counts[i,j]/len(p0):.1%})"
                             for j in range(2)] for i in range(2)])
    plt.figure(figsize=(8, 5))
    sns.heatmap(counts, annot=annotation, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Низкая p1", "Высокая p1"], yticklabels=["Низкая p0", "Высокая p0"])
    plt.title(f"Прокси-сегменты T-Learner · порог {threshold:.0%}\nНе наблюдаемые причинные типы")
    plt.xlabel("Вероятность визита с SMS (p1)")
    plt.ylabel("Вероятность визита без SMS (p0)")
    save_figure(path)
    return pd.DataFrame({"segment": labels.ravel(), "clients": counts.ravel(),
                         "share": counts.ravel()/len(p0), "threshold": threshold})


def plot_importance(importance: pd.DataFrame, path: Path) -> None:
    """Показать 15 наибольших значений важности с разбросом перестановок."""
    top = importance.head(15).iloc[::-1]
    plt.figure(figsize=(11, 7))
    plt.barh(top["feature"], top["importance"], xerr=top["std"], color="#247c91")
    plt.axvline(0, color="black", lw=0.8)
    plt.title("Важность признаков победителя · подвыборка test")
    plt.xlabel("Падение нормированного Qini AUC при перестановке")
    save_figure(path)
