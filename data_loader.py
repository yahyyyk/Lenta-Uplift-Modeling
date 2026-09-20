"""Загрузка Lenta, аудит исходных данных и препроцессинг без утечки."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklift.datasets import fetch_lenta

LOGGER = logging.getLogger(__name__)
SEED = 42


def load_data(cache: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Загрузить исходный датасет; однозначно привести treatment к 0/1."""
    try:
        dataset = fetch_lenta(data_home=str(cache.resolve()))
        x = dataset["data"].copy()
        y = pd.to_numeric(dataset["target"], errors="raise")
        raw = pd.Series(dataset["treatment"])
        if set(raw.dropna().unique()).issubset({0, 1}):
            t = pd.to_numeric(raw)
        else:
            t = raw.astype("string").str.strip().str.lower().map(
                {"control": 0, "test": 1, "treatment": 1, "0": 0, "1": 1}
            )
        if t.isna().any() or y.isna().any():
            raise ValueError("Неизвестные группы или пропуски в treatment/target")
        if set(t.unique()) != {0, 1} or set(y.unique()) != {0, 1}:
            raise ValueError("Ожидаются обе группы и бинарный target")
        if not (len(x) == len(t) == len(y)):
            raise ValueError("Длины X, treatment и target не совпадают")
        return x.reset_index(drop=True), y.to_numpy(dtype=int), t.to_numpy(dtype=int)
    except Exception as exc:
        LOGGER.exception("Не удалось загрузить Lenta из кеша %s", cache)
        raise RuntimeError(
            "Загрузка Lenta не удалась. Проверьте интернет, доступность источника "
            "и свободное место. При повреждении кеша задайте новый --cache."
        ) from exc


def eda(x: pd.DataFrame, y: np.ndarray, t: np.ndarray, out: Path) -> None:
    """Сохранить описательный аудит; его результаты не управляют обучением."""
    pd.DataFrame({"dtype": x.dtypes.astype(str),
                  "missing_count": x.isna().sum(),
                  "missing_share": x.isna().mean()}).to_csv(out / "eda_columns.csv")
    summary = pd.DataFrame({"treatment": t, "target": y}).groupby("treatment").agg(
        clients=("target", "size"), visits=("target", "sum"), response_rate=("target", "mean")
    )
    summary["share"] = summary["clients"] / len(x)
    summary.to_csv(out / "eda_groups.csv")
    pd.Series(y).value_counts().sort_index().to_csv(out / "eda_target.csv")
    LOGGER.info("EDA: %d клиентов, %d колонок; target=1: %.4f\n%s",
                len(x), x.shape[1], y.mean(), summary.to_string())
    LOGGER.info("Типы колонок: %s; пропуски сохранены в eda_columns.csv",
                x.dtypes.astype(str).value_counts().to_dict())


def clean_features(x: pd.DataFrame) -> pd.DataFrame:
    """Удалить идентификаторы/исходы и привести категориальные поля без обучения."""
    excluded = {"cardholder", "customer_id", "client_id", "id", "group",
                "treatment", "target", "response_att"}
    x = x.loc[:, [c for c in x if c.lower() not in excluded
                 and not c.lower().startswith("unnamed:")]].copy()
    for col in x:
        if col in {"gender", "main_format"} or not pd.api.types.is_numeric_dtype(x[col]):
            x[col] = x[col].astype("string").astype(object).where(x[col].notna(), np.nan)
        else:
            x[col] = x[col].replace([np.inf, -np.inf], np.nan)
    return x


def make_preprocessor(train: pd.DataFrame) -> ColumnTransformer:
    """Определить схему по train; медианы, масштаб и OHE обучаются вызовом fit."""
    num = train.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in train if c not in num]
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", drop="first",
                                 sparse_output=False, dtype=np.float32)),
    ])
    return ColumnTransformer([("num", numeric, num), ("cat", categorical, cat)],
                             sparse_threshold=0, verbose_feature_names_out=False)
