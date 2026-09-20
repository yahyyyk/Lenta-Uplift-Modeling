"""Четыре uplift-модели, единые holdout-метрики и permutation importance."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklift.metrics import qini_auc_score
from xgboost import XGBRegressor, XGBClassifier

from data_loader import SEED

LOGGER = logging.getLogger(__name__)


def booster(outcome: bool = True) -> XGBRegressor:
    """Создать небольшой XGBoost; вероятности ограничены логистической целью."""
    return XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                        objective="reg:logistic" if outcome else "reg:squarederror",
                        tree_method="hist", random_state=SEED, n_jobs=2)


class PortableX:
    """Прозрачный X-Learner на XGBoost для окружений без causalml."""

    def fit(self, x: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> PortableX:
        """Обучить два исхода и две регрессии псевдоэффектов только на train."""
        self.p_ = float(treatment.mean())
        c, t = treatment == 0, treatment == 1
        self.mu0_, self.mu1_ = booster(), booster()
        self.mu0_.fit(x[c], y[c])
        self.mu1_.fit(x[t], y[t])
        self.tau0_, self.tau1_ = booster(False), booster(False)
        self.tau0_.fit(x[c], self.mu1_.predict(x[c]) - y[c])
        self.tau1_.fit(x[t], y[t] - self.mu0_.predict(x[t]))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Усреднить эффекты с постоянной train-propensity рандомизированной кампании."""
        return self.p_ * self.tau0_.predict(x) + (1 - self.p_) * self.tau1_.predict(x)


@dataclass
class UpliftModel:
    """Единый адаптер сигнатур causalml и альтернативных моделей."""

    estimator: Any
    kind: str
    propensity: float
    constant: float | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Вернуть одномерный эффект; устранить шум округления константного S-Learner."""
        if self.constant is not None:
            return np.full(len(x), self.constant)
        if self.kind == "causal-x":
            pred = self.estimator.predict(x, p=np.full(len(x), self.propensity), verbose=False)
        elif self.kind == "causal-meta":
            pred = self.estimator.predict(x, verbose=False)
        else:
            pred = self.estimator.predict(x)
        result = np.asarray(pred, dtype=float).reshape(-1)
        if len(result) != len(x) or not np.isfinite(result).all():
            raise ValueError("Некорректный размер или нечисловой прогноз uplift")
        return result


def train_models(x: np.ndarray, y: np.ndarray, t: np.ndarray,
                 backend: str = "causalml", trees: int = 50) -> dict[str, UpliftModel]:
    """Обучить модели с фиксированными параметрами; test сюда не передаётся."""
    p = float(t.mean())
    if backend == "causalml":
        try:
            from causalml.inference.meta import LRSRegressor, BaseTRegressor, BaseXRegressor
            from causalml.inference.tree import UpliftRandomForestClassifier
        except (ImportError, OSError, ValueError) as exc:
            raise RuntimeError("causalml недоступен; установите requirements-core.txt "
                               "и явно запустите --backend sklift") from exc
        specs = {
            "S-Learner (LRS)": (LRSRegressor(control_name=0), "causal-meta"),
            "T-Learner": (BaseTRegressor(learner=booster(), control_name=0), "causal-meta"),
            "X-Learner (XGBoost)": (BaseXRegressor(
                control_outcome_learner=booster(), treatment_outcome_learner=booster(),
                control_effect_learner=booster(False), treatment_effect_learner=booster(False),
                control_name=0), "causal-x"),
            "Uplift Random Forest": (UpliftRandomForestClassifier(
                control_name="control", n_estimators=trees, max_depth=5,
                max_features=min(10, x.shape[1]), min_samples_leaf=500,
                min_samples_treatment=100, n_reg=100, evaluationFunction="KL",
                random_state=SEED, n_jobs=2), "causal-forest"),
        }
    else:
        from sklift.models import SoloModel, TwoModels
        classifier_params = {**booster().get_params(), "objective": "binary:logistic"}
        specs = {
            "S-Learner (sklift logistic)": (SoloModel(LogisticRegression(
                C=0.1, max_iter=500, random_state=SEED)), "portable"),
            "T-Learner": (TwoModels(XGBClassifier(**classifier_params),
                                   XGBClassifier(**classifier_params), method="vanilla"), "portable"),
            "X-Learner (portable XGBoost)": (PortableX(), "portable"),
            "Transformed-outcome RF (fallback)": (RandomForestRegressor(
                n_estimators=trees, max_depth=6, min_samples_leaf=500,
                max_features=0.5, n_jobs=2, random_state=SEED), "transformed"),
        }
        LOGGER.warning("Fallback: четвёртая модель — transformed-outcome RF, не uplift forest")
    fitted = {}
    for name, (estimator, kind) in specs.items():
        LOGGER.info("Обучение: %s", name)
        if kind == "causal-x":
            estimator.fit(x, treatment=t, y=y, p=np.full(len(x), p))
        elif kind == "causal-forest":
            estimator.fit(x, treatment=np.where(t == 1, "sms", "control"), y=y)
        elif kind == "transformed":
            # Эта формула учитывает неравные доли treatment/control.
            estimator.fit(x, y * (t / p - (1 - t) / (1 - p)))
        else:
            estimator.fit(x, treatment=t, y=y)
        adapter = UpliftModel(estimator, kind, p)
        if name == "S-Learner (LRS)":
            adapter.constant = float(adapter.predict(x[:100]).mean())
        fitted[name] = adapter
    return fitted


def ranked_indices(pred: np.ndarray) -> np.ndarray:
    """Отсортировать uplift; равенства разбить воспроизводимо и независимо от y/t."""
    return np.lexsort((np.random.default_rng(SEED).random(len(pred)), -pred))


def group_effect(y: np.ndarray, t: np.ndarray) -> dict[str, float | int]:
    """Оценить разницу долей визитов и приближённый 95%-интервал на holdout."""
    nt, nc = int(t.sum()), int((1 - t).sum())
    if min(nt, nc) == 0:
        return {"uplift": float("nan"), "low": float("nan"), "high": float("nan"),
                "n_treatment": nt, "n_control": nc}
    pt, pc = float(y[t == 1].mean()), float(y[t == 0].mean())
    effect = pt - pc
    se = np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc)
    return {"uplift": effect, "low": effect - 1.96 * se, "high": effect + 1.96 * se,
            "n_treatment": nt, "n_control": nc}


def evaluate(y: np.ndarray, t: np.ndarray,
             predictions: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Посчитать нормированный Qini AUC и разницу долей в общих топ-30% на test."""
    rows, deciles = [], []
    for name, pred in predictions.items():
        order = ranked_indices(pred)
        top = order[:int(np.ceil(0.3 * len(y)))]
        effect = group_effect(y[top], t[top])
        rows.append({"model": name, "Qini AUC": qini_auc_score(y, pred, t),
                     "Uplift@30%": effect["uplift"], "top30_low": effect["low"],
                     "top30_high": effect["high"]})
        for i, ix in enumerate(np.array_split(order, 10), 1):
            deciles.append({"model": name, "decile": i, "clients": len(ix),
                           **group_effect(y[ix], t[ix])})
    return (pd.DataFrame(rows).sort_values("Qini AUC", ascending=False),
            pd.DataFrame(deciles))


def permutation_importance(model: UpliftModel, preprocessor: Any,
                           x: pd.DataFrame, y: np.ndarray, t: np.ndarray,
                           max_rows: int = 5000, repeats: int = 3) -> pd.DataFrame:
    """Измерить падение test-Qini при перестановке признака до OHE; не отбирать фичи."""
    ix = np.arange(len(x))
    if len(ix) > max_rows:
        ix, _ = train_test_split(ix, train_size=max_rows, stratify=2*t+y, random_state=SEED)
    sample, ys, ts = x.iloc[ix].copy(), y[ix], t[ix]
    baseline = qini_auc_score(ys, model.predict(preprocessor.transform(sample)), ts)
    rng, rows = np.random.default_rng(SEED), []
    for number, col in enumerate(sample, 1):
        deltas = []
        for _ in range(repeats):
            changed = sample.copy()
            changed[col] = rng.permutation(changed[col].to_numpy())
            pred = model.predict(preprocessor.transform(changed))
            deltas.append(baseline - qini_auc_score(ys, pred, ts))
        rows.append({"feature": col, "importance": np.mean(deltas),
                     "std": np.std(deltas), "test_rows": len(ix)})
        if number % 25 == 0:
            LOGGER.info("Permutation importance: %d/%d", number, sample.shape[1])
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def outcome_probabilities(t_model: UpliftModel, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Получить p0/p1 от T-Learner для отдельной иллюстративной сегментации."""
    estimator = t_model.estimator
    if t_model.kind == "causal-meta":
        p0 = estimator.models_c[1].predict(x)
        p1 = estimator.models_t[1].predict(x)
    else:
        p0 = estimator.estimator_ctrl.predict_proba(x)[:, 1]
        p1 = estimator.estimator_trmnt.predict_proba(x)[:, 1]
    return np.asarray(p0), np.asarray(p1)
