"""Детерминированные признаки: схема и медианы исходных чисел берутся из train."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = logging.getLogger(__name__)


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Заполнить исходные числа и добавить признаки, не используя target/test."""

    def fit(self, x: pd.DataFrame, y: object = None) -> FeatureEngineer:
        """Запомнить доступные колонки и медианы исключительно обучающей выборки."""
        self.columns_ = x.columns.tolist()
        self.numeric_ = x.select_dtypes(include=np.number).columns.tolist()
        # У полностью пустой колонки медиана не определена; используем 0 явно.
        self.medians_ = x[self.numeric_].median().fillna(0)
        self.kvar_ = [c for c in self.numeric_ if c.startswith("k_var_")]
        candidates = {
            "purchase_variability": (self.kvar_, "Среднее k_var_*: нестабильность покупок; смешивает окна и категории."),
            "promo_dependency": (["promo_share_15d"], "p/(p+1): сжатая монотонная мера промо-зависимости."),
            "communication_response": (["response_sms", "response_viber"], "Сумма исторических откликов: восприимчивость к коммуникациям."),
            "avg_check": (["total_spend", "num_transactions"], "Расходы на транзакцию при совпадающих периодах агрегирования."),
        }
        self.descriptions_ = {}
        for name, (sources, reason) in candidates.items():
            if sources and all(c in self.numeric_ for c in sources):
                self.descriptions_[name] = {"sources": sources, "reason": reason}
            else:
                LOGGER.warning("Признак %s пропущен: нет необходимых числовых полей", name)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Применить зафиксированные правила построчно, без статистик test."""
        x = x.loc[:, self.columns_].copy()
        x.loc[:, self.numeric_] = x[self.numeric_].fillna(self.medians_)
        if "purchase_variability" in self.descriptions_:
            x["purchase_variability"] = x[self.kvar_].mean(axis=1)
        if "promo_dependency" in self.descriptions_:
            p = x["promo_share_15d"]
            x["promo_dependency"] = p / (p + 1).replace(0, np.nan)
        if "communication_response" in self.descriptions_:
            x["communication_response"] = x["response_sms"] + x["response_viber"]
        if "avg_check" in self.descriptions_:
            denominator = x["num_transactions"].where(x["num_transactions"] > 0)
            x["avg_check"] = x["total_spend"] / denominator
        return x.replace([np.inf, -np.inf], np.nan)
