"""Проверки утечки, пограничных случаев и арифметики бизнес-оценки."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from business import business_summary
from data_loader import clean_features, make_preprocessor
from features import FeatureEngineer
from models import evaluate, ranked_indices


class PipelineTests(unittest.TestCase):
    """Проверять контракты на малых данных без предположений об истинном CATE."""

    def test_no_test_statistics(self) -> None:
        """Экстремальные значения test не меняют медианы или схему признаков."""
        train = clean_features(pd.DataFrame({"k_var_a": [1., 3., np.nan],
            "gender": ["M", "F", None], "main_format": [0, 1, 0],
            "response_sms": [0.1, 0.2, 0.3], "response_viber": [0.2, 0.3, 0.4]}))
        test = clean_features(pd.DataFrame({"k_var_a": [np.nan, 9999.],
            "gender": [None, "new"], "main_format": [2, 0],
            "response_sms": [0.1, 0.2], "response_viber": [0.3, 0.4]}))
        fe = FeatureEngineer().fit(train)
        medians = fe.medians_.copy()
        a, b = fe.transform(train), fe.transform(test)
        self.assertEqual(b.iloc[0]["purchase_variability"], 2.)
        pd.testing.assert_series_equal(medians, fe.medians_)
        prep = make_preprocessor(a).fit(a)
        self.assertTrue(np.isfinite(prep.transform(b)).all())
        self.assertNotIn("avg_check", fe.descriptions_)
        self.assertIn("unknown", prep.named_transformers_["cat"].named_steps["encode"].categories_[0])

    def test_invalid_denominators(self) -> None:
        """Нулевой счётчик транзакций и promo=-1 не создают бесконечности."""
        x = pd.DataFrame({"total_spend": [100., 200.], "num_transactions": [0., 2.],
                          "promo_share_15d": [-1., 0.5]})
        transformed = FeatureEngineer().fit_transform(x)
        self.assertTrue(np.isnan(transformed.loc[0, "avg_check"]))
        self.assertEqual(transformed.loc[1, "avg_check"], 100.)
        self.assertTrue(np.isnan(transformed.loc[0, "promo_dependency"]))

    def test_business_and_ties(self) -> None:
        """Суммы визитов и экономия согласованы; равенства не зависят от исходов."""
        pred = np.linspace(-0.03, 0.06, 100)
        t, y = np.tile([0, 1], 50), np.tile([0, 1, 1, 0], 25)
        report = business_summary(pred, y, t, sms_cost=2., visit_margin=500.)
        ix = ranked_indices(pred)[:30]
        self.assertEqual(report["sms_saved"], 70)
        self.assertAlmostEqual(report["predicted_visits_top30"], pred[ix].sum())
        self.assertAlmostEqual(report["predicted_visits_all"], pred.sum())
        self.assertAlmostEqual(report["predicted_profit_difference"],
                               140. - report["predicted_visits_lost"] * 500.)
        np.testing.assert_array_equal(ranked_indices(np.ones(100)), ranked_indices(np.ones(100)))
        metrics, deciles = evaluate(y, t, {"constant": np.ones(100)})
        self.assertAlmostEqual(metrics.iloc[0]["Qini AUC"], 0.)
        self.assertEqual(deciles["clients"].sum(), 100)


if __name__ == "__main__":
    unittest.main()
