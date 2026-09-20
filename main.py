"""Запуск воспроизводимого end-to-end анализа Lenta с независимым test."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from importlib.metadata import version
from pathlib import Path

# Локальный кеш позволяет запускать matplotlib в ограниченном окружении.
os.environ.setdefault("MPLCONFIGDIR", str(Path("work/mplconfig").resolve()))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from data_loader import SEED, clean_features, eda, load_data, make_preprocessor
from features import FeatureEngineer
from models import (evaluate, outcome_probabilities, permutation_importance,
                    ranked_indices, train_models)
from business import business_summary, recommendation
from plots import plot_deciles, plot_importance, plot_matrix, plot_qini

LOGGER = logging.getLogger(__name__)


def arguments() -> argparse.Namespace:
    """Прочитать настройки запуска; значения не подбираются по test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("work/data"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--backend", choices=["causalml", "sklift"], default="causalml")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Только быстрый пример на стратифицированной подвыборке")
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--importance-rows", type=int, default=5000)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--sms-cost", type=float, default=None)
    parser.add_argument("--visit-margin", type=float, default=None)
    parser.add_argument("--segment-threshold", type=float, default=0.1,
                        help="Иллюстративный порог вероятности p0/p1; не порог рассылки")
    parser.add_argument("--save-models", action="store_true",
                        help="Сохранить обученные модели; LRS может занимать много места")
    args = parser.parse_args()
    if not 0 < args.test_size < 1 or not 0 < args.segment_threshold < 1:
        parser.error("test-size и segment-threshold должны лежать между 0 и 1")
    if min(args.trees, args.importance_rows, args.permutation_repeats) < 1:
        parser.error("Число деревьев, строк и повторов должно быть положительным")
    if args.sample_size is not None and args.sample_size < 1000:
        parser.error("Для примера требуется не менее 1000 клиентов")
    if args.sms_cost is not None and (not np.isfinite(args.sms_cost) or args.sms_cost < 0):
        parser.error("Стоимость SMS должна быть неотрицательной и конечной")
    if args.visit_margin is not None and (not np.isfinite(args.visit_margin) or args.visit_margin <= 0):
        parser.error("Маржа за визит должна быть положительной и конечной")
    return args


def write_json(path: Path, value: object) -> None:
    """Записать читаемый UTF-8 JSON; пути представить строками."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    """Обучить на train, один раз сравнить на test и сохранить отчёты/графики."""
    np.random.seed(SEED)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out / "run.log", encoding="utf-8")], force=True)
    write_json(out / "config.json", {**vars(args), "random_state": SEED,
                                    "python": sys.version, "packages": {
                                        name: version(name) for name in ["numpy", "pandas", "scikit-learn",
                                                                         "xgboost", "scikit-uplift"]}})
    x, y, t = load_data(args.cache)
    original_rows = len(x)
    if args.sample_size is not None and args.sample_size < len(x):
        ix, _ = train_test_split(np.arange(len(x)), train_size=args.sample_size,
                                 stratify=2*t+y, random_state=SEED)
        x, y, t = x.iloc[ix].copy(), y[ix], t[ix]
        LOGGER.warning("Демонстрационный запуск: %d из %d клиентов", len(x), original_rows)
    eda(x, y, t, out)
    train, test = train_test_split(np.arange(len(x)), test_size=args.test_size,
                                  stratify=2*t+y, random_state=SEED)
    pd.DataFrame({"source_row": x.index[train]}).to_csv(out / "train_rows.csv", index=False)
    pd.DataFrame({"source_row": x.index[test]}).to_csv(out / "test_rows.csv", index=False)
    raw_train, raw_test = clean_features(x.iloc[train]), clean_features(x.iloc[test])
    del x
    engineer = FeatureEngineer().fit(raw_train)
    train_fe, test_fe = engineer.transform(raw_train), engineer.transform(raw_test)
    del raw_train, raw_test
    preprocessor = make_preprocessor(train_fe)
    x_train = preprocessor.fit_transform(train_fe).astype(np.float32)
    x_test = preprocessor.transform(test_fe).astype(np.float32)
    del train_fe
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("После препроцессинга остались NaN/Inf")
    write_json(out / "new_features.json", engineer.descriptions_)
    pd.Series(preprocessor.get_feature_names_out(), name="feature").to_csv(
        out / "encoded_features.csv", index=False)
    joblib.dump({"features": engineer, "preprocessor": preprocessor}, out / "preprocessing.joblib")
    LOGGER.info("Разбиение: train=%d, test=%d; признаков после OHE=%d",
                len(train), len(test), x_train.shape[1])
    fitted = train_models(x_train, y[train], t[train], args.backend, args.trees)
    del x_train
    predictions = {name: model.predict(x_test) for name, model in fitted.items()}
    table, deciles = evaluate(y[test], t[test], predictions)
    table.to_csv(out / "metrics.csv", index=False)
    deciles.to_csv(out / "deciles.csv", index=False)
    LOGGER.info("Метрики только на test:\n%s", table.to_string(index=False))
    best_name = str(table.iloc[0]["model"])
    best = fitted[best_name]
    LOGGER.warning("Победитель по test-Qini: %s. Для внедрения нужен новый holdout", best_name)
    importance = permutation_importance(best, preprocessor, test_fe, y[test], t[test],
                                        args.importance_rows, args.permutation_repeats)
    importance.to_csv(out / "feature_importance.csv", index=False)
    business = business_summary(predictions[best_name], y[test], t[test], args.sms_cost, args.visit_margin)
    write_json(out / "business.json", {"model": best_name, **business})
    report = recommendation(best_name, business, importance)
    (out / "report.md").write_text(report, encoding="utf-8")
    plot_qini(y[test], t[test], predictions, out / "qini.png")
    plot_deciles(deciles, out / "uplift_deciles.png")
    p0, p1 = outcome_probabilities(fitted["T-Learner"], x_test)
    segments = plot_matrix(p0, p1, args.segment_threshold, out / "uplift_matrix.png")
    segments.to_csv(out / "segments.csv", index=False)
    plot_importance(importance, out / "feature_importance.png")
    ranked = pd.DataFrame({"source_row": test_fe.index, "target": y[test],
                           "treatment": t[test], **predictions})
    order = ranked_indices(predictions[best_name])
    ranked = ranked.iloc[order].copy()
    ranked["selected_top30"] = np.arange(len(ranked)) < int(np.ceil(0.3 * len(ranked)))
    ranked.to_csv(out / "test_ranked.csv.gz", index=False, compression="gzip")
    if args.save_models:
        joblib.dump(fitted, out / "models.joblib", compress=3)
    LOGGER.info("%s\nРезультаты: %s", report, out)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    # Ограничение BLAS снижает расход ресурсов и колебания времени обучения.
    with threadpool_limits(limits=2):
        run(arguments())
