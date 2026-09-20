"""Экономика политики топ-30% и осторожная интерпретация прокси-оценок."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models import group_effect, ranked_indices


def business_summary(pred: np.ndarray, y: np.ndarray, t: np.ndarray,
                     sms_cost: float | None = None,
                     visit_margin: float | None = None) -> dict[str, object]:
    """Сравнить политики на одной test-популяции относительно «никому не писать»."""
    n = len(pred)
    k = int(np.ceil(0.3 * n))
    top = ranked_indices(pred)[:k]
    observed_top = group_effect(y[top], t[top])
    observed_all = group_effect(y, t)
    # Это требуемая модельная оценка, а не число наблюдавшихся причинных визитов.
    visits_top = float(pred[top].mean() * k)
    visits_all = float(pred.mean() * n)
    report = {
        "clients": n, "sms_top30": k, "sms_all": n, "sms_saved": n-k,
        "sms_saved_share": (n-k)/n,
        "predicted_visits_top30": visits_top, "predicted_visits_all": visits_all,
        "predicted_visits_lost": visits_all-visits_top,
        "observed_top30": observed_top, "observed_all": observed_all,
        "observed_incremental_visits_top30": float(observed_top["uplift"] * k),
        "observed_incremental_visits_all": float(observed_all["uplift"] * n),
        "observed_visits_lost": float(observed_all["uplift"] * n - observed_top["uplift"] * k),
        "sms_cost": sms_cost, "visit_margin": visit_margin,
    }
    if sms_cost is not None:
        report["money_saved_sms"] = (n-k) * sms_cost
    if sms_cost is not None and visit_margin is not None:
        report["predicted_profit_top30"] = visits_top * visit_margin - k * sms_cost
        report["predicted_profit_all"] = visits_all * visit_margin - n * sms_cost
        report["predicted_profit_difference"] = (
            report["predicted_profit_top30"] - report["predicted_profit_all"]
        )
        report["break_even_uplift"] = sms_cost / visit_margin
    return report


def interpretation(feature: str) -> str:
    """Дать бизнес-смысл признака без приписывания причинности или направления."""
    families = {
        "purchase_variability": "Сводная нестабильность покупательского поведения.",
        "promo_dependency": "Ориентация на промотовары в последние 15 дней.",
        "communication_response": "Историческая восприимчивость к SMS и Viber.",
        "avg_check": "Расходы на один чек; показатель масштаба покупки.",
        "response_": "Отклик на прошлые коммуникации, доступный до текущей кампании.",
        "k_var_": "Вариативность покупок, скидок или интервалов визитов.",
        "cheque_count_": "Частота покупок в указанной товарной группе и периоде.",
        "crazy_purchases_": "Участие в прошлых специальных промоакциях.",
        "sale_sum_": "Объём расходов в товарной группе за указанный период.",
        "sale_count_": "Количество купленных товаров в группе.",
        "disc_": "Объём полученных скидок.",
        "promo_": "Доля промопокупок.",
        "food_share": "Доля продуктов питания в корзине.",
        "mean_discount": "Средняя глубина скидок.",
        "stdev_": "Разброс скидок или интервалов между визитами.",
        "perdelta_": "Изменение частоты визитов между временными окнами.",
        "months_from_register": "Длительность отношений с программой лояльности.",
        "age": "Возрастной профиль клиента.",
        "gender": "Демографический профиль; связь не означает причинный эффект пола.",
        "children": "Состав домохозяйства.",
        "main_format": "Предпочитаемый формат магазина.",
    }
    return next((text for prefix, text in families.items() if feature.startswith(prefix)),
                "Историческая характеристика покупательского поведения.")


def recommendation(name: str, report: dict[str, object], importance: pd.DataFrame) -> str:
    """Сформировать готовый отчёт; не объявлять test-победителя доказанным в production."""
    top = report["observed_top30"]
    lines = [f"# Результаты: {name}", "",
        f"Test: {report['clients']:,} клиентов. SMS топ-30%: {report['sms_top30']:,}; "
        f"экономия: {report['sms_saved']:,} SMS ({report['sms_saved_share']:.1%}).",
        f"Прогноз инкрементальных визитов: топ-30% {report['predicted_visits_top30']:.1f}, "
        f"всем {report['predicted_visits_all']:.1f}; разница «всем − топ» "
        f"{report['predicted_visits_lost']:.1f}. Отрицательная разница означает выигрыш таргетинга.",
        f"Наблюдаемая разница долей в топ-30%: {top['uplift']:.3%}, "
        f"приближённый 95%-интервал [{top['low']:.3%}; {top['high']:.3%}].",
        f"Оценка инкрементальных визитов по treatment/control: топ-30% "
        f"{report['observed_incremental_visits_top30']:.1f}, всем "
        f"{report['observed_incremental_visits_all']:.1f}; разница "
        f"{report['observed_visits_lost']:.1f}. Расхождение с модельной оценкой "
        "требует проверки калибровки и устойчивости политики.",
        ""]
    if top["low"] > 0:
        lines.append("Топ-30% — кандидат для отдельного рандомизированного пилота таргетинга.")
    else:
        lines.append("Положительный эффект топ-30% пока недостаточно убедителен; масштабировать рано.")
    if "predicted_profit_difference" in report:
        lines.append(f"При заданных стоимости SMS и марже изменение прогнозной прибыли "
                     f"относительно рассылки всем: {report['predicted_profit_difference']:.2f}. "
                     f"Порог окупаемости uplift: {report['break_even_uplift']:.4%}.")
    else:
        lines.append("Денежная прибыль не определена: укажите стоимость SMS и маржу за визит. "
                     "Разница прибыли = сэкономленные SMS × стоимость − потерянные визиты × маржа.")
    lines.extend(["", "Победитель выбран по test-Qini для сравнительного анализа. "
                  "Его метрики и интервалы после выбора оптимистичны и не учитывают выбор модели. "
                  "Для внедрения нужен новый holdout/эксперимент. Qini измеряет ранжирование, "
                  "не калибровку CATE. Прогнозные визиты не равны доказанному эффекту.",
                  "Порог 30% задан заранее и не оптимизировался. Причинная трактовка требует "
                  "рандомизации рассылки, отсутствия вмешательства между клиентами и признаков "
                  "после воздействия. Неизвестный механизм назначения требует отдельного аудита.",
                  "", "## Топ-15 признаков", "",
                  "Падение Qini при перестановке на подвыборке test. Это важность для модели; "
                  "она не показывает знак влияния и не доказывает причинность. "
                  "Коррелирующие исходные и составные признаки могут делить важность.", ""])
    for row in importance.head(15).itertuples():
        lines.append(f"- `{row.feature}`: {row.importance:.5f} ± {row.std:.5f}. "
                     f"{interpretation(row.feature)}")
    return "\n".join(lines) + "\n"
