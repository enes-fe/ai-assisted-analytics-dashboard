"""
Utilities for forecast result interpretation.
"""

from collections.abc import Iterable, Mapping
import math
from typing import Any


def _coerce_forecast_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("forecast", "prediction", "predicted", "value", "y"):
            if key in value:
                value = value[key]
                break

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def getForecastDirection(forecastValues: Iterable[Any]) -> str:
    values = [
        value
        for value in (_coerce_forecast_value(item) for item in forecastValues)
        if value is not None
    ]

    if len(values) < 2:
        return "stable"

    first = values[0]
    last = values[-1]
    baseline = max(abs(first), abs(sum(values) / len(values)), 1.0)
    pct_change = (last - first) / baseline

    if abs(pct_change) >= 0.02:
        return "upward" if pct_change > 0 else "downward"

    n = len(values)
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    denominator = sum((idx - mean_x) ** 2 for idx in range(n))
    if denominator <= 0:
        return "stable"

    slope = sum((idx - mean_x) * (value - mean_y) for idx, value in enumerate(values)) / denominator
    projected_change = slope * (n - 1)
    slope_ratio = projected_change / baseline

    if abs(slope_ratio) >= 0.01:
        return "upward" if slope_ratio > 0 else "downward"
    return "stable"


get_forecast_direction = getForecastDirection
