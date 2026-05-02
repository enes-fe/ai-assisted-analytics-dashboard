"""KPI regression verification script."""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'backend'))
os.chdir(os.path.join(ROOT, 'backend'))

import pandas as pd
import numpy as np

from services.kpi_engine import (
    calculate_kpis,
    MEAN_AGGREGATED_KEYWORDS,
    PERCENT_DISPLAY_KEYWORDS,
    _col_matches_keywords,
)

print("=== Constants check ===")
for kw in ("rating", "score", "xg", "avg", "average", "mean", "accuracy", "completion", "index", "satisfaction"):
    present = kw in MEAN_AGGREGATED_KEYWORDS
    status = "OK" if present else "MISSING"
    print(f"  {kw:15s} in MEAN_AGGREGATED_KEYWORDS: {status}")
    assert present, f"MISSING: {kw} not in MEAN_AGGREGATED_KEYWORDS"

for kw in ("percent", "percentage", "pct", "%"):
    present = kw in PERCENT_DISPLAY_KEYWORDS
    status = "OK" if present else "MISSING"
    print(f"  {kw:15s} in PERCENT_DISPLAY_KEYWORDS: {status}")
    assert present, f"MISSING: {kw} not in PERCENT_DISPLAY_KEYWORDS"

# Cross-sectional player dataset (no date column)
df = pd.DataFrame({
    'player':  ['A','B','C','D','E','F','G','H','I','J'],
    'goals':   [5,  3,  8,  2,  7,  4,  6,  1,  9,  3],
    'rating':  [7.2,6.8,8.1,6.5,7.9,7.0,8.3,5.9,8.5,6.7],
    'touches': [120,98,145,87,133,110,140,75,150,95],
    'assists': [2,  1,  4,  0,  3,  2,  3,  0,  5,  1],
})

print()
print("=== KPI regression test: goals, rating, touches ===")
kpis = calculate_kpis(df, selected_columns=['goals','rating','touches'])
for k in kpis:
    prefix = k['title'].split()[0]
    print(f"  column={k['column']:10s}  prefix={prefix:5s}  value={k['value']:>10s}  rawValue={k['rawValue']:.4f}  trend={k['trendDirection']}")

for k in kpis:
    col = k['column']
    if col == 'rating':
        assert k['title'].startswith('Avg'), f"FAIL: rating title should start with Avg, got: {k['title']!r}"
        assert k['rawValue'] < 20, f"FAIL: rating rawValue should be avg ~7.x (mean), got: {k['rawValue']}"
        assert k['trendDirection'] == 'neutral', f"FAIL: no date col, trend should be neutral, got: {k['trendDirection']}"
        print("  [PASS] rating: Avg prefix, realistic decimal, no trend arrow")
    if col in ('goals', 'touches'):
        assert k['title'].startswith('Total'), f"FAIL: {col} should be Total, got: {k['title']!r}"
        print(f"  [PASS] {col}: Total prefix (sum)")

print()
print("=== Column order test: rating, assists, touches ===")
kpis2 = calculate_kpis(df, selected_columns=['rating','assists','touches'])
cols_returned = [k['column'] for k in kpis2]
print(f"  Columns returned: {cols_returned}")
assert cols_returned == ['rating','assists','touches'], f"FAIL: order mismatch {cols_returned}"
print("  [PASS] Column order preserved")

print()
print("ALL CHECKS PASSED")
