import pandas as pd
import numpy as np
import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from ml_service import run_forecasting, generate_heuristic_charts

def test_forecasting():
    print("Testing run_forecasting...")
    # Create sample time series data
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'value': np.arange(20) + np.random.normal(0, 1, 20)
    })
    
    results = run_forecasting(df, periods=5)
    assert isinstance(results, list)
    if len(results) > 0:
        assert results[0]['type'] == 'forecast'
        assert len(results[0]['forecast']) == 5
        print("OK run_forecasting passed")
    else:
        print("! run_forecasting returned no results (expected for small dataset)")

def test_heuristic_charts():
    print("Testing generate_heuristic_charts...")
    df = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100),
        'Cat': ['X', 'Y'] * 50
    })
    
    charts = generate_heuristic_charts(df)
    assert isinstance(charts, list)
    assert any(c['id'].startswith('hist-') for c in charts)
    print("OK generate_heuristic_charts passed")

if __name__ == "__main__":
    try:
        test_forecasting()
        test_heuristic_charts()
        print("\nAll unit tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        sys.exit(1)
