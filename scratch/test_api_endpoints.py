import io
import os
import sys

import pandas as pd
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from main import app  # noqa: E402


client = TestClient(app)


def test_api_upload():
    print("Testing /api/upload endpoint...")
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=24, freq="W"),
        "Sales": [100 + i * 8 for i in range(24)],
        "Cost": [60 + i * 5 for i in range(24)],
        "Category": ["A", "B", "C"] * 8,
    })
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    response = client.post(
        "/api/upload",
        files={"files": ("test.csv", csv_buf, "text/csv")},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert "dataset_id" in data
    assert "data" in data
    assert "columns" in data
    assert "row_count" in data
    assert "charts" not in data
    print("OK /api/upload contract passed")
    return data["dataset_id"]


def test_core_analytics(dataset_id):
    print(f"Testing /api/analytics/core/{dataset_id} endpoint...")
    response = client.get(f"/api/analytics/core/{dataset_id}")
    assert response.status_code == 200, response.text
    data = response.json()
    assert "kpis" in data
    assert "charts" in data
    assert "statistical_tests" in data
    print("OK /api/analytics/core contract passed")


def test_api_forecast(dataset_id):
    print(f"Testing /api/ml/forecast/{dataset_id} endpoint...")
    response = client.get(f"/api/ml/forecast/{dataset_id}")
    assert response.status_code == 200, response.text
    data = response.json()
    assert "charts" in data
    print("OK /api/ml/forecast contract passed")


def test_api_cluster(dataset_id):
    print(f"Testing /api/ml/cluster/{dataset_id} endpoint...")
    response = client.post(
        f"/api/ml/cluster/{dataset_id}",
        json={"selected_cols": ["Sales", "Cost"]},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "charts" in data
    print("OK /api/ml/cluster contract passed")


if __name__ == "__main__":
    try:
        ds_id = test_api_upload()
        test_core_analytics(ds_id)
        test_api_forecast(ds_id)
        test_api_cluster(ds_id)
        print("\nAll integration tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
