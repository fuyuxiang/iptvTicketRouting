import os
from pathlib import Path

from fastapi.testclient import TestClient

from iptv_ticket_router.api import create_app


def ensure_model(tmp_path: Path):
    # Train a tiny model quickly using included sample data
    import subprocess

    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "config.yaml"
    # ensure model dir exists in working dir
    subprocess.check_call(["python", "scripts/train.py", "--config", str(cfg)], cwd=str(root))


def test_healthz_and_predict():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "config.yaml"
    ensure_model(root / "models")

    os.environ["IPTV_CONFIG"] = str(cfg)
    app = create_app(str(cfg))
    client = TestClient(app)

    r = client.get("/healthz")
    assert r.status_code == 200

    r = client.post("/predict", json={"ticket_id": "T1", "text": "机顶盒无信号，提示网络异常"})
    assert r.status_code == 200
    body = r.json()
    assert body["ticket_id"] == "T1"
    assert "label" in body
    assert "queue" in body
