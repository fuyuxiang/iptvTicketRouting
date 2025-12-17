from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config import abs_path, config_path_from_env, load_config
from .logging_conf import get_logger, setup_logging
from .model import TicketClassifier
from .routing import TicketRouter
from .runtime_metrics import LATENCY, REQUESTS


class PredictRequest(BaseModel):
    ticket_id: str = Field(default="")
    text: str = Field(default="")


class PredictResponse(BaseModel):
    ticket_id: str
    label: str
    confidence: float
    dept: str
    queue: str
    auto_routed: bool
    # optional debug fields
    classes: list[str]
    proba: list[float]


def create_app(config_path: Optional[str] = None) -> FastAPI:
    cfg_path = config_path or config_path_from_env()
    cfg = load_config(cfg_path)

    setup_logging(cfg.get("logging", {}))
    log = get_logger(__name__)

    model_dir = abs_path(cfg, cfg["model"]["model_dir"])
    model_path = str(Path(model_dir) / cfg["model"]["model_file"])
    route_map_path = str(Path(model_dir) / cfg["model"]["route_map_file"])

    if not Path(model_path).exists():
        raise RuntimeError(
            f"Model file not found: {model_path}. "
            "Run: python scripts/train.py --config configs/config.yaml"
        )

    clf = TicketClassifier().load(model_path)

    route_map: Dict[str, str] = {}
    if Path(route_map_path).exists():
        route_map = TicketClassifier.load_json(route_map_path)

    router = TicketRouter(
        route_map=route_map,
        default_queue=cfg["routing"]["default_queue"],
        dept_to_queue=cfg["routing"].get("dept_to_queue", {}),
        threshold_auto_route=cfg["model"].get("threshold_auto_route", 0.65),
    )

    app = FastAPI(title="IPTV Ticket Router", version="1.0.0")

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        endpoint = request.url.path
        start = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            dur = time.perf_counter() - start
            LATENCY.labels(endpoint=endpoint).observe(dur)
            # status might not exist if exception before response; guard
            try:
                REQUESTS.labels(endpoint=endpoint, status=status).inc()
            except Exception:
                REQUESTS.labels(endpoint=endpoint, status="500").inc()

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "ts": int(time.time())}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        text = req.text or ""
        if not text.strip():
            raise HTTPException(status_code=400, detail="text is empty")

        proba = clf.predict_proba([text])[0]
        classes = clf.classes_()
        if not classes:
            raise HTTPException(status_code=500, detail="model not loaded")

        best_idx = int(max(range(len(proba)), key=lambda i: proba[i]))
        pred_label = str(classes[best_idx])
        pred_conf = float(proba[best_idx])

        decision = router.route(pred_label, pred_conf)

        log.info(
            "predict",
            extra={
                "ticket_id": req.ticket_id,
                "label": pred_label,
                "confidence": round(pred_conf, 6),
                "dept": decision.dept,
                "queue": decision.queue,
                "auto_routed": decision.auto_routed,
            },
        )

        return PredictResponse(
            ticket_id=req.ticket_id,
            label=pred_label,
            confidence=round(pred_conf, 6),
            dept=decision.dept,
            queue=decision.queue,
            auto_routed=bool(decision.auto_routed),
            classes=[str(x) for x in classes],
            proba=[round(float(x), 6) for x in proba],
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception):
        log.exception("unhandled_exception", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "internal_error"})

    log.info("app_started", extra={"config": cfg_path, "model_path": model_path})
    return app
