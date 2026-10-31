"""Local FPT Ridge API. Historical experiments remain immutable."""

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.data.contracts import BASE_DIR
from src.models.local_demo import LocalRidgeService


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: Literal["FPT.VN"] = "FPT.VN"
    model: str | None = None
    price_basis: Literal["provider_adjusted_ohlc"] = "provider_adjusted_ohlc"
    observations: list[Observation] | None = Field(default=None, min_length=51, max_length=5000)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    horizon: Literal["today", "tomorrow", "week", "month"] = "today"


def create_app(root: Path = BASE_DIR, service_factory=LocalRidgeService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service_factory(root)
        yield

    app = FastAPI(title="FPT Forecasting API", version="3.0.0", lifespan=lifespan,
                  description="Research demo using adjusted historical prices; not a live trading service.")

    @app.get("/health")
    def health():
        service = app.state.service
        return {"status": "ready", "model_version": service.manifest["run_id"],
                "data_as_of": str(service.features["Date"].iloc[-1].date()),
                "default_demo_model": service.default_model,
                "mode": "local_frozen_snapshot", "automatic_retraining": False,
                "source": service.manifest.get("source", "legacy snapshot"),
                "training_target_end": service.manifest.get("training_target_end")}

    @app.get("/api/v1/models")
    def models():
        service = app.state.service
        return {"models": list(service.manifest["models"]), "default_demo_model": service.default_model,
                "validation_selected_model": (None if service.manifest.get("schema") == "fpt-local-ridge-v1"
                                              else service.manifest["selection"]["selected_model"]),
                "selection": service.manifest["selection"]}

    @app.get("/api/v1/fpt/history")
    def history(limit: int = Query(default=252, ge=1, le=2000)):
        frame = app.state.service.features[["Date", "Close", "Log_Return"]].tail(limit).copy()
        frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
        return {"ticker": "FPT.VN", "price_basis": "provider_adjusted_ohlc", "observations": frame.to_dict("records")}

    @app.get("/api/v1/metrics")
    def metrics():
        manifest = app.state.service.manifest
        return {"model_version": manifest["run_id"], "splits": manifest["splits"],
                "metrics": manifest["metrics"], "selection": manifest["selection"],
                "caveats": manifest["caveats"]}

    @app.post("/api/v1/predict")
    def predict(request: PredictRequest):
        frame = None
        if request.observations is not None:
            frame = pd.DataFrame([row.model_dump() for row in request.observations]).rename(columns={
                "date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        try:
            return app.state.service.predict(request.model or app.state.service.default_model, frame)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/backtest")
    def backtest(scope: Literal["full_2026", "jun_oct_2026"] = "full_2026",
                 month: str | None = Query(default=None, pattern=r"^2026-(0[1-9]|1[0-2])$")):
        try:
            return app.state.service.benchmark(scope, month)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/training-history")
    def training_history():
        import json
        path = Path(root) / "reports/training_history/study_timeline.json"
        return {"stages": json.loads(path.read_text(encoding="utf-8")),
                "documentation": "docs/TRAINING_HISTORY.md"}

    @app.post("/api/v1/forecast")
    def forecast_path(request: ForecastRequest):
        from src.models.forecast_path import forecast
        try:
            return forecast(app.state.service, request.horizon, request.model)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/fpt/ohlcv")
    def ohlcv(limit: int = Query(default=252, ge=1, le=2500)):
        from src.models.local_demo import records
        frame = app.state.service.raw.tail(limit).copy()
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
        return {"ticker": "FPT.VN", "price_basis": "provider_adjusted_ohlc", "observations": records(frame)}

    @app.get("/api/v1/data-quality")
    def data_quality():
        manifest = app.state.service.manifest
        return {"source": manifest.get("source"), "data_as_of": manifest.get("data_as_of"),
                "quality": manifest.get("quality", {}), "feature_columns": manifest.get("feature_columns", []),
                "point_in_time_certified": manifest.get("point_in_time_certified", False),
                "caveats": manifest["caveats"]}

    return app


app = create_app()
