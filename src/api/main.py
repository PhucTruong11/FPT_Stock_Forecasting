"""FastAPI service for the frozen FPT research experiment."""

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.data.contracts import BASE_DIR
from src.models.inference import ModelService


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
    model: str = "LSTM"
    price_basis: Literal["provider_adjusted_ohlc"] = "provider_adjusted_ohlc"
    observations: list[Observation] | None = Field(default=None, min_length=51, max_length=5000)


def create_app(root: Path = BASE_DIR, service_factory=ModelService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service_factory(root)
        yield

    app = FastAPI(title="FPT Forecasting API", version="2.0.0", lifespan=lifespan,
                  description="Research demo using adjusted historical prices; not a live trading service.")

    @app.get("/health")
    def health():
        service = app.state.service
        return {"status": "ready", "model_version": service.manifest["run_id"],
                "data_as_of": str(service.features["Date"].iloc[-1].date()),
                "default_demo_model": service.default_model}

    @app.get("/api/v1/models")
    def models():
        service = app.state.service
        return {"models": list(service.manifest["models"]), "default_demo_model": service.default_model,
                "validation_selected_model": service.manifest["selection"]["selected_model"]}

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
            return app.state.service.predict(request.model, frame)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
