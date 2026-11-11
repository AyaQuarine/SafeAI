"""Application locale de diagnostic. Le moteur reste utilisable sans cette interface."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from safeai.acquisition.cli import serialize_result
from safeai.acquisition.config import load_config
from safeai.acquisition.contracts import SourceConfig, TARGET_SAMPLE_RATE
from safeai.acquisition.errors import AcquisitionError, ConsentRequiredError
from safeai.acquisition.service import (
    AcquisitionService,
    downsample_envelope,
    run_file,
    run_synthetic,
)
from safeai.acquisition.signals import demo_meeting_signal
from safeai.acquisition.sources import SyntheticSource, list_input_devices
from safeai.acquisition.vad import EnergyVADDetector, build_vad

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"

_LIVE: dict[str, AcquisitionService] = {}


def create_app() -> FastAPI:
    app = FastAPI(
        title="SafeAI — Diagnostic d'acquisition",
        description="Interface locale du premier pilier. Système pédagogique, non médical, non certifié.",
        version="1.0.0",
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/common")
    def common_dashboard() -> FileResponse:
        """Visualisation en lecture seule des sorties JSON Common V1.0."""
        return FileResponse(STATIC_DIR / "common.html")

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, object]:
        return {
            "microphone_enabled": os.getenv("SAFEAI_DISABLE_MICROPHONE") != "1",
            "common_mode": "json_viewer_only",
            "offline_inference_from_browser": False,
        }

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "module": "acquisition", "sample_rate": TARGET_SAMPLE_RATE}

    @app.get("/api/devices")
    def devices() -> dict[str, object]:
        if os.getenv("SAFEAI_DISABLE_MICROPHONE") == "1":
            return {"devices": []}
        return {"devices": list_input_devices()}

    @app.post("/api/inspect/file")
    async def inspect_file(
        file: UploadFile = File(...),
        energy_vad: bool = Form(False),
    ) -> dict[str, object]:
        suffix = Path(file.filename or "upload.wav").suffix or ".wav"
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Fichier vide.")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        try:
            return _inspect_path(temp_path, energy_vad, source_label=file.filename or temp_path.name)
        except AcquisitionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            temp_path.unlink(missing_ok=True)

    @app.post("/api/inspect/demo")
    def inspect_demo() -> dict[str, object]:
        config = load_config()
        source = SyntheticSource(
            demo_meeting_signal(),
            session_id="session_demo",
            chunk_duration_s=config.chunk_duration_s,
        )
        result = run_synthetic(source, config=config, vad=EnergyVADDetector(config.vad))
        payload = serialize_result(result, "Démonstration synthétique")
        payload["waveform"] = downsample_envelope(result.chunks)
        payload["disclaimer"] = _disclaimer()
        return payload

    @app.post("/api/session/start")
    def start_session(
        device_id: str | None = Form(default=None),
        consent: bool = Form(False),
        energy_vad: bool = Form(False),
    ) -> dict[str, object]:
        if os.getenv("SAFEAI_DISABLE_MICROPHONE") == "1":
            raise HTTPException(
                status_code=503,
                detail="Microphone indisponible dans cette configuration (Docker).",
            )
        if _LIVE:
            raise HTTPException(status_code=409, detail="Une capture est déjà en cours.")
        if not consent:
            raise HTTPException(
                status_code=400,
                detail="Le consentement est obligatoire avant une capture microphone.",
            )
        config = load_config()
        vad = EnergyVADDetector(config.vad) if energy_vad else build_vad(config.vad)
        service = AcquisitionService(config=config, vad=vad)
        try:
            session = service.start(
                SourceConfig(
                    source_type="microphone",
                    device_id=device_id,
                    recording_consent=consent,
                    chunk_duration_s=config.chunk_duration_s,
                )
            )
        except ConsentRequiredError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AcquisitionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _LIVE[session.session_id] = service
        return {"session_id": session.session_id, "status": service.get_status().to_dict()}

    @app.get("/api/session/status")
    def session_status() -> dict[str, object]:
        service = _current_service()
        if service is None:
            return {"running": False}
        result = None
        if not service.get_status().running:
            collected = service.collect_result()
            result = serialize_result(collected)
            result["waveform"] = downsample_envelope(collected.chunks)
        return {"status": service.get_status().to_dict(), "result": result}

    @app.post("/api/session/stop")
    def stop_session() -> dict[str, object]:
        service = _current_service()
        if service is None:
            raise HTTPException(status_code=404, detail="Aucune session active.")
        service.stop()
        collected = service.collect_result()
        payload = serialize_result(collected)
        payload["waveform"] = downsample_envelope(collected.chunks)
        payload["status"] = service.get_status().to_dict()
        payload["disclaimer"] = _disclaimer()
        _LIVE.clear()
        return payload

    return app


def _current_service() -> AcquisitionService | None:
    if not _LIVE:
        return None
    return next(iter(_LIVE.values()))


def _inspect_path(path: Path, energy_vad: bool, source_label: str) -> dict[str, object]:
    config = load_config()
    vad = EnergyVADDetector(config.vad) if energy_vad else build_vad(config.vad)
    result = run_file(path, config=config, vad=vad)
    payload = serialize_result(result, source_label)
    payload["waveform"] = downsample_envelope(result.chunks)
    payload["disclaimer"] = _disclaimer()
    return payload


def _disclaimer() -> str:
    return (
        "SafeAI est un système pédagogique, non médical, non certifié. "
        "Il n'est pas autorisé à prendre automatiquement des décisions d'urgence."
    )
