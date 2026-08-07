from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iedi.codebook import Codebook
from iedi.pipeline import build_pipeline
from iedi.providers import GoogleGenAIProvider
from iedi.transport import CloudInterpretationService, create_fastapi_app


def create_app():
    """Uvicorn factory; secrets remain in the cloud process environment."""

    codebook = Codebook.from_json(ROOT / "data" / "codebook.demo.json")
    pipeline = build_pipeline(
        "paper4",
        codebook=codebook,
        provider=GoogleGenAIProvider(),
        config_path=ROOT / "configs" / "paper4.json",
    )
    return create_fastapi_app(CloudInterpretationService(pipeline))
