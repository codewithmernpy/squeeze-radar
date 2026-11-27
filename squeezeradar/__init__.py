"""SqueezeRadar — Real-time crypto perpetual futures squeeze scanner & AI analyst."""

from squeezeradar.config import (
    ACTIVE_SCORE_THRESHOLD,
    DEFAULT_SYMBOLS,
    ELEVATED_SCORE_THRESHOLD,
    LiqMapConfig,
    MLConfig,
    ProviderConfig,
    ScoreConfig,
    StateConfig,
)
from squeezeradar.scores import state_for_score

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ACTIVE_SCORE_THRESHOLD",
    "DEFAULT_SYMBOLS",
    "ELEVATED_SCORE_THRESHOLD",
    "LiqMapConfig",
    "ScoreConfig",
    "StateConfig",
    "MLConfig",
    "ProviderConfig",
    "state_for_score",
]
