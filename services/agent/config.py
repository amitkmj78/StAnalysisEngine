"""AGT-33: every agent limit in one frozen, versioned config.

Changing a value changes config_version (a hash of the canonical JSON),
which every journal row records, so any past run can be tied to the
exact limits it used. Changes go through git, not the database.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class AgentConfig:
    max_position_pct: float = 5.0
    max_sector_pct: float = 25.0
    max_positions: int = 20
    regime_exposure_pct: Mapping[str, float] = field(
        default_factory=lambda: {"Risk-On": 100.0, "Constructive": 90.0, "Neutral": 80.0, "Cautious": 50.0, "Risk-Off": 20.0}
    )
    daily_loss_limit_pct: float = 2.0
    drawdown_breaker_pct: float = 10.0
    drawdown_breaker_exposure_pct: float = 20.0
    stop_atr_multiple: float = 2.5
    stop_min_pct: float = 3.0
    stop_max_pct: float = 15.0
    min_avg_dollar_volume: float = 50_000_000.0
    dollar_volume_window_days: int = 20
    volatility_window_days: int = 63
    trend_sma_days: int = 200
    earnings_blackout_calendar_days: int = 4
    rebalance_band_pct: float = 1.0
    vol_target_annual_pct: float = 15.0
    max_candidates_scanned: int = 40


CONFIG = AgentConfig()


def config_version(config: AgentConfig = CONFIG) -> str:
    canonical = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
