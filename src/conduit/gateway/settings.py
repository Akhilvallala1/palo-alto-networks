"""`config/gateway.yaml` — the static API-key table and its defaults.

Auth is a lookup table, exactly as in `mcp-gateway`: a key is a row, and the row
carries the three things the rest of the request needs — which team to attribute
the traffic to, what quota the key holds, and what tier ceiling its policy
allows. Nothing here interprets those values; the pipeline does.
"""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

# The same YAML + `CONDUIT_<SECTION>__<KEY>` override engine every other config
# file uses. Telemetry reuses it for the same reason: a second loader would let
# gateway config drift from the rest of the system's.
from conduit.config import _load_section
from conduit.contracts import Complexity

__all__ = ["ApiKeySettings", "GatewaySettings", "load_gateway"]


class ApiKeySettings(BaseModel):
    """One key: who it bills to, how fast it may go, how far up it may route."""

    key: str = Field(min_length=1)
    team: str = Field(min_length=1)

    #: None means "use the gateway-wide default".
    rate_limit_per_minute: int | None = Field(default=None, gt=0)

    #: None means "use `routing.yaml`'s `budgets.default_daily_usd`". Set, it
    #: becomes a per-key limit in the router's spend ledger.
    daily_budget_usd: float | None = Field(default=None, ge=0.0)

    #: Routing ceiling. A key capped at `trivial` never reaches an opus model
    #: however complex its prompt classifies.
    max_tier: Complexity = Complexity.COMPLEX


class GatewaySettings(BaseModel):
    """`config/gateway.yaml` — keys, default quota, and the auth switch."""

    api_keys: list[ApiKeySettings] = Field(default_factory=list)
    rate_limit_per_minute: int = Field(default=60, gt=0)

    #: False disables the key check entirely. Only sane for a local single-user
    #: run, so it defaults to on and the app logs loudly when it is off.
    require_auth: bool = True

    @field_validator("api_keys")
    @classmethod
    def _keys_are_unique(cls, value: list[ApiKeySettings]) -> list[ApiKeySettings]:
        seen: set[str] = set()
        for entry in value:
            if entry.key in seen:
                raise ValueError(f"duplicate api key for team {entry.team!r}")
            seen.add(entry.key)
        return value


def load_gateway(
    config_dir: Path | None = None, env: dict[str, str] | None = None
) -> GatewaySettings:
    """Load `gateway.yaml`, applying `CONDUIT_GATEWAY__*` overrides."""
    return _load_section(GatewaySettings, "gateway", config_dir, env)
