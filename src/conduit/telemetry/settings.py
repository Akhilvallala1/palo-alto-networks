"""`config/telemetry.yaml` — where rows are written and what may be persisted.

Text persistence is the one setting with a safety consequence, so it is a config
value rather than a constructor default: the shipped config has
`persist_text: false` and the recorder drops prompt and completion text unless
it is explicitly flipped on.
"""

from pathlib import Path

from pydantic import BaseModel, Field

# `_load_section` is the same YAML + `CONDUIT_<SECTION>__<KEY>` override engine
# the four shipped config files already use. Forking it here would let telemetry
# config drift from the rest of the system's, so this reuses it deliberately.
from conduit.config import _load_section

__all__ = ["TelemetrySettings", "load_telemetry"]


class TelemetrySettings(BaseModel):
    """`config/telemetry.yaml` — telemetry store paths and retention policy."""

    enabled: bool = True
    dir: Path = Path("var/telemetry")
    jsonl_filename: str = "events.jsonl"
    sqlite_filename: str = "events.db"

    # Off in the shipped config. Turning it on persists raw prompt and
    # completion text next to the cost rows, which is a data-handling decision,
    # not a debugging convenience.
    persist_text: bool = False
    max_text_chars: int = Field(default=2000, gt=0)

    @property
    def jsonl_path(self) -> Path:
        return self.dir / self.jsonl_filename

    @property
    def sqlite_path(self) -> Path:
        return self.dir / self.sqlite_filename


def load_telemetry(
    config_dir: Path | None = None, env: dict[str, str] | None = None
) -> TelemetrySettings:
    """Load `telemetry.yaml`, applying `CONDUIT_TELEMETRY__*` overrides."""
    return _load_section(TelemetrySettings, "telemetry", config_dir, env)
