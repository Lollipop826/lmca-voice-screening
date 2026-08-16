from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class VoiceRouteResult:
    handled: bool
    stop_connection: bool = False

    def __bool__(self) -> bool:
        return self.handled
