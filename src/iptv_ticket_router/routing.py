from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class RouteDecision:
    dept: str
    queue: str
    auto_routed: bool


class TicketRouter:
    """Map predicted label -> dept/queue.

    Design:
    - route_map: label -> dept
    - dept_to_queue: dept -> queue
    - if confidence < threshold: go to default_queue (manual triage)
    """

    def __init__(
        self,
        route_map: Dict[str, str],
        default_queue: str,
        dept_to_queue: Dict[str, str] | None = None,
        threshold_auto_route: float = 0.65,
    ) -> None:
        self.route_map = route_map or {}
        self.default_queue = default_queue
        self.dept_to_queue = dept_to_queue or {}
        self.threshold_auto_route = float(threshold_auto_route)

    def route(self, label: str, confidence: float) -> RouteDecision:
        if confidence < self.threshold_auto_route:
            return RouteDecision(dept="", queue=self.default_queue, auto_routed=False)

        dept = self.route_map.get(label, "")
        queue = self.dept_to_queue.get(dept, self.default_queue)
        return RouteDecision(dept=dept, queue=queue, auto_routed=True)
