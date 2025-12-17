from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "iptv_ticket_router_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)

LATENCY = Histogram(
    "iptv_ticket_router_request_latency_seconds",
    "Request latency (seconds)",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)
