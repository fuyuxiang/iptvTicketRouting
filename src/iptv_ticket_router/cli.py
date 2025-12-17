from __future__ import annotations

import argparse
import os

import uvicorn

from .api import create_app
from .config import config_path_from_env, load_config


def main() -> None:
    ap = argparse.ArgumentParser(prog="iptv-ticket-router")
    sub = ap.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Start HTTP service")
    serve.add_argument("--config", default=os.getenv("IPTV_CONFIG"), help="Path to config YAML")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true", help="Dev reload")

    args = ap.parse_args()

    if args.cmd == "serve":
        cfg_path = args.config or config_path_from_env()
        cfg = load_config(cfg_path)
        host = args.host or cfg.get("app", {}).get("host", "0.0.0.0")
        port = int(args.port or cfg.get("app", {}).get("port", 8080))

        app = create_app(cfg_path)
        uvicorn.run(app, host=host, port=port, reload=bool(args.reload))
