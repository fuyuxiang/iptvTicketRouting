from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load YAML config.

    - The repo/project root is inferred as parent directory of the config directory.
      (e.g. configs/config.yaml -> repo root = .)
    - Relative paths in config are resolved against repo root.
    """
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg["_config_path"] = str(p)
    cfg["_repo_root"] = str(p.parent.parent)  # configs/.. -> repo root
    return cfg


def abs_path(cfg: Dict[str, Any], maybe_rel_path: str) -> str:
    base = Path(cfg.get("_repo_root", ".")).resolve()
    return str((base / maybe_rel_path).resolve())


def config_path_from_env(env_key: str = "IPTV_CONFIG", default: str = "configs/config.yaml") -> str:
    return os.getenv(env_key, default)
