from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from iptv_ticket_router.config import abs_path, load_config
from iptv_ticket_router.feature import build_text_from_row
from iptv_ticket_router.model import TicketClassifier


def build_route_map(df: pd.DataFrame, label_field: str, dept_field: str) -> dict[str, str]:
    """Majority vote: label -> dept."""
    route_map: dict[str, str] = {}
    grouped = df.groupby([label_field, dept_field]).size().reset_index(name="cnt")
    for label in df[label_field].dropna().astype(str).unique():
        sub = grouped[grouped[label_field].astype(str) == str(label)].sort_values("cnt", ascending=False)
        if len(sub) > 0:
            route_map[str(label)] = str(sub.iloc[0][dept_field])
    return route_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    df = pd.read_csv(abs_path(cfg, data_cfg["train_csv"]))
    text_fields = data_cfg["text_fields"]
    label_field = data_cfg["label_field"]
    dept_field = data_cfg["dept_field"]

    texts = [build_text_from_row(r, text_fields) for _, r in df.iterrows()]
    y = df[label_field].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    clf = TicketClassifier().fit(X_train, y_train)

    pred = clf.predict(X_test)
    # lightweight eval
    from sklearn.metrics import classification_report, accuracy_score

    acc = accuracy_score(y_test, pred)
    rep = classification_report(y_test, pred, digits=4)
    print(f"=== Offline Evaluation (label={label_field}) ===")
    print(f"accuracy={acc:.4f}\n{rep}")

    model_dir = abs_path(cfg, cfg["model"]["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, cfg["model"]["model_file"])
    label_map_path = os.path.join(model_dir, cfg["model"]["label_map_file"])
    route_map_path = os.path.join(model_dir, cfg["model"]["route_map_file"])
    meta_path = os.path.join(model_dir, "model_meta.json")

    clf.save(model_path, label_map_path, meta_path=meta_path)

    route_map = build_route_map(df, label_field, dept_field)
    with open(route_map_path, "w", encoding="utf-8") as f:
        json.dump(route_map, f, ensure_ascii=False, indent=2)

    print("Saved model to:", model_path)
    print("Saved label map to:", label_map_path)
    print("Saved route map to:", route_map_path)
    print("Saved meta to:", meta_path)


if __name__ == "__main__":
    main()
