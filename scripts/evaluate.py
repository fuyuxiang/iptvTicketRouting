from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from iptv_ticket_router.config import abs_path, load_config
from iptv_ticket_router.feature import build_text_from_row
from iptv_ticket_router.model import TicketClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model_dir", default=None, help="override model dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    df = pd.read_csv(abs_path(cfg, data_cfg["train_csv"]))
    text_fields = data_cfg["text_fields"]
    label_field = data_cfg["label_field"]

    texts = [build_text_from_row(r, text_fields) for _, r in df.iterrows()]
    y_true = df[label_field].astype(str).tolist()

    model_dir = abs_path(cfg, cfg["model"]["model_dir"])
    if args.model_dir:
        model_dir = args.model_dir

    model_path = f"{model_dir}/{cfg['model']['model_file']}"
    clf = TicketClassifier().load(model_path)

    y_pred = clf.predict(texts)
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=4)
    print(f"accuracy={acc:.4f}\n{rep}")


if __name__ == "__main__":
    main()
