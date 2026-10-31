import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_time_folds_no_overlap():
    payload = json.loads((ROOT / "splits/time_folds.json").read_text(encoding="utf-8"))
    assert payload["n_splits"] == 5
    for f in payload["folds"]:
        tr = np.asarray(f["train_idx"])
        va = np.asarray(f["valid_idx"])
        assert len(np.intersect1d(tr, va)) == 0
        assert len(tr) > 0 and len(va) > 0


def test_processed_shapes():
    import pandas as pd
    tr = pd.read_parquet(ROOT / "data/processed/train.parquet")
    te = pd.read_parquet(ROOT / "data/processed/test.parquet")
    assert len(tr) == 10886
    assert len(te) == 6493
    assert "count" in tr.columns
    assert "count" not in te.columns
