"""P3 interpretation: full-train LightGBM + TreeSHAP (``pred_contrib``).

The model is trained on all 10,886 training rows with the same feature builder
used by the pipeline.  SHAP values are computed on a 5,000-row sample of the
training set, so the results are **in-sample / full-train** and optimistic.
No ``shap`` package is required: LightGBM's native ``pred_contrib=True`` is
used, which is the same TreeSHAP algorithm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config, path_of
from src.eval import _utils as U
from src.eval._modeling import fit_lgbm, predict_model, prepare_features
from src.features.build import FeatureBuilder


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TreeSHAP interpretation for the full-train LightGBM (P3)")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--out-dir", default=None, help="output dir (default: reports/eval)")
    p.add_argument("--fig-dir", default=None, help="figure dir (default: reports/figures)")
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--target",
        choices=("auto", "count", "log"),
        default="auto",
        help="auto: match P2 final model when reports/p2_final_metrics.json exists",
    )
    return p.parse_args(argv)


def _resolve_target_mode(cfg: dict, requested: str) -> tuple[str, str]:
    """Return (mode, source).  ``mode`` is 'count' or 'log'."""
    if requested in ("count", "log"):
        return requested, "cli"
    p = path_of(cfg, "reports") / "p2_final_metrics.json"
    if p.exists():
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            model = str(payload.get("model", ""))
            if model == "lgbm_log":
                return "log", f"p2_final_metrics.json (model={model})"
            if model in ("lgbm", "lgbm_poisson"):
                return "count", f"p2_final_metrics.json (model={model})"
        except (json.JSONDecodeError, OSError):
            pass
    objective = str(cfg.get("model", {}).get("lgbm", {}).get("objective", "poisson"))
    return ("count" if objective == "poisson" else "log"), "config objective"


def _importance_frame(shap: np.ndarray, feature_names: list[str], X: pd.DataFrame, model) -> pd.DataFrame:
    mean_abs = np.mean(np.abs(shap), axis=0)
    mean_shap = np.mean(shap, axis=0)
    std_shap = np.std(shap, axis=0)
    gain = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype="float64")
    split = np.asarray(model.booster_.feature_importance(importance_type="split"), dtype="float64")
    total = float(mean_abs.sum()) or 1.0
    total_gain = float(gain.sum()) or 1.0
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_type": [
                "categorical" if isinstance(X[c].dtype, pd.CategoricalDtype) else "numeric"
                for c in feature_names
            ],
            "mean_abs_shap": mean_abs,
            "mean_shap": mean_shap,
            "std_shap": std_shap,
            "mean_abs_share": mean_abs / total,
            "gain": gain,
            "gain_share": gain / total_gain,
            "split": split,
        }
    )
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _plot_importance(imp: pd.DataFrame, out: Path, n_show: int, meta: dict, plt):
    d = imp.head(n_show).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, max(5.2, 0.34 * len(d) + 1.8)))
    colors = [U.COLORS["accent"] if t == "categorical" else U.COLORS["resid"] for t in d["feature_type"]]
    ax.barh(np.arange(len(d)), d["mean_abs_shap"], color=colors, alpha=0.88)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["feature"], fontsize=9)
    ax.set_xlabel(f"mean |TreeSHAP| ({meta.get('shap_space_label', 'raw score')})")
    ax.set_title("TreeSHAP feature importance — full-train LightGBM (in-sample)")
    vmax = float(d["mean_abs_shap"].max()) if len(d) else 1.0
    for i, v in enumerate(d["mean_abs_shap"]):
        ax.text(v + vmax * 0.012, i, f"{v:.3f} ({v / imp['mean_abs_shap'].sum():.1%})", va="center", fontsize=8)
    ax.set_xlim(0, vmax * 1.24)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=U.COLORS["resid"], alpha=0.88, label="numeric"),
            Patch(facecolor=U.COLORS["accent"], alpha=0.88, label="categorical"),
        ],
        loc="lower right",
    )
    ax.text(
        0.02,
        0.02,
        f"sample n={meta['sample_size']:,} / train n={meta['n_train']:,}\n"
        f"mean |SHAP| sum={imp['mean_abs_shap'].sum():.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    return U.save_fig(fig, out)


def _pick_color_feature(feature: str, X: pd.DataFrame) -> str | None:
    for cand in ("hour", "temp", "humidity", "windspeed", "dayofyear", "year"):
        if cand in X.columns and cand != feature and not isinstance(X[cand].dtype, pd.CategoricalDtype):
            return cand
    return None


def _plot_dependence(
    shap: np.ndarray,
    imp: pd.DataFrame,
    Xs: pd.DataFrame,
    out: Path,
    meta: dict,
    plt,
):
    top = imp.head(4)["feature"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    axes = axes.reshape(-1)
    rng = np.random.default_rng(meta["seed"])
    for ax, feat in zip(axes, top):
        j = list(Xs.columns).index(feat)
        sv = shap[:, j]
        vals = Xs[feat]
        mean_abs = float(np.mean(np.abs(sv)))
        if isinstance(vals.dtype, pd.CategoricalDtype):
            cats = list(vals.cat.categories)
            if feat == "hour_x_workingday":
                cats = sorted(
                    cats,
                    key=lambda s: tuple(int(x) for x in str(s).split("_") if str(x).isdigit()),
                )
            data = [sv[(vals == c).to_numpy()] for c in cats]
            bp = ax.boxplot(
                data,
                positions=np.arange(len(cats)),
                widths=0.6,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#1b1b1b", "linewidth": 1.0},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(U.COLORS["accent"])
                patch.set_alpha(0.55)
            ax.set_xticks(np.arange(len(cats)))
            ax.set_xticklabels([str(c) for c in cats], rotation=90, fontsize=6)
            ax.set_xlabel(feat)
        else:
            color_feat = _pick_color_feature(feat, Xs)
            if color_feat is not None:
                cv = pd.to_numeric(Xs[color_feat], errors="coerce").to_numpy(dtype="float64")
                sc = ax.scatter(
                    vals.to_numpy(dtype="float64"),
                    sv,
                    c=cv,
                    cmap="viridis",
                    s=9,
                    alpha=0.55,
                    edgecolors="none",
                    rasterized=True,
                )
                cb = fig.colorbar(sc, ax=ax, pad=0.01)
                cb.set_label(color_feat, fontsize=8)
                cb.ax.tick_params(labelsize=7)
            else:
                ax.scatter(vals.to_numpy(dtype="float64"), sv, s=9, alpha=0.45, color=U.COLORS["resid"], edgecolors="none", rasterized=True)
            # binned median trend
            try:
                bins = pd.qcut(vals, q=20, duplicates="drop")
                grp = pd.DataFrame({"bin": bins, "shap": sv, "x": vals}).groupby("bin", observed=True)
                mids = np.array([iv.mid for iv in grp.groups.keys()], dtype="float64")
                med = grp["shap"].median().to_numpy()
                ax.plot(mids, med, color=U.COLORS["pred"], lw=1.8, label="binned median")
                ax.legend(loc="best", fontsize=8)
            except (ValueError, TypeError):
                pass
            ax.set_xlabel(feat)
        ax.axhline(0, color="#333333", lw=0.8, ls="--")
        ax.set_ylabel(f"SHAP value ({meta.get('shap_space_label', 'raw score')})")
        ax.set_title(f"{feat}  (mean |SHAP|={mean_abs:.3f})", fontsize=10)
    fig.suptitle("TreeSHAP dependence — top 4 features (full-train model, in-sample)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return U.save_fig(fig, out)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    out_dir = U.ensure_dir(U.resolve_path(args.out_dir, path_of(cfg, "reports") / "eval"))
    fig_dir = U.ensure_dir(U.resolve_path(args.fig_dir, path_of(cfg, "figures")))

    train = U.load_train(cfg)
    builder = FeatureBuilder(use_target_aggregates=True).fit(train)
    X = prepare_features(builder.transform(train))
    y = train["count"].to_numpy(dtype="float64")

    target_mode, target_source = _resolve_target_mode(cfg, args.target)
    log_target = target_mode == "log"
    if log_target:
        target_desc = "log1p(count) + L2"
        shap_space = "log1p(count)"
        shap_space_label = "log1p space"
    else:
        target_desc = "raw count + Poisson objective"
        shap_space = "Poisson raw score (log mean)"
        shap_space_label = "Poisson raw score"

    print(
        f"[P3-SHAP] training full-train LightGBM on {len(train):,} rows / {X.shape[1]} features "
        f"| target={target_desc} (from {target_source}) ..."
    )
    model = fit_lgbm(X, y, cfg, quick=False, seed=args.seed, log_target=log_target)
    pred_all = predict_model(model, X, log_target=log_target)
    in_sample = U.metric_block(y, pred_all)

    n = len(X)
    sample_size = int(min(max(args.sample_size, 1), n))
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n, size=sample_size, replace=False))
    Xs = X.iloc[idx].copy()

    contrib = model.booster_.predict(Xs, pred_contrib=True)
    contrib = np.asarray(contrib, dtype="float64")
    if contrib.ndim != 2 or contrib.shape[1] != Xs.shape[1] + 1:
        raise RuntimeError(f"unexpected pred_contrib shape: {contrib.shape}; expected ({len(Xs)}, {Xs.shape[1] + 1})")
    shap = contrib[:, :-1]
    expected = float(np.mean(contrib[:, -1]))

    raw_pred = np.asarray(model.booster_.predict(Xs, raw_score=True), dtype="float64")
    reconstruction = np.max(np.abs(shap.sum(axis=1) + expected - raw_pred))

    feature_names = list(Xs.columns)
    booster_names = list(model.booster_.feature_name())
    if booster_names != feature_names:
        # LightGBM preserves DataFrame names in normal cases; fall back to
        # booster order only if they differ, and warn in metadata.
        feature_names = booster_names
        print("[P3-SHAP] WARNING: booster feature names differ from DataFrame columns; using booster order.")

    imp = _importance_frame(shap, feature_names, Xs, model)
    imp_path = out_dir / "shap_importance.csv"
    imp.to_csv(imp_path, index=False, float_format="%.8f")

    meta = {
        "method": "LightGBM native pred_contrib=True (TreeSHAP)",
        "model": "LGBMRegressor",
        "target_mode": target_mode,
        "target_source": target_source,
        "target_transform": target_desc,
        "objective": str(model.get_params().get("objective")),
        "shap_space": shap_space,
        "shap_space_label": shap_space_label,
        "n_train": int(n),
        "n_features": int(Xs.shape[1]),
        "sample_size": int(sample_size),
        "seed": int(args.seed),
        "expected_value_raw": expected,
        "reconstruction_max_abs_error": float(reconstruction),
        "in_sample_metrics": in_sample,
        "params": model.get_params(),
        "top_features": imp.head(15)[["rank", "feature", "mean_abs_shap", "mean_abs_share", "gain_share"]].to_dict("records"),
        "caveat": (
            "SHAP is computed on a sample of the same training rows used to fit the full-train model. "
            "It is in-sample, includes target-aggregate features fitted on all train rows, and is therefore "
            "optimistic; it describes model behaviour, not causal effects or out-of-time importance."
        ),
        "generated_at": U.utc_now(),
    }
    meta_path = out_dir / "shap_meta.json"
    U.save_json(meta_path, meta)

    plt = U.setup_matplotlib()
    imp_fig = _plot_importance(imp, fig_dir / "p3_shap_importance.png", n_show=20, meta=meta, plt=plt)
    dep_fig = _plot_dependence(shap, imp, Xs, fig_dir / "p3_shap_dependence.png", meta, plt)

    print(f"[P3-SHAP] in-sample RMSLE={in_sample['rmsle']:.5f}  R2={in_sample['r2']:.5f}")
    print(f"[P3-SHAP] expected_value(raw)={expected:.5f}  reconstruction_max_abs_error={reconstruction:.2e}")
    print("[P3-SHAP] top features:")
    for _, r in imp.head(10).iterrows():
        print(f"           {int(r['rank']):2d}. {r['feature']:<26s} mean|SHAP|={r['mean_abs_shap']:.4f} ({r['mean_abs_share']:.1%})")
    print(f"[P3-SHAP] wrote {imp_path}")
    print(f"[P3-SHAP] wrote {meta_path}")
    print(f"[P3-SHAP] wrote {imp_fig}")
    print(f"[P3-SHAP] wrote {dep_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
