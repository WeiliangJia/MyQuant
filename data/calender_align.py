from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PRICE_COLS = ["Open", "High", "Low", "Close"]
MAX_FFILL_GAP = 3
RETURN_OUTLIER_ABS = 0.30
MIN_STALE_STREAK = 3
MIN_ADV20_DOLLAR = 1_000_000.0

QC_THRESHOLDS = {
    "max_bad_price_rate": 0.002,           # <= 0.2%
    "max_negative_volume_rate": 0.0001,    # <= 0.01%
    "max_in_life_gap_unfilled_rows": 1000,
    "min_tradable_ratio": 0.90,            # vs cleaned observed rows
}


def _resolve_processed_dir() -> Path:
    """Support both data/processed and data/data/processed layouts."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "data" / "processed",        # current layout in this repo
        script_dir.parent / "data" / "processed",  # canonical project layout
    ]

    for c in candidates:
        if (c / "panel.parquet").exists():
            return c
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _clean_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    stats: dict[str, int] = {}
    original_rows = len(panel)

    required = {"Date", "ticker", "Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    panel = panel.copy()
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce")
    panel["ticker"] = panel["ticker"].astype("string").str.strip()

    for col in PRICE_COLS + ["Volume"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    bad_key = panel["Date"].isna() | panel["ticker"].isna() | (panel["ticker"] == "")
    stats["drop_bad_key_rows"] = int(bad_key.sum())
    panel = panel.loc[~bad_key].copy()

    dup = panel.duplicated(subset=["Date", "ticker"], keep="last")
    stats["drop_duplicate_key_rows"] = int(dup.sum())
    panel = panel.loc[~dup].copy()

    non_positive = (panel[PRICE_COLS] <= 0).any(axis=1)
    relation_bad = (
        (panel["High"] < panel[["Open", "Close", "Low"]].max(axis=1))
        | (panel["Low"] > panel[["Open", "Close", "High"]].min(axis=1))
    )
    bad_price = non_positive | relation_bad
    stats["bad_price_rows"] = int(bad_price.sum())
    panel.loc[bad_price, PRICE_COLS] = pd.NA

    neg_vol = panel["Volume"] < 0
    stats["negative_volume_rows"] = int(neg_vol.sum())
    panel.loc[neg_vol, "Volume"] = pd.NA

    # Return outlier flag on observed close series.
    panel = panel.sort_values(["ticker", "Date"]).reset_index(drop=True)
    panel["ret_1d"] = panel.groupby("ticker")["Close"].pct_change(fill_method=None)
    panel["flag_return_outlier"] = panel["ret_1d"].abs() > RETURN_OUTLIER_ABS
    panel["flag_return_outlier"] = panel["flag_return_outlier"].fillna(False)

    # Stale flag: close unchanged + zero volume for consecutive days.
    stale_cond = (
        panel.groupby("ticker")["Close"].diff().fillna(1).eq(0)
        & panel["Volume"].fillna(0).eq(0)
    )
    streak = stale_cond.groupby(panel["ticker"]).transform(
        lambda s: s.groupby((~s).cumsum()).cumcount() + 1
    )
    panel["flag_stale"] = stale_cond & (streak >= MIN_STALE_STREAK)

    panel["flag_bad_price"] = bad_price
    panel["flag_negative_volume"] = neg_vol

    stats["input_rows"] = int(original_rows)
    stats["clean_rows"] = int(len(panel))
    stats["return_outlier_rows"] = int(panel["flag_return_outlier"].sum())
    stats["stale_rows"] = int(panel["flag_stale"].sum())
    return panel, stats


def _align_one(df: pd.DataFrame, calendar: pd.Index) -> pd.DataFrame:
    ticker = str(df.name)
    df = df.sort_values("Date").set_index("Date")
    observed_index = df.index.copy()
    first_date = observed_index.min()
    last_date = observed_index.max()

    aligned = df.reindex(calendar)
    aligned["ticker"] = ticker
    aligned["is_observed"] = aligned.index.isin(observed_index)

    in_life = (aligned.index >= first_date) & (aligned.index <= last_date)
    observed_bad_price = aligned["is_observed"] & aligned[PRICE_COLS].isna().any(axis=1)

    # Missing type classification before/after short-gap fill.
    aligned["missing_type"] = "observed_ok"
    aligned.loc[observed_bad_price, "missing_type"] = "observed_bad_price"
    aligned.loc[~in_life & (aligned.index < first_date), "missing_type"] = "pre_listing"
    aligned.loc[~in_life & (aligned.index > last_date), "missing_type"] = "post_delist"

    before_fill_missing = aligned[PRICE_COLS].isna().any(axis=1)
    aligned[PRICE_COLS] = aligned[PRICE_COLS].ffill(limit=MAX_FFILL_GAP)
    aligned.loc[~in_life, PRICE_COLS] = pd.NA
    after_fill_missing = aligned[PRICE_COLS].isna().any(axis=1)

    in_life_unobserved = in_life & (~aligned["is_observed"])
    filled_gap = in_life_unobserved & before_fill_missing & (~after_fill_missing)
    unfilled_gap = in_life_unobserved & after_fill_missing

    aligned.loc[filled_gap, "missing_type"] = "filled_gap"
    aligned.loc[unfilled_gap, "missing_type"] = "in_life_gap_unfilled"

    aligned["is_synthetic"] = filled_gap

    aligned.loc[aligned["is_synthetic"], "Volume"] = 0
    aligned.loc[~in_life, "Volume"] = pd.NA

    for col in ["flag_bad_price", "flag_negative_volume", "flag_return_outlier", "flag_stale"]:
        aligned[col] = aligned[col].fillna(False).astype(bool)

    aligned["DollarVolume"] = aligned["Close"] * aligned["Volume"].fillna(0)
    aligned["ADV20"] = (
        aligned["DollarVolume"]
        .rolling(window=20, min_periods=20)
        .median()
        .astype(float)
    )
    aligned["is_liquid"] = aligned["ADV20"].fillna(0) >= MIN_ADV20_DOLLAR

    valid_price = aligned[PRICE_COLS].notna().all(axis=1)
    valid_volume = aligned["Volume"].fillna(0) > 0
    clean_signal = (
        ~aligned["flag_bad_price"]
        & ~aligned["flag_negative_volume"]
        & ~aligned["flag_return_outlier"]
        & ~aligned["flag_stale"]
    )
    aligned["is_tradable"] = (
        aligned["is_observed"]
        & valid_price
        & valid_volume
        & clean_signal
        & aligned["is_liquid"]
    )

    return aligned.reset_index()


def main() -> None:
    processed_dir = _resolve_processed_dir()
    input_path = processed_dir / "panel.parquet"
    clean_output_path = processed_dir / "panel_cleaned.parquet"
    output_path = processed_dir / "panel_aligned.parquet"
    tradable_path = processed_dir / "panel_tradable.parquet"
    anomalies_path = processed_dir / "panel_anomalies.csv"
    report_path = processed_dir / "panel_aligned_qc.json"
    gate_path = processed_dir / "panel_qc_gate.json"

    panel = pd.read_parquet(input_path)
    panel, clean_stats = _clean_panel(panel)
    panel.to_parquet(clean_output_path, index=False)

    calendar = pd.Index(sorted(panel["Date"].unique()), name="Date")
    aligned = (
        panel.groupby("ticker", group_keys=False)
        .apply(_align_one, calendar=calendar)
        .sort_values(["Date", "ticker"])
        .reset_index(drop=True)
    )

    tradable = aligned.loc[aligned["is_tradable"]].copy()

    anomalies = aligned.loc[
        aligned["flag_bad_price"]
        | aligned["flag_negative_volume"]
        | aligned["flag_return_outlier"]
        | aligned["flag_stale"]
        | (aligned["missing_type"] == "in_life_gap_unfilled")
    , [
        "Date",
        "ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "missing_type",
        "flag_bad_price",
        "flag_negative_volume",
        "flag_return_outlier",
        "flag_stale",
        "is_observed",
        "is_tradable",
    ]].sort_values(["Date", "ticker"])

    missing_type_counts = {
        k: int(v) for k, v in aligned["missing_type"].value_counts(dropna=False).to_dict().items()
    }
    cleaned_observed_rows = int(aligned["is_observed"].sum())
    tradable_rows = int(aligned["is_tradable"].sum())
    bad_price_rate = clean_stats["bad_price_rows"] / max(clean_stats["clean_rows"], 1)
    negative_volume_rate = clean_stats["negative_volume_rows"] / max(clean_stats["clean_rows"], 1)
    tradable_ratio = tradable_rows / max(cleaned_observed_rows, 1)
    in_life_gap_unfilled_rows = int((aligned["missing_type"] == "in_life_gap_unfilled").sum())

    report = {
        **clean_stats,
        "calendar_days": int(len(calendar)),
        "aligned_rows": int(len(aligned)),
        "added_rows_by_align": int(len(aligned) - len(panel)),
        "missing_price_rows_after_align": int(aligned[PRICE_COLS].isna().any(axis=1).sum()),
        "synthetic_rows_after_align": int(aligned["is_synthetic"].sum()),
        "tradable_rows_after_align": tradable_rows,
        "tradable_ratio_vs_observed": tradable_ratio,
        "liquid_rows_after_align": int(aligned["is_liquid"].sum()),
        "missing_type_counts": missing_type_counts,
        "in_life_gap_unfilled_rows": in_life_gap_unfilled_rows,
        "anomalies_exported_rows": int(len(anomalies)),
        "paths": {
            "input_path": str(input_path),
            "clean_output_path": str(clean_output_path),
            "aligned_output_path": str(output_path),
            "tradable_output_path": str(tradable_path),
            "anomalies_output_path": str(anomalies_path),
        },
        "output_path": str(output_path),
    }

    gate_checks = {
        "bad_price_rate_ok": bad_price_rate <= QC_THRESHOLDS["max_bad_price_rate"],
        "negative_volume_rate_ok": negative_volume_rate <= QC_THRESHOLDS["max_negative_volume_rate"],
        "in_life_gap_unfilled_ok": (
            in_life_gap_unfilled_rows <= QC_THRESHOLDS["max_in_life_gap_unfilled_rows"]
        ),
        "tradable_ratio_ok": tradable_ratio >= QC_THRESHOLDS["min_tradable_ratio"],
    }
    gate = {
        "pass": bool(all(gate_checks.values())),
        "checks": gate_checks,
        "thresholds": QC_THRESHOLDS,
        "metrics": {
            "bad_price_rate": bad_price_rate,
            "negative_volume_rate": negative_volume_rate,
            "in_life_gap_unfilled_rows": in_life_gap_unfilled_rows,
            "tradable_ratio_vs_observed": tradable_ratio,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(output_path, index=False)
    tradable.to_parquet(tradable_path, index=False)
    anomalies.to_csv(anomalies_path, index=False)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Full strict cleaning + alignment done")
    print(f"Saved clean panel: {clean_output_path}")
    print(f"Saved aligned panel: {output_path}")
    print(f"Saved tradable panel: {tradable_path}")
    print(f"Saved anomalies: {anomalies_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved QC gate: {gate_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
