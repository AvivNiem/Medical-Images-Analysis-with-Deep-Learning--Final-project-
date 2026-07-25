"""
stats.py
--------
Statistical comparison between model variants, mirroring the original Attention
U-Net paper, which reports paired significance tests / p-values (e.g. "increase
recall values (p = .005)").

Because our test set is fixed (same patient split for every variant), we can run a
PAIRED test on the per-slice Dice scores: each test slice is scored by every
variant, so the scores are naturally paired sample-by-sample. We use the
Wilcoxon signed-rank test (non-parametric, appropriate since per-slice Dice is
not normally distributed and is bounded in [0, 1]).

We also aggregate the multi-seed runs into a per-variant mean +/- std of the
run-level test Dice, for the paper-style results table.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np


def aggregate_by_variant(results: Dict) -> Dict[str, Dict[str, float]]:
    """Collapse multi-seed runs into per-variant summary statistics (mean/std of
    the run-level test metrics), for the main results table."""
    by_variant = defaultdict(lambda: defaultdict(list))
    for run in results["runs"]:
        att = run["attention_type"]
        m = run["test_metrics"]
        by_variant[att]["dice"].append(m["dice_mean"])
        by_variant[att]["precision"].append(m["precision_mean"])
        by_variant[att]["recall"].append(m["recall_mean"])
        by_variant[att]["empty_fp"].append(m["empty_gt_fp_rate"])
        by_variant[att]["params"].append(run["num_params"])
        by_variant[att]["inf_time"].append(run["inference_time_s"])

    summary = {}
    for att, d in by_variant.items():
        summary[att] = {
            "dice_mean": float(np.mean(d["dice"])),
            "dice_std": float(np.std(d["dice"])),
            "precision_mean": float(np.mean(d["precision"])),
            "recall_mean": float(np.mean(d["recall"])),
            "empty_fp_mean": float(np.mean(d["empty_fp"])),
            "params": int(np.mean(d["params"])),
            "inf_time_mean": float(np.mean(d["inf_time"])),
            "n_seeds": len(d["dice"]),
        }
    return summary


def aggregate_sweep(results: Dict) -> Dict:
    """Aggregate a low-data sweep into per-(variant, train_size) mean/std Dice.

    Returns a nested dict: summary[variant][train_size] = {'dice_mean', 'dice_std',
    'n_seeds'}, plus a sorted list of the training sizes encountered. `train_size`
    None (full training set) is reported under the key 'all'."""
    by = defaultdict(lambda: defaultdict(list))
    sizes = set()
    for run in results["runs"]:
        att = run["attention_type"]
        ts = run.get("train_size", None)
        key = "all" if ts is None else ts
        sizes.add(key)
        by[att][key].append(run["test_metrics"]["dice_mean"])

    summary = {}
    for att, per_size in by.items():
        summary[att] = {}
        for key, vals in per_size.items():
            summary[att][key] = {
                "dice_mean": float(np.mean(vals)),
                "dice_std": float(np.std(vals)),
                "n_seeds": len(vals),
            }

    # Order sizes numerically with 'all' last.
    numeric = sorted(s for s in sizes if s != "all")
    ordered = numeric + (["all"] if "all" in sizes else [])
    return {"summary": summary, "train_sizes": ordered}


def paired_wilcoxon(per_slice_dice: Dict[str, List[float]], variant_a: str,
                    variant_b: str) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test on per-slice Dice between two variants.

    `per_slice_dice` maps variant name -> list of per-slice Dice values, where the
    lists are aligned slice-by-slice (same order, same slices). Returns the test
    statistic, p-value, and the median paired difference (a - b)."""
    from scipy.stats import wilcoxon

    a = np.asarray(per_slice_dice[variant_a])
    b = np.asarray(per_slice_dice[variant_b])
    assert a.shape == b.shape, "per-slice arrays must be aligned and equal length"

    # Drop exact ties (zero differences) which Wilcoxon can't rank.
    diff = a - b
    nonzero = diff != 0
    if nonzero.sum() == 0:
        return {"statistic": float("nan"), "p_value": 1.0, "median_diff": 0.0}

    stat, p = wilcoxon(a[nonzero], b[nonzero])
    return {"statistic": float(stat), "p_value": float(p),
            "median_diff": float(np.median(diff))}


def format_results_table(summary: Dict[str, Dict[str, float]]) -> str:
    """Pretty-print the paper-style results table (mean +/- std across seeds)."""
    name_map = {
        "None": "U-Net (baseline)",
        "spatial": "Attention U-Net",
        "cbam": "AG + CBAM (bolt-on)",
        "hybrid": "AG + context-gated (ours)",
    }
    order = ["None", "spatial", "cbam", "hybrid"]

    header = (f"{'Method':28s} {'DSC':>16s} {'Prec':>8s} {'Rec':>8s} "
              f"{'emptyFP':>9s} {'Params':>10s} {'Inf(s)':>8s}")
    lines = [header, "-" * len(header)]
    for att in order:
        if att not in summary:
            continue
        s = summary[att]
        lines.append(
            f"{name_map.get(att, att):28s} "
            f"{s['dice_mean']:.3f}+/-{s['dice_std']:.3f}   "
            f"{s['precision_mean']:.3f}   {s['recall_mean']:.3f}   "
            f"{s['empty_fp_mean']:.4f}  {s['params']:>10,d} {s['inf_time_mean']:.4f}")
    return "\n".join(lines)
