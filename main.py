from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / ".cache"
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines import run_baselines
from data_analysis import plot_age_distribution, plot_correlation_matrix, plot_gender_distribution, plot_race_distribution
from data_utils import load_and_encode
from explainability import run_shap_analysis
from mitigation import get_snapshot_predictions, run_mitigation
from toolkit_comparison import run_toolkit_comparison

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

from plots import plot_accuracy, plot_baseline_confusion_matrices, plot_fairness, plot_post_confusion_matrices, plot_selection_rates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the project pipeline and write all generated outputs into a results folder."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT_DIR / "dataset" / "adult_recruitment.csv",
        help="Input recruitment CSV file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT_DIR / "results",
        help="Directory where all generated files will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio for baseline evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for baseline evaluation.",
    )
    return parser


def write_run_summary(
    results_dir: Path,
    data_path: Path,
    baseline_summary: pd.DataFrame,
    snapshot: pd.DataFrame,
    ttests: pd.DataFrame,
) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    # Phase-by-phase snapshot table (mean ± std per model × variant)
    snapshot_lines = []
    for model_name in sorted(snapshot["model"].unique()):
        snapshot_lines.append(f"\n  {model_name}")
        sub = snapshot[snapshot["model"] == model_name]
        for _, row in sub.iterrows():
            variant = row["variant"]
            snapshot_lines.append(
                f"    [{variant:<11}] "
                f"acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}  "
                f"f1={row['f1_mean']:.4f}±{row['f1_std']:.4f}  "
                f"auc={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f}  "
                f"DI={row['DI_mean']:.4f}±{row['DI_std']:.4f}  "
                f"DPD={row['DPD_mean']:.4f}±{row['DPD_std']:.4f}  "
                f"EOD={row['EOD_mean']:.4f}±{row['EOD_std']:.4f}"
            )

    lines = [
        "Project pipeline completed successfully.",
        f"Dataset: {data_path}",
        f"Results directory: {results_dir}",
        "",
        "Generated files:",
        "- baseline_metrics.csv        (single-seed reference baselines)",
        "- baseline_group_metrics.csv  (per-group breakdown)",
        "- per_run_metrics.csv         (all seeds × models × phases)",
        "- summary_stats.csv           (mean ± std per model × phase)",
        "- paired_ttests.csv           (pre vs in_process, pre vs post)",
        "- fairness_main.png",
        "- accuracy_secondary.png",
        "- confusion_matrices.png",
        "- selection_rates.png",
        "",
        "── Baseline (single seed, no mitigation) ──────────────────────────────",
        baseline_summary.to_string(index=False),
        "",
        "── Phase snapshot (5-seed mean ± std) ─────────────────────────────────",
        "  Phases: pre = no class weights | in_process = balanced weights |",
        "          post = balanced weights + group-specific threshold opt.",
        *snapshot_lines,
        "",
        "── Paired t-tests (pre vs later phases) ───────────────────────────────",
        ttests.to_string(index=False),
    ]
    (results_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Dataset distribution plots (use raw CSV before encoding)
    print("Generating dataset distribution plots...")
    plot_gender_distribution(args.data, results_dir)
    plot_race_distribution(args.data, results_dir)
    plot_age_distribution(args.data, results_dir)

    df = load_and_encode(args.data)
    plot_correlation_matrix(df, results_dir)

    baseline_summary, baseline_groups, baseline_preds = run_baselines(
        df,
        results_dir,
        test_size=args.test_size,
        random_state=args.seed,
    )
    per_run, snapshot, ttests = run_mitigation(df, results_dir)
    # Align baseline sensitive_test index so selection-rate masks work correctly
    baseline_preds["sensitive_test"] = baseline_preds["sensitive"].reset_index(drop=True)

    snap_preds = get_snapshot_predictions(df, seed=args.seed)

    plot_fairness(per_run, ttests, results_dir)
    plot_accuracy(per_run, ttests, results_dir)
    plot_baseline_confusion_matrices(baseline_preds, results_dir)
    plot_post_confusion_matrices(snap_preds, results_dir)
    plot_selection_rates(baseline_preds, results_dir)

    print("\nComputing SHAP explainability plots...")
    run_shap_analysis(df, results_dir)

    print("\nRunning fairness toolkit comparison (AIF360 + FairLearn)...")
    run_toolkit_comparison(df, results_dir)

    write_run_summary(results_dir, args.data, baseline_summary, snapshot, ttests)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n── Baseline metrics ──")
    print(baseline_summary.to_string(index=False))
    print("\n── Baseline group metrics ──")
    print(baseline_groups.to_string(index=False))
    print("\n── Phase snapshot (5-seed mean ± std) ──")
    print(snapshot.to_string(index=False))
    print("\n── Paired t-tests ──")
    print(ttests.to_string(index=False))
    print(f"\nResults written to: {results_dir}")


if __name__ == "__main__":
    main()
