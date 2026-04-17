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
from data_utils import load_and_encode
from mitigation import run_mitigation

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

from plots import plot_accuracy, plot_fairness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the project pipeline and write all generated outputs into a results folder."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT_DIR / "dataset" / "recruitment.csv",
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
    mitigation_summary: pd.DataFrame,
) -> None:
    lines = [
        "Project pipeline completed successfully.",
        f"Dataset: {data_path}",
        f"Results directory: {results_dir}",
        "",
        "Generated files:",
        "- baseline_metrics.csv",
        "- baseline_group_metrics.csv",
        "- per_run_metrics.csv",
        "- summary_stats.csv",
        "- paired_ttests.csv",
        "- fairness_main.png",
        "- accuracy_secondary.png",
        "",
        "Baseline models:",
        baseline_summary.to_string(index=False),
        "",
        "Mitigation summary:",
        mitigation_summary.to_string(index=False),
    ]
    (results_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_encode(args.data)
    baseline_summary, baseline_groups = run_baselines(
        df,
        results_dir,
        test_size=args.test_size,
        random_state=args.seed,
    )
    per_run, mitigation_summary, paired_tests = run_mitigation(df, results_dir)
    plot_fairness(per_run, paired_tests, results_dir)
    plot_accuracy(per_run, paired_tests, results_dir)
    write_run_summary(results_dir, args.data, baseline_summary, mitigation_summary)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("\nBaseline metrics:")
    print(baseline_summary.to_string(index=False))
    print("\nBaseline group metrics:")
    print(baseline_groups.to_string(index=False))
    print("\nMitigation summary:")
    print(mitigation_summary.to_string(index=False))
    print(f"\nGenerated results in: {results_dir}")


if __name__ == "__main__":
    main()
