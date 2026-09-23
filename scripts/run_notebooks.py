"""Reproduce the notebook sequence and refresh the compact CSV collection."""

from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
SUMMARY_SOURCES = {
    "population.csv": "artifacts/generated/tables/01_population.csv",
    "splits.csv": "artifacts/generated/tables/01_splits.csv",
    "voltage_metrics.csv": "artifacts/generated/tables/07_voltage.csv",
    "loss_metrics.csv": "artifacts/generated/tables/07_loss.csv",
    "probability_metrics.csv": "artifacts/generated/tables/07_probability.csv",
    "oof_metrics.csv": "artifacts/generated/tables/07_oof.csv",
    "uncertainty.csv": "artifacts/generated/tables/07_uncertainty.csv",
    "ablations.csv": "artifacts/generated/tables/07_ablations.csv",
    "paired_statistics.csv": "artifacts/generated/tables/07_statistics.csv",
    "efficiency.csv": "artifacts/generated/tables/07_efficiency.csv",
    "decisions.csv": "artifacts/generated/tables/07_decisions.csv",
    "external_comparison.csv": "artifacts/generated/tables/07_external_comparison.csv",
    "external_adaptation.csv": "artifacts/generated/tables/07_external.csv",
    "subgroups.csv": "artifacts/generated/tables/06_subgroups.csv",
    "seed_stability.csv": "artifacts/generated/tables/06_seed_stability.csv",
    "configuration.csv": "artifacts/generated/tables/05_configuration.csv",
    "telemetry_stress.csv": "artifacts/generated/results/telemetry_stress.csv",
}


def main() -> None:
    """Execute in fresh kernels using the active Python environment."""
    notebooks = sorted((PROJECT / "notebooks").glob("0[1-7]_*.ipynb"))
    assert len(notebooks) == 7
    for notebook in notebooks:
        subprocess.run(
            [
                sys.executable,
                str(PROJECT / "scripts/execute_notebook.py"),
                notebook.name,
            ],
            cwd=PROJECT,
            check=True,
        )
    for name, source in SUMMARY_SOURCES.items():
        shutil.copy2(PROJECT / source, PROJECT / "artifacts" / name)


if __name__ == "__main__":
    main()
