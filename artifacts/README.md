# Numerical results

This directory contains 17 compact CSV summaries. Values are copied from the executed notebook results without rounding or recalculation. The notebooks show the corresponding readable grouped tables.

| CSV | Notebook | Contents |
| --- | --- | --- |
| [population.csv](population.csv) | [01](../notebooks/01_Data_and_Research_Design.ipynb) | Population and observation structure |
| [splits.csv](splits.csv) | [01](../notebooks/01_Data_and_Research_Design.ipynb) | Topology-disjoint subsets and eligible targets |
| [voltage_metrics.csv](voltage_metrics.csv) | [07](../notebooks/07_Comparative_Results.ipynb) | Minimum-voltage accuracy on 605 converged test states |
| [loss_metrics.csv](loss_metrics.csv) | [07](../notebooks/07_Comparative_Results.ipynb) | Active-loss accuracy on the same 605 test states |
| [probability_metrics.csv](probability_metrics.csv) | [07](../notebooks/07_Comparative_Results.ipynb) | Calibrated classification metrics on all 618 test states |
| [oof_metrics.csv](oof_metrics.csv) | [07](../notebooks/07_Comparative_Results.ipynb) | Training-only topology-fold results |
| [uncertainty.csv](uncertainty.csv) | [07](../notebooks/07_Comparative_Results.ipynb) | Nominal coverage, empirical coverage and interval width |
| [ablations.csv](ablations.csv) | [07](../notebooks/07_Comparative_Results.ipynb) | Matched-seed component and objective comparisons |
| [paired_statistics.csv](paired_statistics.csv) | [07](../notebooks/07_Comparative_Results.ipynb) | Effect sizes, cluster intervals and Holm-adjusted tests |
| [efficiency.csv](efficiency.csv) | [07](../notebooks/07_Comparative_Results.ipynb) | Measured training and prediction resources |
| [decisions.csv](decisions.csv) | [07](../notebooks/07_Comparative_Results.ipynb) | Candidate filtering, execution and abstention outcomes |
| [external_comparison.csv](external_comparison.csv) | [07](../notebooks/07_Comparative_Results.ipynb) | Zero-shot comparison on 800 common external states |
| [external_adaptation.csv](external_adaptation.csv) | [07](../notebooks/07_Comparative_Results.ipynb) | Local supervision and zero-shot comparison on 720 common states |
| [subgroups.csv](subgroups.csv) | [06](../notebooks/06_Full_Model_Analysis.ipynb) | Test performance by operating regime |
| [seed_stability.csv](seed_stability.csv) | [06](../notebooks/06_Full_Model_Analysis.ipynb) | Variation across fixed optimization seeds |
| [configuration.csv](configuration.csv) | [05](../notebooks/05_Proposed_Model_and_Ablations.ipynb) | Full-model configuration |
| [telemetry_stress.csv](telemetry_stress.csv) | [06](../notebooks/06_Full_Model_Analysis.ipynb) | Response to controlled telemetry degradation |

A full run writes predictions, fitted estimators, checkpoints, training histories and detailed source tables to `generated/`. These intermediate files are intentionally excluded from version control. `scripts/run_notebooks.py` refreshes the compact CSV summaries only after all seven notebooks complete.

The regression rows use a shared 605-state test set; probability metrics use all 618 test states. External zero-shot results on 800 states and local-adaptation results on 720 states are separate comparisons and must not be pooled. Empty seed standard deviations are not applicable for a single deterministic physical component.
