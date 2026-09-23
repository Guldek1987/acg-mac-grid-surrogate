# Figures

Numbers 1–14 follow the manuscript captions. English filenames are concise translations of those captions. Each figure is available as PNG and vector PDF; figures 1 and 5 include high-resolution diagram exports.

S1–S6 identify additional notebook diagnostics and use a supplementary numbering sequence specific to this repository.

| Figure | Caption | Reproduction source |
| --- | --- | --- |
| [1 PNG](1_Data_preparation_pipeline.png) · [PDF](1_Data_preparation_pipeline.pdf) | Physically consistent data preparation and topology-disjoint model-ready representations | [Editable diagram](ACG_MAC_diagrams.drawio) |
| [2 PNG](2_Target_and_telemetry_distributions.png) · [PDF](2_Target_and_telemetry_distributions.pdf) | Distributions of physical targets and telemetry characteristics across topology-disjoint subsets | [Notebook 02](../notebooks/02_EDA_and_Preprocessing.ipynb) |
| [3 PNG](3_Training_rank_associations.png) · [PDF](3_Training_rank_associations.pdf) | Rank associations among system characteristics, physical proxies and targets in the training set | [Notebook 02](../notebooks/02_EDA_and_Preprocessing.ipynb) |
| [4 PNG](4_Missingness_and_telemetry_age.png) · [PDF](4_Missingness_and_telemetry_age.pdf) | Joint missingness structure and telemetry-age distribution in the training set | [Notebook 02](../notebooks/02_EDA_and_Preprocessing.ipynb) |
| [5 PNG](5_ACG_MAC_internal_architecture.png) · [PDF](5_ACG_MAC_internal_architecture.pdf) | Internal ACG-MAC architecture and network-state prediction path | [Editable diagram](ACG_MAC_diagrams.drawio) |
| [6 PNG](6_Hidden_width_sensitivity.png) · [PDF](6_Hidden_width_sensitivity.pdf) | Sensitivity of the selection criterion to the ACG-MAC hidden width | [Notebook 05](../notebooks/05_Proposed_Model_and_Ablations.ipynb) |
| [7 PNG](7_Ensemble_training_dynamics.png) · [PDF](7_Ensemble_training_dynamics.pdf) | Training dynamics of five independently initialized ACG-MAC members | [Notebook 05](../notebooks/05_Proposed_Model_and_Ablations.ipynb) |
| [8 PNG](8_Node_pooling_weights.png) · [PDF](8_Node_pooling_weights.pdf) | Learned node-pooling weights as a function of telemetry availability and age | [Notebook 05](../notebooks/05_Proposed_Model_and_Ablations.ipynb) |
| [9 PNG](9_Feasibility_calibration_and_discrimination.png) · [PDF](9_Feasibility_calibration_and_discrimination.pdf) | Calibration and discrimination of feasibility probabilities for selected models | [Notebook 06](../notebooks/06_Full_Model_Analysis.ipynb) |
| [10 PNG](10_Calibrated_feasibility_confusion_matrix.png) · [PDF](10_Calibrated_feasibility_confusion_matrix.pdf) | Confusion matrix of calibrated ACG-MAC feasibility classification | [Notebook 06](../notebooks/06_Full_Model_Analysis.ipynb) |
| [11 PNG](11_Prediction_interval_coverage.png) · [PDF](11_Prediction_interval_coverage.pdf) | Empirical and nominal coverage of ACG-MAC residual prediction intervals | [Notebook 06](../notebooks/06_Full_Model_Analysis.ipynb) |
| [12 PNG](12_Telemetry_degradation_errors.png) · [PDF](12_Telemetry_degradation_errors.pdf) | Regression errors under joint telemetry degradation | [Notebook 06](../notebooks/06_Full_Model_Analysis.ipynb) |
| [13 PNG](13_Conditional_feature_group_reliance.png) · [PDF](13_Conditional_feature_group_reliance.pdf) | ACG-MAC reliance on input-feature groups under conditional block permutation | [Notebook 06](../notebooks/06_Full_Model_Analysis.ipynb) |
| [14 PNG](14_Error_regimes_and_ensemble_disagreement.png) · [PDF](14_Error_regimes_and_ensemble_disagreement.pdf) | High-error regimes and ACG-MAC ensemble disagreement | [Notebook 06](../notebooks/06_Full_Model_Analysis.ipynb) |
| [S1 PNG](S1_Baseline_residuals.png) · [PDF](S1_Baseline_residuals.pdf) | Baseline residual distributions and prediction-dependent errors | [Notebook 03](../notebooks/03_Baseline_Models.ipynb) |
| [S2 PNG](S2_Baseline_training_size_curve.png) · [PDF](S2_Baseline_training_size_curve.pdf) | Baseline error as a function of training-set size | [Notebook 03](../notebooks/03_Baseline_Models.ipynb) |
| [S3 PNG](S3_Graph_comparator_training.png) · [PDF](S3_Graph_comparator_training.pdf) | Graph-comparator training dynamics | [Notebook 04](../notebooks/04_Graph_Comparators.ipynb) |
| [S4 PNG](S4_Graph_training_size_curve.png) · [PDF](S4_Graph_training_size_curve.pdf) | Graph-comparator error as a function of training-set size | [Notebook 04](../notebooks/04_Graph_Comparators.ipynb) |
| [S5 PNG](S5_ACG_MAC_computational_structure.png) · [PDF](S5_ACG_MAC_computational_structure.pdf) | Compact computational structure of ACG-MAC | [Notebook 05](../notebooks/05_Proposed_Model_and_Ablations.ipynb) |
| [S6 PNG](S6_Paired_effect_intervals.png) · [PDF](S6_Paired_effect_intervals.pdf) | Paired error differences with topology-cluster intervals | [Notebook 07](../notebooks/07_Comparative_Results.ipynb) |

Matplotlib figures use Times New Roman, 18–20 pt source text, Unicode minus signs and 350 dpi PNG export. Figures 1 and 5 are schematic drawings. Their editable source is included in the two-page Draw.io file.
