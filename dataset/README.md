# Simulation dataset

The dataset contains 4,800 independently sampled, balanced steady-state scenarios: 4,000 for the 33-bus benchmark and 800 for the external 69-bus benchmark. These are simulations, not measurements from a utility network.

## Files

| File | Role |
| --- | --- |
| `raw/case33bw_bus.csv`, `raw/case69_bus.csv` | Nominal bus demand used to check the embedded benchmark definitions |
| `raw/case33bw_branch.csv`, `raw/case69_branch.csv` | Branch parameters underlying the benchmark definitions |
| `raw/source_provenance.csv` | Benchmark URLs, original-source DOIs and nominal voltage |
| `processed/scenario_metadata.csv` | One row per scenario: identifiers, split, topology, telemetry conditions and targets |
| `processed/scenario_arrays.npz` | Aligned node features, adjacency, masks, tabular features and targets |
| `processed/case33_topologies.csv` | Identifiers, switching labels, edge counts and split assignments of the 60 primary-network topologies |
| `processed/topology_folds.csv` | Three out-of-fold assignments restricted to training topologies |

Rows in the arrays follow the metadata row order. Preserve that alignment; `sample_id` identifies an observation. Array shapes are `node_features: (4800, 69, 10)`, `adjacency: (4800, 69, 69)`, `node_mask: (4800, 69)`, `tabular: (4800, 21)` and `targets: (4800, 3)`. Networks are padded to 69 nodes and `node_mask` excludes padding.

The target columns are minimum voltage (p.u.), active loss (kW) and feasibility (0/1). Use `converged` in the metadata to identify accepted regression references; all models exclude the same 13 nonconverged test states from regression. Classification retains those states as inadmissible.

Notebook 01 contains the simulation and preparation code, regenerates the prepared arrays with seed 20260922, and checks their scenario/split alignment against the included metadata. No external downloads are needed to run the notebooks after installing the dependencies.

## Source attribution

The electrical benchmarks are derived from MATPOWER `case33bw` and `case69`; their identifiers, source links and original references are recorded in `raw/source_provenance.csv`. The simulation code uses the local numerical definitions in `src/benchmarks.py`.
