# pyrolysis-audit

Version 0.1 provides analysis code for **A passive diagnostic for depletion-omission errors in first-order pyrolysis sources**, by Xuejin Zhou, Xuyang Li, Huazheng Sun, Jiaqi Xu, Ran Tu and Yi Zeng.

The diagnostic assesses omission of reactant availability in a declared first-order pyrolysis source. It compares finite and undepleted sources on the same temperature history. It does not estimate an error in flame-spread speed.

## Run

Python 3.10 or newer is required. From the repository root:

```bash
python -m pip install -r requirements.txt
python endpoint/portable_source_audit.py --self-test
python scripts/fetch_public_data.py
python public_tga/analyze_tga.py
python slab/analyze_slab_gasification.py
```

`requirements-tested.txt` records the principal package versions used for release verification. The endpoint hand checks require only NumPy and work offline. The download command fetches public measurements and material specifications from a pinned MaCFP commit and checks all 31 SHA-256 hashes. It stops if an existing input does not match. No experimental datasets or research-result arrays are bundled in this code-only release.

| Code | Purpose |
|---|---|
| `endpoint/portable_source_audit.py` | Exact source suppression, finite-source-relative bias, Jensen bound and C90 for supplied rate/activity/capacity arrays. |
| `public_tga/analyze_tga.py` | No-refit one-step PMMA analysis of five NIST TGA repeats and the UMD seven-test mean. Output is written beside the script. |
| `slab/analyze_slab_gasification.py` | Existing 1D heated-PMMA model, passive diagnostic, layer-compaction comparison and resolution checks. Outputs are written to `slab/outputs/`. |

The slab command reruns the existing suite, including 450 s trajectories and shorter resolution/sensitivity calculations. Physical end times belong to the cases; there is no hard wall-clock stop. The release preserves the existing numerical algorithm. Packaging changes make paths portable, separate generated outputs, and replace the deprecated NumPy name `trapz` by `trapezoid`.

## Supply your own endpoint arrays

```bash
python endpoint/portable_source_audit.py my_endpoints.npz --output my_audit.csv
```

The NPZ fields are `time_s` with shape `(nt,)`, `k_s_inv` and `chi` with shape `(nt, ny, nx)`, and `M_kg_per_m` with shape `(ny, nx)`. Rates must be nonnegative, activity must lie in [0, 1], and capacities must be positive. The capacity unit is kg per metre of span for the 2D convention used here. Both source sums omit the common gas-yield factor; multiply both by it for mapped gas-fuel rates. Activity must be integrated consistently with the declared law and temperature history.

C90 ranks cells by source per capacity and includes a fractional final cell. The hand checks cover unequal capacities, complete source depletion, and zero-weight cells. Undefined active-source C90 is returned as NaN; zero finite source gives infinite relative bias. These conventions are explicit rather than clipped.

## Scope and verification

The included TGA and complete slab analyses were run before release and matched the local manuscript reference results in the tested environment. The endpoint algorithm was also checked against local accepted states. The endpoint self-test uses small algebraic examples only.

This release does not contain the full 2D CFD solver, processed numerical-result files or complete accepted-state archives. Supporting numerical results and complete 2D histories can be requested from the corresponding authors. It cannot regenerate every 2D manuscript figure from a new cold start.

## Public inputs and attribution

Inputs are downloaded from [MaCFP/matl-db](https://github.com/MaCFP/matl-db/tree/579b88265a329878a68e1161a8b9ceb5e62a9979/PMMA), commit `579b88265a329878a68e1161a8b9ceb5e62a9979`. The gasification measurements are described by Leventon and De Lannoye, [NIST Gasification Apparatus Data for Cast Black PMMA](https://doi.org/10.18434/mds2-2940). Original contributors retain attribution. This study did not collect new experimental data.

The original TGA manifest includes LCPP inputs and metadata. They are downloaded for provenance but not analyzed because the upstream README flags unresolved duplicate-temperature-rate datasets. No NIST or UMD result is excluded based on agreement with the model.

`SHA256SUMS.json` records released code and metadata bytes. Downloaded inputs and generated outputs are ignored by Git. The manuscript describes the governing assumptions and numerical limits. Correspondence: Xuejin Zhou (zhouxj@hqu.edu.cn) and Yi Zeng (antigy@hqu.edu.cn).
