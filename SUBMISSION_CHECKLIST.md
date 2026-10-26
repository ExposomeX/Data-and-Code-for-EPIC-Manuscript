# Code and software submission checklist

| Required item | Evidence in this package | Verification |
|---|---|---|
| Source code | `source/` contains 51 Python and R source files across four research stages. | Files inventoried in `FILE_MANIFEST.json`. |
| Small real/simulated demo dataset | `demo/data/` includes sampled model outputs, derived priority evidence, and synthetic county-year hospitalization records (`SYN` IDs). | `demo/run_demo.py` checks schemas and synthetic IDs. |
| OS and dependency versions | `README.md` §1 and `requirements.txt`. | Compact demo and priority dependencies tested; R and full XGBoost stack not tested together. |
| Tested versions | `README.md` §1. | Python/Windows and locally observed scientific package versions recorded. |
| Nonstandard hardware | `README.md` §1. | None needed for compact demo. |
| Installation instructions/time | `README.md` §2. | Demo requires only Python standard library; optional full-stack time is an estimate. |
| Demo instructions/output/time | `README.md` §3, `demo/output/`, `RUN_SUMMARY.md`. | Compact demo executed; actual priority selection executed. |
| Own-data use | `README.md` §4. | Inputs and configuration for each stage listed. |
| Quantitative reproduction guidance (optional) | `README.md` §5. | Full reproduction requires restricted hospitalization and large geospatial/model inputs not supplied here. |

The compact demo checks public interfaces and one actual priority selection. It is not a substitute for reproducing all manuscript estimates. This distinction is stated in the README so that editors and reviewers can assess the package accurately.
