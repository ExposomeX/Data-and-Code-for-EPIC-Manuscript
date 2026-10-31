# EPIC manuscript code and compact demonstration

This package contains the source code for four stages of the manuscript analysis and a compact dataset for checking the public data interfaces. Run commands from the package root.

## 1. System requirements

- **Operating system:** Tested for the compact demo on Windows NT 10.0 build 26200 (64-bit). Linux and macOS have not been tested.
- **Python:** The compact demo was run with CPython 3.12.10 and needs only the standard library. The priority calculation was run with Python 3.12 and pandas 3.0.6, NumPy 2.5.4, SciPy 1.18.1 and Matplotlib 3.11.2. Other local packages observed in the emission environment were xarray 2026.9.0, GeoPandas 1.2.0, Cartopy 0.26.0, netCDF4 1.7.5, openpyxl 3.1.5 and Pillow 13.0.0. Install the packages in `requirements.txt` for the wider Python workflows. The complete dependency set, including XGBoost, scikit-learn and seaborn, was **not** installed and tested together in this packaging session.
- **R:** The association source requires R 4.x with `data.table` (>=1.15), `ggplot2` (>=3.5), `fixest` (>=0.12), and the base R `splines` package. These are compatibility minima, not locally tested versions. R was unavailable on the packaging computer, so its execution and exact package versions could not be verified here.
- **Hardware:** No GPU or other non-standard hardware is required for the compact demo. A conventional 64-bit desktop with at least 8 GB RAM is sufficient for the demo; full-resolution geospatial/model training workflows may need substantially more memory and disk space. The historical proxy NetCDF inputs alone occupy multiple GB.

## 2. Installation

1. Extract the ZIP to a writable directory. Keep the folder structure.
2. For the compact demo, confirm `python --version` reports Python 3.12 and run the command in Section 3. No package installation is needed.
3. For the full Python scripts, create and activate an isolated Python 3.12 virtual environment, then run `python -m pip install -r requirements.txt`.
4. To run the hospitalization models, install R 4.x separately and run `Rscript -e "install.packages(c('data.table','ggplot2','fixest'))"` in an R session with repository access.

**Installation time:** Extracting this compact package and using the standard-library demo generally takes under two minutes on a normal desktop. Installing the optional Python/R scientific stack is estimated at 10–30 minutes, depending on network and binary-wheel availability; this estimate was not timed on a clean desktop.

## 3. Demonstration

From the package root run:

```powershell
python demo/run_demo.py
```

The demo checks four input schemas and writes descriptive outputs into `demo/output/`: `emission_example_summary.csv`, `association_demo_qc.csv`, `priority_demo_epi_screen.csv`, `future_example_summary.csv`, and `demo_run.json`. It should print `"status": "PASS"`. On the packaging computer it read 120 emission, 245 **synthetic** hospitalization, 168 priority and 36 future rows in 0.092 seconds of measured Python execution (about 2 seconds of command wall time). This is an **interface/smoke demo**; the epidemiological score in its priority output is illustrative. It does not refit the XGBoost models, run the R association models, or recreate the manuscript estimates.

The actual released priority-selection script can also be run from its directory after the scientific Python dependencies are installed:

```powershell
cd 'source/Supplementary Code_3_Create_Emission_Index/create_index'
$env:MPLBACKEND='Agg'
$env:PYTHONIOENCODING='utf-8'
python code_3_select_pols.py
```

This script reads the included `code_1_tot_results/3_combined_2_pds_data.csv` and `code_2_results/Disease_Specific_DeepTox_Scores.csv`; it writes four CSV files to `code_3_results/`. The included output has 83 selected disease–pollutant–source rows, and its unique exposure targets agree with the author's original output. It took approximately 8 seconds with the locally available scientific Python environment. The packaged copy omits an unused seaborn import and uses a pandas-3-compatible `groupby` expression; the ranking and selection formula is unchanged. `code_3_results/` contains the observed expected output.

## 4. File map and running on your own data

| Stage | Source folder | Core input and schema | Output and configuration |
|---|---|---|---|
| Historical emissions | `source/Supplementary Code 1_Cal_Emission/` | Production/activity tables, emission factors, province polygons, and six gridded proxy NetCDFs; see `code_3.2_pred_grid_emission.py` and `code_4.1_predict_PAH.py`. The small `demo/data/emission_example.csv` illustrates the gridded output (`Year`, `lat`, `lon`, pollutant–sector columns). | Put inputs under the script's `data_raw/` and the expected prior-stage `results/` folders. `code_3.2_pred_grid_emission.py` accepts `EMISSION_OUTPUT_DIR` for its output directory. `code_5_merge_export_netcdf.py` merges gridded emissions and exports NetCDF; `code_5_plot_selected_emission.py` plots selected maps. These stages require the full upstream inputs, which are not contained in this compact demo. |
| Hospitalization associations | `source/Supplementary Code_2_Epidemic_Association/` | `demo/data/association_example.csv` is fully synthetic and has county/year, outcome (`Y_`), exposure (`X_`), population and covariate columns. It is **not patient data**. | To use the R analysis, set `EWAS_INPUT_FILE` to a county-year CSV with the same schema and `EWAS_OUTPUT_DIR` to a writable folder; run `code_1_hospotal_all_add_eco.R` from its own folder. Output includes association/RR tables. The confidential hospitalization data used in the paper are not distributed here. |
| Biological evidence and priority | `source/Supplementary Code_3_Create_Emission_Index/` | The included aggregate association/PDS table and toxicity score table are the inputs to `create_index/code_3_select_pols.py`. `ebio_network/` computes network evidence; `create_index/code_1_combine_stat_pds.py` and `code_2_process_carbazole_toxicity.py` produce the prior inputs when their full source data are available. | Set input folders as specified at the tops of the individual scripts. The selection script produces `code_3_results/`. Subsequent CEI, scenario burden, economic and lag analyses are in the same `create_index/` directory and require additional full-size input files. |
| Future emissions | `source/Supplementary Code_4_Pred_Future_Emission/` | `demo/data/future_example.csv` illustrates `scenario`, `time`, `lat`, `lon` and pollutant–sector predictions. Full prediction needs scenario-specific proxy, GDP and population tables plus trained XGBoost JSON models. | `code_5_ML_pred.py --data-dir <folder> --model-dir <folder> --output <pred_results_all.csv>` writes one combined future CSV. Training inputs and all fitted models are not in this compact demo. |

For own-data execution, keep the exact column names and units expected by the corresponding source script. Inspect the script's input constants or `--help` before launching a full run. Use a separate output directory where a script exposes an environment variable or CLI option, and retain the original input files unchanged. Do not interpret the small sampled demo data as a representative national estimate; sampling changes normalization and ranking.

## 5. Reproducing manuscript results and access limits

The code is ordered by module and file prefix. Full reproduction starts with emission/activity inputs, generates gridded historical emissions, fits county-year hospitalization associations, combines epidemiological and biological evidence for priority selection, constructs disease-specific CEIs, then projects future emissions and health/economic outcomes. The included compact examples test interfaces and one complete selection step only. The full historical geospatial inputs, full scenario predictors and model files, and ethics-restricted hospitalization microdata must be obtained through the manuscript's Data Availability process. The R analysis and full four-stage reproduction were not verified in this packaging session. This limitation should remain visible to reviewers.

`FILE_MANIFEST.json` records per-file SHA-256 hashes and sizes. `SUBMISSION_CHECKLIST.md` maps the Nature checklist items to the files and flags the verification boundaries.
