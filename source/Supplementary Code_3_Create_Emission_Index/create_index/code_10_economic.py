#%%
"""Merged economic and inequality analysis with 2020 national public-hospital costs.

Inputs: corrected county hospitalization changes (code_6.3_results).
Outputs: code_10_results. Seven original disease categories are retained.
AC is a compatibility alias for Delta_H; no disease-risk model is refitted here.
Run: python code_10.1_economic.py
Optional: --data-dir PATH --out PATH --no-maps --tables-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import audit_panel_alignment as _alignment

#%%
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "code_6_results"
OUT = BASE / "code_10_results"
FIG = OUT / "figures"
PRICE_YEAR = 2020
START, END = 2030, 2100
BILLION_RMB = 1e9
SSPS = ["ssp126", "ssp245", "ssp370"]
SSP_LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0"}
SSP_COLORS = {"ssp126": "#52779D", "ssp245": "#619E91", "ssp370": "#C7756D"}
DISEASE_ORDER = ["Circulation", "Digestive", "Respiration", "Blood", "Endocrine", "Skin", "Congenital"]
DISEASE_COLORS = dict(zip(DISEASE_ORDER,
    ["#C66B62", "#75A4B1", "#559D8B", "#52698D", "#C3A367", "#9395B4", "#9BBDAD"]))
# Table 5-8-1, 2021 China Health Statistics Yearbook, data year 2020.
# These are national category costs, NOT measured county-hospital expenditures.
UNIT_COSTS = {"Circulation": 12030.61, "Digestive": 9378.41,
              "Respiration": 6804.46, "Blood": 7944.61,
              "Endocrine": 8105.28, "Skin": 6719.94, "Congenital": 14639.82}

CATEGORY_METADATA = {
    "Blood": (3, "血液、造血器官及免疫疾病", "D50-D89", 793258, 6.76, 147),
    "Endocrine": (4, "内分泌、营养和代谢疾病", "E00-E90", 2853815, 8.56, 147),
    "Circulation": (9, "循环系统疾病", "I00-I99", 14230649, 8.93, 147),
    "Respiration": (10, "呼吸系统疾病", "J00-J99", 10098551, 7.66, 148),
    "Digestive": (11, "消化系统疾病", "K00-K93", 9398521, 7.25, 148),
    "Skin": (12, "皮肤和皮下组织疾病", "L00-L99", 713988, 8.98, 148),
    "Congenital": (17, "先天性畸形、变形和染色体异常", "Q00-Q99", 400412, 7.27, 149),
}
COST_SOURCE = "2021 China Health Statistics Yearbook, Table 5-8-1 (2020 public-hospital discharges)"
SOURCE_URL = "https://www.nhc.gov.cn/mohwsbwstjxxzx/tjtjnj/202305/03b91f308f7843e681e70c63e86781ff/files/1740022548902_91015.pdf"
MONEY_ANNUAL = "Annual inpatient cost change\n(billion USD/year; 2020 prices)"
MONEY_SAVINGS = "Annual inpatient cost savings\n(billion USD/year; 2020 prices)"
MONEY_CUM = "Cumulative inpatient cost savings\n(billion USD; 2020 prices)"
PAIRS = [("ssp370", "ssp245"), ("ssp370", "ssp126"), ("ssp245", "ssp126")]
FIGURE_LOG = []

def unit_cost(disease):
    disease = {"Respiratory": "Respiration"}.get(disease, disease)
    if disease not in UNIT_COSTS:
        raise ValueError(f"No verified category cost: {disease}")
    return UNIT_COSTS[disease]

def csv(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")

def cumulative_trapezoid(values, years):
    values, years = np.asarray(values, float), np.asarray(years, float)
    if (len(values) != len(years) or len(values) == 0
        or not np.isfinite(values).all() or not np.isfinite(years).all()
        or (np.diff(years) <= 0).any()):
        raise ValueError("Integration requires finite rates and strictly increasing years.")
    return np.r_[0., np.cumsum((values[:-1] + values[1:]) * .5 * np.diff(years))]

def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "axes.unicode_minus": True,
    })

def canvas(n=1, columns=None, height=3.1, sharey=False):
    columns = columns or n
    rows = int(np.ceil(n / columns))
    width = 7.2 if columns > 1 else 4.2
    fig, axes = plt.subplots(rows, columns, figsize=(width, height * rows),
                              sharey=sharey, squeeze=False)
    fig.subplots_adjust(left=.23 if columns == 1 else .12, right=.96 if columns == 1 else .98, bottom=.31 if columns == 1 else (.22 if rows == 1 else .12),
                        top=.88, wspace=.45, hspace=.60)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, list(axes[:n])

def shared_legend(fig, handles, labels=None):
    fig.legend(handles=handles, labels=labels, loc="lower center",
               bbox_to_anchor=(.5, .01), ncol=min(len(handles), 4), fontsize=7)

def savefig(fig, name, axes=None):
    # Shared legends sit outside plotting areas so grid/series cannot cross text.
    collected = {}
    for axis in fig.axes:
        if axis.get_legend() is not None:
            handles, labels = axis.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                collected.setdefault(label, handle)
            axis.get_legend().remove()
    if collected and not fig.legends:
        shared_legend(fig, list(collected.values()), list(collected))
    axes = axes or [ax for ax in fig.axes if ax.get_visible() and ax.get_label() != "<colorbar>"]
    if len(axes) > 1:
        for letter, ax in zip("abcdefghijklmnopqrstuvwxyz", axes):
            ax.text(0, 1.025, letter, transform=ax.transAxes, fontsize=9, weight="bold")
        fig.canvas.draw()
        _alignment.require_matplotlib_panel_alignment(
            fig, axes=axes, panel_ids=list("abcdefghijklmnopqrstuvwxyz"[:len(axes)]),
            json_out=FIG / (name + ".alignment.json"), tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, strict=True, require_panel_labels=True)
    else:
        (FIG / (name + ".alignment.json")).write_text(
            json.dumps({"status": "NOT APPLICABLE", "reason": "Single-panel figure"}), encoding="utf-8")
    fig.savefig(FIG / (name + ".png"), dpi=400, bbox_inches="tight", pad_inches=.08)
    fig.savefig(FIG / (name + ".pdf"), dpi=400, bbox_inches="tight", pad_inches=.08)
    fig.savefig(FIG / (name + ".svg"), dpi=400, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)
    FIGURE_LOG.append(name)
    print("  saved", name)

def zero_grid(ax, horizontal=True):
    if horizontal:
        ax.axhline(0, color="#A0A0A0", lw=.6)
        ax.grid(axis="y", color="#E6E6E6", lw=.45)
    else:
        ax.axvline(0, color="#A0A0A0", lw=.6)
        ax.grid(axis="x", color="#E6E6E6", lw=.45)
    ax.set_axisbelow(True)

def validate_and_load():
    records, audit = [], []
    for file in sorted(DATA_DIR.glob("Future_CEI_AC_*.csv")):
        disease, ssp = file.stem[len("Future_CEI_AC_"):].rsplit("_", 1)
        disease = {"Respiratory": "Respiration"}.get(disease, disease)
        if disease not in DISEASE_ORDER or ssp not in SSPS:
            audit.append({"File": file.name, "Status": "Excluded",
                          "Reason": "Outside original seven-disease / three-SSP analysis scope"})
            continue
        g = pd.read_csv(file)
        preferred = "Delta_H_" + disease
        legacy = "AC_" + disease
        value = preferred if preferred in g else legacy
        required = ["unique_id", "time", value, "Future_CEI", "population"]
        missing = set(required) - set(g)
        if missing:
            raise ValueError(f"{file.name}: missing columns {sorted(missing)}")
        if g.duplicated(["unique_id", "time"]).any():
            raise ValueError(f"{file.name}: duplicate county-year keys")
        if preferred in g and legacy in g and not np.allclose(g[preferred], g[legacy], equal_nan=True):
            raise ValueError(f"{file.name}: Delta_H and AC compatibility alias disagree")
        s = g[required].rename(columns={value: "AC"}).copy()
        for col in ["unique_id", "time", "AC", "Future_CEI", "population"]:
            s[col] = pd.to_numeric(s[col], errors="raise")
        if not np.isfinite(s.to_numpy()).all() or (s.population <= 0).any():
            raise ValueError(f"{file.name}: missing/non-finite outputs or nonpositive population")
        if not np.equal(s.time, s.time.astype(int)).all():
            raise ValueError(f"{file.name}: noninteger years")
        s.time = s.time.astype(int)
        s["Disease"], s["SSP"] = disease, ssp
        records.append(s)
        audit.append({"File": file.name, "Status": "Included", "Value_column": value,
                      "Rows": len(s), "Counties": s.unique_id.nunique(),
                      "Risk_method": g.Risk_method.iloc[0] if "Risk_method" in g else "Legacy AC; inspect upstream estimator",
                      "Reason": ""})
    if not records:
        raise FileNotFoundError(f"No usable model outputs: {DATA_DIR}")
    panel = pd.concat(records, ignore_index=True)
    expected = set(np.arange(START, END + 1, 10))
    coverage = []
    for disease in DISEASE_ORDER:
        for ssp in SSPS:
            sub = panel[(panel.Disease == disease) & (panel.SSP == ssp)]
            if set(sub.time) != expected:
                raise ValueError(f"{disease}/{ssp}: all eight decadal horizons required")
        for year in sorted(expected):
            groups = [panel[(panel.Disease == disease) & (panel.time == year) & (panel.SSP == s)]
                      .set_index("unique_id") for s in SSPS]
            if any(set(g.index) != set(groups[0].index) for g in groups[1:]):
                raise ValueError(f"{disease}/{year}: county coverage differs across scenarios")
            coverage.append({"Disease": disease, "Year": year, "Counties": len(groups[0])})
    csv(pd.DataFrame(audit), "Input_file_audit.csv")
    csv(pd.DataFrame(coverage), "County_coverage_by_disease_year.csv")
    csv(panel, "County_model_input_used.csv")
    return panel

def load_macro(panel):
    rows = []
    for ssp in SSPS:
        file = BASE / "data_raw" / f"county_level_data_{ssp}_new.csv"
        m = pd.read_csv(file, usecols=["unique_id", "time", "GDP", "population"])
        if m.duplicated(["unique_id", "time"]).any():
            raise ValueError(f"Duplicate macroeconomic county-year keys: {file.name}")
        m = m.rename(columns={"population": "Macro_population"})
        rows.append(m.assign(SSP=ssp))
    macro = pd.concat(rows, ignore_index=True)
    # GDP-per-capita ranking retains the original SSP macroeconomic definition.
    macro["GDP_per_capita"] = macro.GDP / macro.Macro_population.where(macro.Macro_population > 0)
    # Concentration/Lorenz weights use the same population as the health estimate.
    csv(macro, "Macro_data_used.csv")
    return macro

def national_panel(panel):
    return panel.groupby(["time", "SSP", "Disease"], as_index=False).agg(
        AC=("AC", "sum"), Future_CEI=("Future_CEI", "sum"),
        Population=("population", "sum"), N_counties=("unique_id", "nunique"))

def annualise(national):
    records = []
    years = np.arange(START, END + 1)
    for (disease, ssp), g in national.groupby(["Disease", "SSP"], sort=False):
        g = g.sort_values("time")
        row = pd.DataFrame({"Year": years, "Disease": disease, "SSP": ssp})
        for col in ["AC", "Future_CEI"]:
            row[col] = np.interp(years, g.time, g[col])
        row["Unit_cost_CNY_per_admission"] = unit_cost(disease)
        row["Annual_cost_change_CNY"] = row.AC * unit_cost(disease)
        row["Annual_cost_change_billion_CNY"] = row.Annual_cost_change_CNY / BILLION_RMB
        row["Cumulative_cost_change_billion_CNY"] = cumulative_trapezoid(row.Annual_cost_change_billion_CNY, years)
        row["Price_year"] = PRICE_YEAR
        row["Value_origin"] = np.where(np.isin(years, g.time), "Decadal model output", "Linear interpolation")
        records.append(row)
    return pd.concat(records, ignore_index=True)

def benefits(annual, costs=None, discount_rate=0):
    costs = costs or UNIT_COSTS
    records = []
    for hi, lo in PAIRS:
        for disease in DISEASE_ORDER:
            a = annual[(annual.Disease == disease) & (annual.SSP == hi)].set_index("Year").sort_index()
            b = annual[(annual.Disease == disease) & (annual.SSP == lo)].set_index("Year").sort_index()
            if not a.index.equals(b.index):
                raise ValueError("Scenario horizons must match")
            years = a.index.to_numpy()
            delta_h = (a.AC - b.AC).to_numpy()
            delta_cei = (a.Future_CEI - b.Future_CEI).to_numpy()
            money = delta_h * costs[disease] / BILLION_RMB
            discounted = money / (1 + discount_rate) ** (years - START)
            row = pd.DataFrame({
                "Year": years, "Disease": disease, "Baseline": hi, "Scenario": lo,
                "Annual_hospitalization_difference": delta_h,
                "Annual_benefit_billion_CNY": money,
                "Discounted_annual_benefit_billion_CNY": discounted,
                "Cumulative_benefit_billion_CNY": cumulative_trapezoid(discounted, years),
                "Annual_sum_cumulative_billion_CNY": np.cumsum(discounted),
                "Cumulative_hospitalization_difference": cumulative_trapezoid(delta_h, years),
                "Annual_CEI_difference": delta_cei,
                "Integrated_CEI_difference_index_years": cumulative_trapezoid(delta_cei, years),
                "Discount_rate": discount_rate, "Discount_base_year": START,
            })
            records.append(row)
    result = pd.concat(records, ignore_index=True)
    keys = ["Year", "Baseline", "Scenario"]
    for stem, col in [("Annual", "Annual_benefit_billion_CNY"), ("Cumulative", "Cumulative_benefit_billion_CNY")]:
        total = result.groupby(keys)[col].transform("sum")
        magnitude = result[col].abs().groupby([result[k] for k in keys]).transform("sum")
        result[stem + "_signed_contribution_percent"] = result[col] / total.where(total.abs() > 1e-12) * 100
        result[stem + "_absolute_magnitude_share_percent"] = result[col].abs() / magnitude.where(magnitude > 1e-12) * 100
    return result

def write_tables(national, annual, delta):
    metadata = []
    for disease in DISEASE_ORDER:
        index, label, icd, discharges, stay, page = CATEGORY_METADATA[disease]
        metadata.append({
            "Disease": disease, "Source_category_number": index, "Source_category": label,
            "Expected_ICD10_range": icd, "Outcome_ICD_mapping_status": "Verify against study outcome definition",
            "Unit_cost_CNY_per_admission": unit_cost(disease), "Source_discharges": discharges,
            "Source_mean_stay_days": stay, "Data_year": 2020, "Price_year": 2020,
            "Hospital_scope": "National public hospitals; not county-specific",
            "Aggregation": "Published disease-category mean; no averaging of selected diagnoses",
            "Source": COST_SOURCE, "Source_URL": SOURCE_URL, "Printed_page": page,
            "Inflation_adjustment": "None", "PPP_conversion": "None",
        })
    csv(pd.DataFrame(metadata), "Table_S13_unit_costs_and_assumptions.csv")
    national = national.assign(
        Unit_cost_CNY_per_admission=national.Disease.map(UNIT_COSTS),
        Annual_cost_change_billion_CNY=national.AC * national.Disease.map(UNIT_COSTS) / BILLION_RMB,
        Price_year=PRICE_YEAR)
    csv(national, "Table_S14_decadal_cost_change.csv")
    csv(annual, "Table_S14_annual_interpolated_cost_change.csv")
    csv(delta, "Table_S15_annual_and_cumulative_savings.csv")
    end = delta[delta.Year == END].copy()
    denominator = end.Integrated_CEI_difference_index_years
    end["Cost_savings_CNY_per_CEI_index_year"] = (
        end.Cumulative_benefit_billion_CNY * BILLION_RMB /
        denominator.where(denominator.abs() > 1e-12))
    csv(end, "Table_S16_integrated_savings_per_CEI.csv")
    csv(delta[delta.Baseline == "ssp370"], "11_cumulative_cobenefit_vs_ssp370.csv")
    sensitivity = []
    for name, costs, price_year in [
        ("National_category_costs_2020", UNIT_COSTS, "2020")]:
        for rate in [0., .03]:
            tab = benefits(annual, costs, rate)
            for (hi, lo), g in tab[tab.Year == END].groupby(["Baseline", "Scenario"]):
                sensitivity.append({
                    "Cost_schedule": name, "Cost_price_year": price_year, "Baseline": hi, "Scenario": lo,
                    "Discount_rate": rate, "Discount_base_year": START,
                    "Cumulative_savings_billion_CNY": g.Cumulative_benefit_billion_CNY.sum(),
                    "Annual_sum_71_years_billion_CNY": g.Annual_sum_cumulative_billion_CNY.sum(),
                    "Excluding_Skin_Congenital_billion_CNY": g.loc[
                        ~g.Disease.isin(["Skin", "Congenital"]), "Cumulative_benefit_billion_CNY"].sum()})
    summary = pd.DataFrame(sensitivity)
    csv(summary, "Economic_headline_and_sensitivity.csv")
    disease_stats = national.groupby(["SSP", "Disease"]).AC.describe().reset_index()
    cumulative = []
    for (ssp, disease), g in annual.groupby(["SSP", "Disease"]):
        cumulative.append({"SSP": ssp, "Disease": disease,
            "Cumulative_hospitalization_change_2030_2100": cumulative_trapezoid(g.AC, g.Year)[-1]})
    csv(disease_stats.merge(pd.DataFrame(cumulative), on=["SSP", "Disease"]), "04_disease_stats_summary.csv")
    with pd.ExcelWriter(OUT / "Economic_tables_2020_CNY.xlsx") as writer:
        for name, data in [("S13_unit_costs", pd.DataFrame(metadata)), ("S14_decadal", national),
                           ("S14_annual", annual), ("S15_scenario_differences", delta),
                           ("S16_savings_per_CEI", end), ("Headline_sensitivity", summary)]:
            data.to_excel(writer, sheet_name=name, index=False)
    return summary

def stack_absolute_relative(pivot, title, name, ylabel):
    pivot = pivot.reindex(columns=DISEASE_ORDER).sort_index()
    if pivot.isna().any().any():
        raise ValueError("Missing disease-year values cannot be filled with zero for plotting")
    shares = pivot.abs().div(pivot.abs().sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig, axes = canvas(2, height=3.4)
    x = np.arange(len(pivot))
    positive, negative, bottom = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
    for disease in DISEASE_ORDER:
        values = pivot[disease].to_numpy()
        base = np.where(values >= 0, positive, negative)
        axes[0].bar(x, values, bottom=base, color=DISEASE_COLORS[disease], width=.72,
                    edgecolor="white", linewidth=.25)
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
        pct = shares[disease].to_numpy()
        axes[1].bar(x, pct, bottom=bottom, color=DISEASE_COLORS[disease], width=.72,
                    edgecolor="white", linewidth=.25)
        bottom += np.nan_to_num(pct)
    axes[0].set_ylabel(ylabel)
    axes[1].set_ylabel("Share of absolute magnitude (%)")
    axes[1].set_ylim(0, 100)
    axes[1].yaxis.set_major_formatter(PercentFormatter(100))
    for ax, label in zip(axes, ["Signed amount", "Absolute-magnitude contribution"]):
        ax.set_xticks(x, pivot.index.astype(int), rotation=45, ha="right")
        ax.set_title(label, fontsize=8)
        zero_grid(ax)
    fig.suptitle(title, fontsize=9, y=.99)
    shared_legend(fig, [Patch(color=DISEASE_COLORS[d], label=d) for d in DISEASE_ORDER])
    savefig(fig, name, axes)
    csv(shares.rename_axis("Year").reset_index(), name + "_relative_source.csv")

def plot_trajectories(national, annual, delta):
    fig, axes = canvas(7, columns=3, height=2.25)
    for ax, disease in zip(axes, DISEASE_ORDER):
        for ssp in SSPS:
            g = national[(national.Disease == disease) & (national.SSP == ssp)]
            ax.plot(g.time, g.AC / 1e4, color=SSP_COLORS[ssp], lw=1.4, marker="o", ms=2)
        ax.set_title(disease)
        ax.set_xlabel("Year")
        ax.set_ylabel("Hospitalization change\n(10,000 admissions/year)")
        zero_grid(ax)
    shared_legend(fig, [Line2D([], [], color=SSP_COLORS[s], label=SSP_LABELS[s]) for s in SSPS])
    savefig(fig, "01_SSP_trajectories_by_disease", axes)
    csv(national, "01_national_AC_by_disease_ssp.csv")
    fig, axes = canvas(3, height=3.3, sharey=True)
    for ax, ssp in zip(axes, SSPS):
        for disease in DISEASE_ORDER:
            g = national[(national.Disease == disease) & (national.SSP == ssp)]
            ax.plot(g.time, g.AC / 1e4, color=DISEASE_COLORS[disease], lw=1.3)
        ax.set_title(SSP_LABELS[ssp])
        ax.set_xlabel("Year")
        zero_grid(ax)
    axes[0].set_ylabel("Hospitalization change\n(10,000 admissions/year)")
    shared_legend(fig, [Line2D([], [], color=DISEASE_COLORS[d], label=d) for d in DISEASE_ORDER])
    savefig(fig, "04_intra_SSP_disease_comparison", axes)
    fig, axes = canvas(2, height=3.3)
    total = annual.groupby(["Year", "SSP"], as_index=False).agg(
        AC=("AC", "sum"), Cost=("Annual_cost_change_billion_CNY", "sum"))
    for ssp in SSPS:
        g = total[total.SSP == ssp]
        axes[0].plot(g.Year, g.AC / 1e4, color=SSP_COLORS[ssp], lw=1.5)
        axes[1].plot(g.Year, g.Cost, color=SSP_COLORS[ssp], lw=1.5)
    axes[0].set_ylabel("Hospitalization change\n(10,000 admissions/year)")
    axes[1].set_ylabel(MONEY_ANNUAL)
    for ax in axes:
        ax.set_xlabel("Year")
        zero_grid(ax)
    shared_legend(fig, [Line2D([], [], color=SSP_COLORS[s], label=SSP_LABELS[s]) for s in SSPS])
    savefig(fig, "09_national_scenario_gap", axes)
    csv(total, "09_national_AC_cost_by_ssp.csv")
    fig, axes = canvas(2, height=3.3)
    for ssp in SSPS:
        g = total[total.SSP == ssp]
        axes[0].plot(g.Year, g.Cost, color=SSP_COLORS[ssp], lw=1.5, label=SSP_LABELS[ssp])
    cumulative = delta[delta.Baseline == "ssp370"].groupby(["Year", "Scenario"], as_index=False).Cumulative_benefit_billion_CNY.sum()
    for ssp in ["ssp126", "ssp245"]:
        g = cumulative[cumulative.Scenario == ssp]
        axes[1].plot(g.Year, g.Cumulative_benefit_billion_CNY, color=SSP_COLORS[ssp], lw=1.5, label=SSP_LABELS[ssp])
    axes[0].set_ylabel(MONEY_ANNUAL)
    axes[1].set_ylabel(MONEY_CUM)
    axes[0].set_title("Change relative to 2023 reference")
    axes[1].set_title("Savings relative to SSP3-7.0")
    for ax in axes:
        ax.set_xlabel("Year")
        ax.legend(fontsize=6.5, loc="best")
        zero_grid(ax)
    savefig(fig, "Fig5_economic_costs_2020_CNY", axes)
    fig, axes = canvas(1)
    for ssp in ["ssp126", "ssp245"]:
        g = cumulative[cumulative.Scenario == ssp]
        axes[0].plot(g.Year, g.Cumulative_benefit_billion_CNY, color=SSP_COLORS[ssp], label=SSP_LABELS[ssp], lw=1.7)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel(MONEY_CUM)
    axes[0].legend()
    zero_grid(axes[0])
    savefig(fig, "11_cumulative_cobenefit_vs_ssp370")

def plot_compositions(national, delta):
    nat = national.assign(Cost=national.AC * national.Disease.map(UNIT_COSTS) / BILLION_RMB)
    csv(nat, "03_monetization_absolute_detail.csv")
    for ssp in SSPS:
        g = nat[nat.SSP == ssp]
        stack_absolute_relative(g.pivot(index="time", columns="Disease", values="Cost"),
            SSP_LABELS[ssp] + " | cost-change composition", "03_burden_abs_rel_" + ssp, MONEY_ANNUAL)
    snapshots = delta[delta.Year.isin(national.time.unique())]
    for hi, lo in PAIRS + [("ssp245", "ssp370")]:
        if (hi, lo) in PAIRS:
            g = snapshots[(snapshots.Baseline == hi) & (snapshots.Scenario == lo)].copy()
        else:
            g = snapshots[(snapshots.Baseline == lo) & (snapshots.Scenario == hi)].copy()
            g["Baseline"], g["Scenario"] = hi, lo
            for column in ["Annual_benefit_billion_CNY", "Annual_hospitalization_difference",
                           "Discounted_annual_benefit_billion_CNY", "Cumulative_benefit_billion_CNY",
                           "Annual_sum_cumulative_billion_CNY", "Cumulative_hospitalization_difference",
                           "Annual_CEI_difference", "Integrated_CEI_difference_index_years"]:
                g[column] *= -1
        name = f"03_cobenefits_vs_{hi}_{lo}_abs_rel"
        csv(g, name + "_detail.csv")
        stack_absolute_relative(g.pivot(index="Year", columns="Disease", values="Annual_benefit_billion_CNY"),
            SSP_LABELS[lo] + " relative to " + SSP_LABELS[hi], name, MONEY_SAVINGS)
        stack_absolute_relative(g.pivot(index="Year", columns="Disease", values="Annual_hospitalization_difference") / 1e4,
            SSP_LABELS[lo] + " relative to " + SSP_LABELS[hi], name + "_hospitalizations",
            "Annual hospitalization difference\n(10,000 admissions/year)")
    plot_snapshot(national)

def plot_snapshot(national):
    fig, axes = canvas(1, height=3.3)
    x = np.arange(len(DISEASE_ORDER))
    for i, ssp in enumerate(SSPS):
        g = national[(national.time == 2050) & (national.SSP == ssp)].set_index("Disease").reindex(DISEASE_ORDER)
        axes[0].bar(x + (i - 1) * .24, g.AC / 1e4, width=.24, color=SSP_COLORS[ssp], label=SSP_LABELS[ssp])
    axes[0].set_xticks(x, DISEASE_ORDER, rotation=65, ha="right", rotation_mode="anchor")
    axes[0].set_ylabel("Hospitalization change in 2050\n(10,000 admissions/year)")
    axes[0].legend()
    zero_grid(axes[0])
    savefig(fig, "10_scenario_grouped_bars_2050")

def plot_delta_heatmap(delta):
    fig, axes = canvas(3, height=3.1, sharey=True)
    snapshots = delta[delta.Year % 10 == 0]
    matrices = []
    for hi, lo in PAIRS:
        matrices.append(snapshots[(snapshots.Baseline == hi) & (snapshots.Scenario == lo)]
            .pivot(index="Disease", columns="Year", values="Annual_hospitalization_difference")
            .reindex(DISEASE_ORDER) / 1e4)
    limit = max(np.nanmax(np.abs(m.to_numpy())) for m in matrices)
    for ax, (hi, lo), matrix in zip(axes, PAIRS, matrices):
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=60, ha="right")
        ax.set_yticks(range(len(DISEASE_ORDER)), DISEASE_ORDER)
        ax.set_title(SSP_LABELS[hi] + " minus " + SSP_LABELS[lo], fontsize=8)
        ax.set_xlabel("Year")
        csv(matrix.rename_axis("Disease").reset_index(), f"08_scenario_delta_{hi}_{lo}.csv")
    fig.subplots_adjust(left=.14, right=.87, wspace=.18, bottom=.24, top=.85)
    cax = fig.add_axes([.91, .25, .015, .60])
    fig.colorbar(image, cax=cax, label="Hospitalization difference (10,000/year)")
    savefig(fig, "08_scenario_delta_AC_heatmap", axes)

def plot_integrated_metric(delta):
    end = delta[delta.Year == END].copy()
    end["Savings_per_CEI"] = end.Cumulative_benefit_billion_CNY * BILLION_RMB / end.Integrated_CEI_difference_index_years.where(
        end.Integrated_CEI_difference_index_years.abs() > 1e-12)
    csv(end, "07_integrated_savings_per_CEI_all_transitions.csv")
    fig, axes = canvas(1, height=3.5)
    y = np.arange(len(DISEASE_ORDER))
    for i, (hi, lo) in enumerate(PAIRS):
        g = end[(end.Baseline == hi) & (end.Scenario == lo)].set_index("Disease").reindex(DISEASE_ORDER)
        axes[0].scatter(g.Savings_per_CEI, y + (i - 1) * .18, s=22,
                         marker=["o", "s", "^"][i], color=SSP_COLORS[lo],
                         label=SSP_LABELS[hi] + " to " + SSP_LABELS[lo])
    axes[0].set_yticks(y, DISEASE_ORDER)
    axes[0].set_xlabel("Integrated savings / CEI difference\n(CNY per CEI index-year; 2020 prices)")
    axes[0].legend(fontsize=6.5)
    zero_grid(axes[0], horizontal=False)
    savefig(fig, "07_Marginal_Health_ROI_by_transition")
    # Negative results remain visible; the metric is not an investment ROI.

def concentration_curve(population, value, ranking):
    d = pd.DataFrame({"population": population, "value": value, "ranking": ranking})
    d = d.sort_values("ranking", kind="stable")
    if (d.population <= 0).any() or (d.value < 0).any() or d.value.sum() <= 0:
        return np.nan, np.array([0., 1.]), np.array([np.nan, np.nan])
    # Pool ties to avoid arbitrary ordering within identical GDP/rate values.
    d = d.groupby("ranking", sort=True)[["population", "value"]].sum()
    x = np.r_[0., (d.population / d.population.sum()).cumsum()]
    y = np.r_[0., (d.value / d.value.sum()).cumsum()]
    index = 1 - np.sum((y[1:] + y[:-1]) * np.diff(x))
    return float(index), x, y

def plot_inequality(panel, macro):
    year, ssp = 2050, "ssp245"
    g = panel[(panel.time == year) & (panel.SSP == ssp)].merge(
        macro[(macro.time == year) & (macro.SSP == ssp)][["unique_id", "GDP_per_capita"]],
        on="unique_id", how="left", validate="many_to_one")
    missing = g.GDP_per_capita.isna() | ~np.isfinite(g.GDP_per_capita)
    csv(g.loc[missing, ["unique_id", "Disease", "time"]], "Inequality_missing_GDP_exclusions.csv")
    g = g.loc[~missing]
    summaries, curves = [], []
    for disease in DISEASE_ORDER:
        sub = g[g.Disease == disease]
        fig, axes = canvas(2, height=3.2)
        for ax, kind, value, color in [
            (axes[0], "Decreases", (-sub.AC).clip(lower=0), "#52779D"),
            (axes[1], "Increases", sub.AC.clip(lower=0), "#C7756D")]:
            # Retain every valid county in the reference population, including zero changes.
            ci, x, y = concentration_curve(sub.population, value, sub.GDP_per_capita)
            summaries.append({"Disease": disease, "Type": kind, "CI": ci,
                "n_counties": len(sub), "n_nonzero_counties": int((value > 0).sum()),
                "Year": year, "SSP": ssp, "Population_basis": "Upstream health-model population"})
            if np.isfinite(ci):
                ax.plot(x, y, color=color, lw=1.6, label=f"CI = {ci:.2f}")
                ax.legend()
                curves.extend({"Disease": disease, "Type": kind, "Cumulative_population": a,
                               "Cumulative_change": b} for a, b in zip(x, y))
            else:
                ax.text(.5, .5, "No positive magnitude", transform=ax.transAxes, ha="center")
            ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=.7)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(kind)
            ax.set_xlabel("Cumulative population (poorer to richer)")
            ax.set_ylabel("Cumulative share of " + kind.lower())
        fig.suptitle(disease + " | 2050 | SSP2-4.5", fontsize=9, y=.99)
        savefig(fig, "02_concentration_curve_" + disease, axes)
    ci = pd.DataFrame(summaries)
    csv(ci, "02_concentration_index_2050_ssp245.csv")
    csv(pd.DataFrame(curves), "02_concentration_curve_source.csv")
    fig, axes = canvas(1, height=3.4)
    for i, (kind, color) in enumerate([("Decreases", "#52779D"), ("Increases", "#C7756D")]):
        values = ci[ci.Type == kind].set_index("Disease").reindex(DISEASE_ORDER).CI
        axes[0].scatter(values, np.arange(7) + (i - .5) * .20, color=color, s=24, label=kind)
    axes[0].set_yticks(range(7), DISEASE_ORDER)
    axes[0].set_xlim(-1, 1)
    axes[0].set_xlabel("Income-related concentration index")
    axes[0].legend()
    zero_grid(axes[0], False)
    savefig(fig, "02_CI_summary_2050")
    # Gini applies to nonnegative magnitude, not signed net changes.
    county = panel[(panel.time == year) & (panel.SSP == ssp)].groupby("unique_id").agg(
        AC=("AC", "sum"), population=("population", "first"))
    consistency = panel[(panel.time == year) & (panel.SSP == ssp)].groupby("unique_id").population.nunique()
    if (consistency > 1).any():
        raise ValueError("Disease-specific populations differ for the same county; cannot pool Gini weights")
    fig, axes = canvas(2, height=3.2)
    rows = []
    for ax, kind, values, color in [
        (axes[0], "Decreases", (-county.AC).clip(lower=0), "#52779D"),
        (axes[1], "Increases", county.AC.clip(lower=0), "#C7756D")]:
        index, x, y = concentration_curve(county.population, values, values / county.population)
        rows.append({"Year": year, "SSP": ssp, "Type": kind, "Gini": index,
                     "n_counties": len(county), "n_nonzero_counties": int((values > 0).sum())})
        if np.isfinite(index):
            ax.plot(x, y, color=color, lw=1.6, label=f"Gini = {index:.2f}")
            ax.legend()
        ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=.7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(kind)
        ax.set_xlabel("Cumulative population (ranked by rate)")
        ax.set_ylabel("Cumulative share of magnitude")
    savefig(fig, "05_gini_lorenz_2050", axes)
    csv(pd.DataFrame(rows), "05_gini_2050.csv")

def plot_maps(panel):
    import geopandas as gpd
    from matplotlib.colors import AsinhNorm
    candidates = list((BASE / "data_raw").glob("**/*县*面*.shp"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one county boundary file, found {candidates}")
    shapes = gpd.read_file(candidates[0])
    shapes["unique_id"] = shapes.index
    if shapes.crs is None:
        raise ValueError("County shapefile CRS is undefined")
    shapes = shapes.to_crs(4326)
    for disease in DISEASE_ORDER:
        g = panel[(panel.time == 2050) & (panel.Disease == disease)]
        # Full extrema are preserved; zero is the common midpoint.
        limit = max(float(g.AC.abs().max()), 1.)
        linear_width = max(float(g.AC.abs().median()), .1)
        norm = AsinhNorm(linear_width=linear_width, vmin=-limit, vmax=limit)
        fig, axes = canvas(3, height=2.6)
        for ax, ssp in zip(axes, SSPS):
            sub = shapes.merge(g[g.SSP == ssp][["unique_id", "AC"]], on="unique_id", how="left",
                               validate="one_to_one")
            sub.plot(column="AC", ax=ax, cmap="RdBu_r", norm=norm, edgecolor="#BBBBBB",
                     linewidth=.08, rasterized=True, missing_kwds={"color": "#F0F0F0"})
            ax.set_xlim(73, 135); ax.set_ylim(18, 54)
            ax.set_aspect("equal")
            ax.set_title(SSP_LABELS[ssp], fontsize=8)
            ax.set_axis_off()
        fig.subplots_adjust(left=.02, right=.91, bottom=.18, top=.85, wspace=.04)
        cax = fig.add_axes([.94, .25, .012, .48])
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax,
                     label="Annual hospitalization change (asinh scale)")
        color_ticks = norm.inverse(np.array([0., .25, .5, .75, 1.]))
        colorbar.set_ticks(color_ticks)
        colorbar.set_ticklabels([f"{v:,.0f}" if abs(v) >= 1 else f"{v:.2g}" for v in color_ticks])
        colorbar.ax.tick_params(labelsize=7)
        shared_legend(fig, [Patch(color="#F0F0F0", label="No modeled county data")])
        fig.suptitle(disease + " | 2050", fontsize=9, y=.98)
        savefig(fig, "06_spatial_" + disease + "_2050", axes)

def write_notes(summary):
    main = summary[(summary.Cost_schedule == "National_category_costs_2020") & (summary.Discount_rate == 0)]
    lines = [
        "# Economic analysis: national disease-category cost schedule",
        "Input: " + str(DATA_DIR),
        "The original seven-disease scope is retained; Cancer, Neural and all-cause outputs are outside this scope.",
        "## Methods",
        "Disease-specific mean inpatient medical expenditures were obtained from the 2021 China Health Statistics Yearbook, Table 5-8-1, using 2020 public-hospital discharge statistics. Published category-level costs were matched to the study outcomes and applied uniformly across counties and scenarios. Monetary estimates are expressed in CNY at 2020 prices, without inflation or PPP adjustment. The costs provide a common national valuation schedule rather than measured county-hospital expenditures. The ICD-10 mapping should be verified against the study outcome definitions.",
        "Annual modeled hospitalization changes relative to the upstream 2023 reference were multiplied by the corresponding category cost. Scenario-associated cost savings were calculated as the higher-emission pathway cost change minus the comparison pathway cost change. Signed differences were retained. These comparisons include the SSP-specific population trajectories and are not an isolated emission-only effect or a causal estimate of actual budget savings.",
        "Cumulative savings were calculated by trapezoidal integration from 2030 to 2100 (70 years). Annual values between decadal model outputs were linearly interpolated. The inclusive sum of 71 annual snapshots is provided separately as an accounting sensitivity, not the primary cumulative estimator. A 3% discount-rate sensitivity uses 2030 as the discount reference. There is no extrapolation outside 2030–2100.",
        "The previous cost schedule is a separate sensitivity analysis and has not been harmonized to 2020 prices; its difference from the main estimate reflects both costing approach and source-year differences. Sensitivity analyses are assumptions, not statistical confidence intervals.",
        "## Inequality definitions",
        "GDP-per-capita ranking retains the original SSP macroeconomic panel. Population weights use the population in the upstream health estimates. Concentration curves include all eligible counties, with zero values retained, and pool equal ranking values. Decreases and increases are analyzed separately. The Gini plots show nonnegative decreases/increases in net seven-category hospitalization change; negative values are not silently clipped into an undifferentiated burden.",
        "## Table S13",
        "National disease-category inpatient costs used to value modeled hospitalization changes. Columns document source category, expected ICD-10 range, category cost, discharges, mean stay, data/price year, hospital scope and original cost-schedule sensitivity. Skin and Congenital now use published disease-category costs; they are no longer fallback values.",
        "## Table S14",
        "Modeled annual inpatient cost changes relative to the upstream 2023 reference, in billion CNY/year at 2020 prices. Decadal model outputs and interpolated annual values are supplied separately.",
        "## Table S15",
        "Annual and cumulative scenario-associated inpatient cost savings, retaining signed disease differences. Relative contributions are supplied both as signed shares of the net total and shares of absolute magnitudes. The two denominators differ when disease signs differ.",
        "## Table S16",
        "Integrated inpatient cost savings per integrated CEI difference over 2030–2100, expressed in CNY per CEI index-year. This is an index-based comparison, not an economic investment return; near-zero denominators yield missing ratios.",
        "## Figure 5 caption",
        "Scenario-associated inpatient cost changes and cumulative savings under a national disease-specific cost schedule. a, Annual modeled inpatient cost changes relative to the 2023 reference under SSP1-2.6, SSP2-4.5 and SSP3-7.0. b, Cumulative modeled inpatient cost savings under SSP1-2.6 and SSP2-4.5 relative to SSP3-7.0, integrated from 2030 to 2100. All seven disease categories use published 2020 national public-hospital category costs. Monetary values are in billion CNY at 2020 prices. Annual trajectories are linearly interpolated between decadal outputs. The main estimates are undiscounted; no statistical confidence intervals are shown.",
        "## Headline estimates",
    ]
    for row in main.itertuples():
        lines.append(f"{SSP_LABELS[row.Scenario]} relative to {SSP_LABELS[row.Baseline]}: modeled cumulative inpatient cost savings of CNY {row.Cumulative_savings_billion_CNY:.3f} billion at 2020 prices, 2030–2100.")
    (OUT / "Methods_and_table_figure_descriptions.md").write_text("\n\n".join(lines), encoding="utf-8")
    (OUT / "accounting_contract.json").write_text(json.dumps({
        "input_directory": str(DATA_DIR), "diseases": DISEASE_ORDER,
        "data_year": 2020, "price_year": 2020, "unit_costs": UNIT_COSTS,
        "source": COST_SOURCE, "source_url": SOURCE_URL, "money_divisor": BILLION_RMB,
        "main_integration": "trapezoidal 2030–2100", "duration_years": 70,
        "inclusive_annual_sum_count": 71, "primary_discount_rate": 0,
        "unit_cost_scope": "National public hospitals, not county hospitals",
        "interpretation": "Scenario-associated modeled hospitalization/cost changes",
        "uncertainty": "No full propagated statistical CI",
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_cumulative_savings_supplement(delta):
    d = delta.loc[(delta.Baseline == "ssp370") & delta.Scenario.isin(["ssp126", "ssp245"]) & (delta.Discount_rate == 0)].copy()
    if d.duplicated(["Year", "Disease", "Scenario"]).any():
        raise ValueError("Duplicate cumulative accounting rows")
    expected = {(y, dis, sc) for y in range(START, END+1) for dis in DISEASE_ORDER for sc in ["ssp126", "ssp245"]}
    if set(d[["Year", "Disease", "Scenario"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete seven-disease annual accounting coverage")
    if not np.isfinite(d.Cumulative_benefit_billion_CNY).all():
        raise ValueError("Nonfinite cumulative savings")
    
    d['Cumulative_benefit_billion_USD'] = d['Cumulative_benefit_billion_CNY'] / 6.9232
    totals = d.groupby(["Year", "Scenario"]).Cumulative_benefit_billion_USD.sum().unstack().sort_index()
    totals["Extra_ssp126_vs_ssp245_billion_USD"] = totals.ssp126 - totals.ssp245
    totals["Price_year"] = PRICE_YEAR
    totals["Baseline"] = "ssp370"
    csv(totals.reset_index(), "Source_12_cumulative_savings_extra_gain.csv")
    endpoint = d.loc[d.Year == END, ["Year", "Disease", "Baseline", "Scenario", "Cumulative_benefit_billion_USD"]].copy()
    endpoint["Price_year"] = PRICE_YEAR
    csv(endpoint, "Source_13_cumulative_savings_by_disease_2100.csv")
    values = endpoint.pivot(index="Disease", columns="Scenario", values="Cumulative_benefit_billion_USD").reindex(DISEASE_ORDER)
    for sc in ["ssp126", "ssp245"]:
        assert np.isclose(values[sc].sum(), totals.loc[END, sc])
    fig, axes = canvas(height=3.1)
    ax = axes[0]
    years = totals.index.to_numpy()
    for sc in ["ssp126", "ssp245"]:
        ax.plot(years, totals[sc], color=SSP_COLORS[sc], lw=1.6, label=SSP_LABELS[sc])
        decade = totals.loc[totals.index % 10 == 0]
        ax.scatter(decade.index, decade[sc], s=12, color=SSP_COLORS[sc], edgecolors="white", linewidths=.35, zorder=4)
    ax.fill_between(years, totals.ssp126.to_numpy(), totals.ssp245.to_numpy(), color="#DCE1E7", alpha=.75, label="Additional SSP1 savings", zorder=0)
    ax.set(xlim=(START, END), xlabel="Year", ylabel="Cumulative modeled cost savings\n(billion USD, 2020 prices)")
    ax.set_xticks([2030, 2050, 2070, 2090, 2100])
    #zero_grid(ax)
    ax.legend()
    savefig(fig, "12_cumulative_savings_extra_gain_vs_ssp370", axes)
    fig, axes = canvas(height=3.1)
    ax = axes[0]
    y = np.arange(len(DISEASE_ORDER))
    for shift, sc in zip([-.18, .18], ["ssp126", "ssp245"]):
        ax.barh(y+shift, values[sc], height=.31, color=SSP_COLORS[sc], label=SSP_LABELS[sc], edgecolor="none")
    ax.set_yticks(y, DISEASE_ORDER)
    ax.set_xlabel("Cumulative modeled cost savings by 2100\n(billion USD, 2020 prices)")
    ax.margins(x=.06, y=.06)
    #zero_grid(ax, horizontal=False)
    ax.legend()
    savefig(fig, "13_cumulative_savings_by_disease_2100", axes)
    (OUT/"Supplementary_cumulative_savings_captions.md").write_text(
        "**Cumulative scenario-associated inpatient cost savings relative to SSP3-7.0.** Cumulative modeled inpatient cost savings under SSP1-2.6 and SSP2-4.5 integrated over 2030-2100. Shading indicates the additional savings under SSP1-2.6 relative to SSP2-4.5, not an uncertainty interval. Markers denote decadal model-output years. Annual cost differences are linearly interpolated between decadal outputs and integrated using the trapezoidal rule; cumulative savings start at zero in 2030.\n\n"
        "**Disease-specific cumulative scenario-associated inpatient cost savings by 2100.** Cumulative modeled inpatient cost savings under SSP1-2.6 and SSP2-4.5 relative to SSP3-7.0 for the seven disease categories, integrated over 2030-2100. Disease totals reconcile with the trajectory endpoints.\n\n"
        "Both figures use undiscounted national disease-category inpatient costs, in billion CNY at 2020 prices. Signed differences and SSP-specific population changes are retained. These are modeled scenario comparisons, not causal estimates of actual budget savings. Source: Table S15.\n", encoding="utf-8")
    print("2100 cumulative totals (billion CNY):")
    print(totals.loc[END].to_string())


def update_figure_manifest(preserve=False):
    path = OUT/"Figure_manifest.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if preserve and path.exists() else []
    path.write_text(json.dumps(list(dict.fromkeys(previous+FIGURE_LOG)), indent=2), encoding="utf-8")


def self_check():
    assert np.isclose(cumulative_trapezoid([1., 1.], [2030, 2100])[-1], 70)
    assert np.isclose(cumulative_trapezoid([2., 4.], [2030, 2040])[-1], 30)
    assert unit_cost("Respiratory") == 6804.46
    assert unit_cost("Congenital") == 14639.82
    assert set(UNIT_COSTS) == set(DISEASE_ORDER)

def main(argv=None):
    global DATA_DIR, OUT, FIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--no-maps", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--refresh-maps-snapshot", action="store_true", help="Refresh only maps and the 2050 snapshot")
    parser.add_argument("--supplementary-only", action="store_true", help="Draw two added figures from existing Table S15")
    if argv is None and "ipykernel" in sys.modules:
        argv = []  # Ignore Jupyter kernel's -f connection-file argument.
    args = parser.parse_args(argv)
    DATA_DIR, OUT = args.data_dir.resolve(), args.out.resolve()
    FIG = OUT / "figures"
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    self_check(); set_style()
    if args.supplementary_only:
        plot_cumulative_savings_supplement(pd.read_csv(OUT/"Table_S15_annual_and_cumulative_savings.csv"))
        update_figure_manifest(preserve=True)
        print(f"Completed in {time.perf_counter()-started:.1f} seconds. Output: {FIG}")
        return
    print("Input:", DATA_DIR)
    print("Cost schedule:", COST_SOURCE)
    panel = validate_and_load()
    national = national_panel(panel)
    annual = annualise(national)
    delta = benefits(annual)
    summary = write_tables(national, annual, delta)
    write_notes(summary)
    if args.refresh_maps_snapshot:
        plot_snapshot(national)
        plot_maps(panel)
    elif not args.tables_only:
        stages = [
            ("SSP trajectories and cumulative economics", lambda: plot_trajectories(national, annual, delta)),
            ("Added cumulative savings figures", lambda: plot_cumulative_savings_supplement(delta)),
            ("Absolute and relative contributions", lambda: plot_compositions(national, delta)),
            ("Scenario differences", lambda: plot_delta_heatmap(delta)),
            ("Integrated savings per CEI", lambda: plot_integrated_metric(delta)),
            ("Income-related and spatial inequality", lambda: plot_inequality(panel, load_macro(panel))),
        ]
        if not args.no_maps:
            stages.append(("County maps", lambda: plot_maps(panel)))
        for title, function in stages:
            step = time.perf_counter(); print(">>>", title)
            function()
            print(f"    stage {time.perf_counter()-step:.1f}s; elapsed {time.perf_counter()-started:.1f}s")
    (OUT / "README_2020_CNY.txt").write_text("Current merged analysis: corrected code_6.3_results, seven disease categories, national 2020 public-hospital costs. See Economic_tables_2020_CNY.xlsx, Methods_and_table_figure_descriptions.md and Figure_manifest.json. PNG/PDF/SVG exports are in figures/. County polygon fills are rasterized at 400 dpi in PDF/SVG; labels remain editable. Maps use an asinh color scale (all values retained; median absolute change is the linear width) with grey for unavailable counties. The original health-risk model is not refitted. Pre-existing files outside this run manifest are not current outputs.", encoding="utf-8")
    update_figure_manifest(preserve=args.refresh_maps_snapshot or args.tables_only)
    print(summary[(summary.Cost_schedule == "National_category_costs_2020") &
                  (summary.Discount_rate == 0)].to_string(index=False))
    print(f"Completed in {time.perf_counter()-started:.1f} seconds. Output: {OUT}")

if __name__ == "__main__":
    main()
#%%
