"""Standalone national CEI and lag-alignment analysis.

Combines the original CEI standardization and alignment engine with the
update-specific country selection and figures. Results use the update output
directory, code_12_results_update/. The original scripts remain unchanged.
"""
#%%
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "code_12_results"
FIG_DIR = OUT_DIR / "figures"

EMI_PATH = BASE_DIR / "code_11_results" / "1_global_emission_data.csv"
WEIGHT_PATH = BASE_DIR / "code_11_results" / "2_disease_pollutant_sector_weights.csv"
PAIR_PATH = BASE_DIR / "code_3_results" / "3_Final_Top50pct_Categorized_Targets.csv"
# 可选：国名→ISO3
GDP_PATH = (
    BASE_DIR.parent
    / "global_national_cei"
    / "data_raw"
    / "IHME_GDP_1960_2050_Y2021M09D22.CSV"
)

REFERENCE_NAME = "China"
WINDOW = 10
EPI_START = 2017
MIN_CORR = 0.6
# Z-score 参照期：与中国栅格逻辑类似，用全样本历史分布；可改为固定年份区间
ZSCORE_YEAR_MIN = 2000
ZSCORE_YEAR_MAX = 2021

EXCLUDE_DISEASES = {"Cancer", "Neural"}

# 高收入对照（按国名，匹配 GBD_NAME）
CONTROL_NAMES = {
    "United States of America",
    "United States",
    "United Kingdom",
    "Japan",
    "Germany",
    "France",
    "Australia",
    "Canada",
    "Republic of Korea",
    "South Korea",
    "Italy",
    "Spain",
    "Netherlands",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Switzerland",
    "Austria",
    "Belgium",
    "Ireland",
    "New Zealand",
    "Singapore",
    "Israel",
    "Portugal",
    "Greece",
    "Taiwan (Province of China)",
}

# 全球南方优先国家（GBD 常用名）
PRIORITY_SOUTH = [
    "India",
    "Indonesia",
    "Vietnam",
    "Bangladesh",
    "Pakistan",
    "Philippines",
    "Thailand",
    "Myanmar",
    "Cambodia",
    "Lao People's Democratic Republic",
    "Nepal",
    "Sri Lanka",
    "Malaysia",
    "Mongolia",
    "Nigeria",
    "Ethiopia",
    "Egypt",
    "South Africa",
    "Kenya",
    "Ghana",
    "United Republic of Tanzania",
    "Uganda",
    "Algeria",
    "Morocco",
    "Angola",
    "Mozambique",
    "Democratic Republic of the Congo",
    "Sudan",
    "Cameroon",
    "Côte d'Ivoire",
    "Cote d'Ivoire",
    "Senegal",
    "Zambia",
    "Zimbabwe",
    "Madagascar",
    "Brazil",
    "Mexico",
    "Argentina",
    "Colombia",
    "Peru",
    "Ecuador",
    "Bolivia (Plurinational State of)",
    "Venezuela (Bolivarian Republic of)",
    "Paraguay",
    "Guatemala",
    "Honduras",
    "Nicaragua",
    "Dominican Republic",
    "Cuba",
    "Turkey",
    "Iran (Islamic Republic of)",
    "Iraq",
    "Kazakhstan",
    "Uzbekistan",
    "Azerbaijan",
    "Ukraine",
    "Russian Federation",
]

CORE_DISEASES = ["Circulation", "Blood", "Digestive", "Skin", "Congenital"]
DISEASE_ORDER = [
    "Circulation",
    "Respiration",
    "Endocrine",
    "Digestive",
    "Blood",
    "Skin",
    "Congenital",
]

NAME_ALIASES = {
    "United States of America": "United States",
    "Czech Republic": "Czechia",
    "Swaziland": "Eswatini",
    "Turkey": "Turkey",
    "Viet Nam": "Vietnam",
    "Vietnam": "Vietnam",
}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def parse_exposure(token: str) -> tuple[str, str]:
    e = str(token).strip()
    if e.startswith("X_"):
        e = e[2:]
    return e.rsplit("_", 1)


def load_weights() -> pd.DataFrame:
    if WEIGHT_PATH.exists():
        df_w = pd.read_csv(WEIGHT_PATH)
        df_w = df_w[~df_w["Disease"].isin(EXCLUDE_DISEASES)].copy()
        return df_w

    # 回退：从 Priority 表重建
    raw = pd.read_csv(PAIR_PATH)
    disease_col = "Disease_Name" if "Disease_Name" in raw.columns else "Label"
    rows = []
    for disease, g in raw.groupby(disease_col):
        if disease in EXCLUDE_DISEASES:
            continue
        for _, row in g.iterrows():
            expos = re.split(r",\s*(?=X_)", str(row["Exposure"]).strip())
            for e in expos:
                e = e.strip()
                if not e:
                    continue
                pol, sec = parse_exposure(e)
                rows.append(
                    {
                        "Disease": disease,
                        "Exposure": e if e.startswith("X_") else f"X_{e}",
                        "pollutant": pol,
                        "sector": sec,
                        "weight": float(row.get("Priority_Index", 1.0)),
                    }
                )
    df_w = pd.DataFrame(rows)
    return (
        df_w.groupby(["Disease", "pollutant", "sector", "Exposure"], as_index=False)[
            "weight"
        ].max()
    )


def _norm_name(s: str) -> str:
    s = str(s).strip()
    return NAME_ALIASES.get(s, s)


# ---------------------------------------------------------------------------
# ① 标准化国家级 wCEI
# ---------------------------------------------------------------------------
def build_standardized_national_wcei() -> pd.DataFrame:
    """
    Z_{i,c,t} = (E_{i,c,t} - μ_i) / σ_i
    μ,σ 来自全球所有国家×年（ZSCORE_YEAR_MIN–MAX）上该 pollutant|sector 的分布
    wCEI_{d,c,t} = Σ_i Z_{i,c,t} * w_{i,d}
    """
    print(">>> [1/3] 构建标准化国家级 wCEI ...")
    df_w = load_weights()
    df_emi = pd.read_csv(EMI_PATH)
    df_emi["Year"] = df_emi["Year"].astype(int)
    df_emi["pollutant"] = df_emi["pollutant"].astype(str).str.strip()
    df_emi["sector"] = df_emi["sector"].astype(str).str.strip()
    df_emi["GBD_NAME"] = df_emi["GBD_NAME"].map(_norm_name)
    df_emi["emission_g"] = pd.to_numeric(df_emi["emission_g"], errors="coerce").fillna(0.0)

    emi = (
        df_emi.groupby(["GBD_NAME", "Year", "pollutant", "sector"], as_index=False)[
            "emission_g"
        ]
        .sum()
    )

    # 参照分布：全球国家×年
    ref = emi[
        (emi["Year"] >= ZSCORE_YEAR_MIN) & (emi["Year"] <= ZSCORE_YEAR_MAX)
    ]
    stats = (
        ref.groupby(["pollutant", "sector"])["emission_g"]
        .agg(mu="mean", sigma="std")
        .reset_index()
    )
    # ddof=1 时若仅1个点 std=NaN；用总体标准差兜底
    stats["sigma"] = stats["sigma"].replace(0, np.nan)
    # 重新算 population std
    pop_std = (
        ref.groupby(["pollutant", "sector"])["emission_g"]
        .apply(lambda s: float(np.std(s.to_numpy(), ddof=0)))
        .reset_index(name="sigma0")
    )
    stats = stats.merge(pop_std, on=["pollutant", "sector"], how="left")
    stats["sigma"] = stats["sigma"].fillna(stats["sigma0"]).fillna(1.0)
    stats.loc[stats["sigma"] < 1e-12, "sigma"] = 1.0
    stats.to_csv(OUT_DIR / "1_zscore_params_global.csv", index=False)

    emi = emi.merge(stats[["pollutant", "sector", "mu", "sigma"]], on=["pollutant", "sector"], how="left")
    emi["Z"] = (emi["emission_g"] - emi["mu"]) / emi["sigma"]
    emi["Z"] = emi["Z"].fillna(0.0)

    # 合并权重
    linked = emi.merge(
        df_w[["Disease", "pollutant", "sector", "weight", "Exposure"]],
        on=["pollutant", "sector"],
        how="inner",
    )
    linked["contribution"] = linked["Z"] * linked["weight"]

    cei = (
        linked.groupby(["GBD_NAME", "Year", "Disease"], as_index=False)
        .agg(
            wCEI=("contribution", "sum"),
            n_targets_used=("Exposure", "nunique"),
        )
    )
    n_tot = df_w.groupby("Disease")["Exposure"].nunique().rename("n_targets_total")
    cei = cei.merge(n_tot, on="Disease", how="left")
    cei["coverage"] = cei["n_targets_used"] / cei["n_targets_total"]

    # 同时保留非标准化 CEI（质量加权）便于对照
    linked_raw = emi.merge(
        df_w[["Disease", "pollutant", "sector", "weight", "Exposure"]],
        on=["pollutant", "sector"],
        how="inner",
    )
    linked_raw["contribution_raw"] = linked_raw["emission_g"] * linked_raw["weight"]
    cei_raw = (
        linked_raw.groupby(["GBD_NAME", "Year", "Disease"], as_index=False)[
            "contribution_raw"
        ]
        .sum()
        .rename(columns={"contribution_raw": "CEI_raw"})
    )
    cei = cei.merge(cei_raw, on=["GBD_NAME", "Year", "Disease"], how="left")

    cei.to_csv(OUT_DIR / "2_national_wCEI_standardized.csv", index=False)
    linked[
        ["GBD_NAME", "Year", "Disease", "pollutant", "sector", "Exposure", "weight", "emission_g", "Z", "contribution"]
    ].to_csv(OUT_DIR / "2_national_wCEI_contributions.csv", index=False)

    wide = cei.pivot_table(
        index=["GBD_NAME", "Year"], columns="Disease", values="wCEI", aggfunc="sum"
    ).reset_index()
    wide.columns = [
        c if c in ("GBD_NAME", "Year") else f"wCEI_{c}" for c in wide.columns
    ]
    wide.to_csv(OUT_DIR / "2_national_wCEI_wide.csv", index=False)

    print(
        f"    国家={cei['GBD_NAME'].nunique()}, "
        f"年={cei['Year'].min()}-{cei['Year'].max()}, "
        f"病种={sorted(cei['Disease'].unique().tolist())}"
    )
    print("    说明: Z 的 μ/σ 来自全球国家×年全样本，不是中国栅格 2017-2023。")
    return cei


# ---------------------------------------------------------------------------
# ② 全球南方筛选 + 时滞对齐
# ---------------------------------------------------------------------------
def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    return (x - np.nanmean(x)) / sd


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), 1e18)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(np.sqrt(dp[n, m]))




def build_china_windows(series: pd.DataFrame, disease: str, window: int = WINDOW) -> dict:
    c = series[series["Disease"] == disease].sort_values("Year")
    years = c["Year"].to_numpy(dtype=int)
    y = c["wCEI"].to_numpy(dtype=float)
    if len(y) < window:
        return {"years": years, "starts": np.array([], dtype=int), "Z": np.empty((0, window))}
    starts = np.arange(0, len(y) - window + 1)
    Z = np.vstack([_zscore(y[i : i + window]) for i in starts])
    keep = ~np.all(np.isclose(Z, 0.0), axis=1)
    return {"years": years, "starts": starts[keep], "Z": Z[keep], "Yraw": y}


def align_one(
    win: dict,
    other: pd.DataFrame,
    disease: str,
    country: str,
    window: int = WINDOW,
) -> dict | None:
    o = other[(other["Disease"] == disease) & (other["GBD_NAME"] == country)].sort_values(
        "Year"
    )
    if len(o) < window or len(win["starts"]) == 0:
        return None
    o_win = o.tail(window)
    years_o = o_win["Year"].to_numpy(dtype=int)
    z_o = _zscore(o_win["wCEI"].to_numpy(dtype=float))
    if np.allclose(z_o, 0):
        return None

    # r = Z_win @ z_o / W
    r_all = (win["Z"] @ z_o) / window
    best_k = int(np.nanargmax(r_all))
    r = float(r_all[best_k])
    start_idx = int(win["starts"][best_k])
    years = win["years"]
    c0 = int(years[start_idx])
    c1 = int(years[start_idx + window - 1])
    dtw = _dtw(win["Z"][best_k], z_o)
    _, p = pearsonr(win["Z"][best_k], z_o)

    s0, s1 = int(years_o[0]), int(years_o[-1])
    lag = s0 - c0
    policy = EPI_START - c1
    implied = s1 + policy
    quality = "strong" if r >= 0.8 else "moderate" if r >= MIN_CORR else "weak"
    # 政策窗口 ≤ 0：相对中国 2017 流行病学挂钩时点已错过，显示为 "-"，表示已不能干预
    closed = int(policy) <= 0
    policy_label = "-" if closed else str(int(policy))
    status = "closed_no_intervention" if closed else ("urgent" if int(policy) <= 3 else "open")

    return {
        "Disease": disease,
        "Country": country,
        "sentinel_start": s0,
        "sentinel_end": s1,
        "china_match_start": c0,
        "china_match_end": c1,
        "time_lag_years": int(lag),
        "r_shape": r,
        "p_shape": float(p),
        "dtw": float(dtw),
        "match_quality": quality,
        "implied_epi_entry_year": int(implied),
        "policy_window_years_from_now": int(policy),
        "policy_window_label": policy_label,
        "intervention_status": status,
        "interpretation": (
            f"{country} {disease} ({s0}-{s1}) matches China ({c0}-{c1}); "
            f"lag≈{lag}y; policy window={policy_label}"
            + (
                " (already past epi-linked stage — no longer intervenable)."
                if closed
                else f"y from {s1} (epi-entry≈{int(implied)})."
            )
        ),
    }




# ---------------------------------------------------------------------------
# ③ 可视化
# ---------------------------------------------------------------------------
def plot_early_warning_trajectories(cei: pd.DataFrame) -> None:
    """重点前哨国轨迹 + 中国标尺（标准化 wCEI）。"""
    focus = [
        "United States",
        "Germany",
        "China",
        "India",
        "Indonesia",
        "Vietnam",
        "Bangladesh",
        "Nigeria",
        "Brazil",
    ]
    focus = [c for c in focus if c in set(cei["GBD_NAME"])]
    colors = {
        "United States": "#636363",
        "Germany": "#969696",
        "China": "#b2182b",
        "India": "#e66101",
        "Indonesia": "#5e3c99",
        "Vietnam": "#1b7837",
        "Bangladesh": "#2166ac",
        "Nigeria": "#8c510a",
        "Brazil": "#35978f",
    }
    diseases = [d for d in ["Circulation", "Blood", "Digestive", "Skin"] if d in set(cei["Disease"])]

    for disease in diseases:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        d = cei[cei["Disease"] == disease].copy()
        # A: 原始标准化 wCEI
        ax = axes[0]
        chn = d[d["GBD_NAME"] == "China"].sort_values("Year")
        epi_level = float(chn.loc[chn["Year"] >= EPI_START, "wCEI"].mean()) if len(chn) else np.nan
        for name in focus:
            sub = d[d["GBD_NAME"] == name].sort_values("Year")
            sub = sub[sub["Year"] >= 2000]
            if sub.empty:
                continue
            lw = 3.2 if name == "China" else 1.4 if name in ("United States", "Germany") else 2.2
            ls = (0, (3, 2)) if name in ("United States", "Germany") else "-"
            if name == "China":
                pre, epi = sub[sub["Year"] < EPI_START], sub[sub["Year"] >= EPI_START]
                if len(pre):
                    ax.plot(pre["Year"], pre["wCEI"], color=colors[name], lw=2.0)
                if len(epi):
                    ax.plot(epi["Year"], epi["wCEI"], color=colors[name], lw=lw, label=name)
                    ax.scatter([epi["Year"].iloc[-1]], [epi["wCEI"].iloc[-1]], s=35, color=colors[name], zorder=5)
            else:
                ax.plot(sub["Year"], sub["wCEI"], color=colors.get(name, "#333"), lw=lw, ls=ls, label=name)
        if np.isfinite(epi_level):
            ax.axhline(epi_level, color="#b2182b", ls=(0, (4, 3)), lw=1.3)
            ax.text(2000.3, epi_level, "China epi-linked level", color="#b2182b", fontsize=7.5, fontweight="bold", va="bottom")
        ax.set_title(f"A  Standardized wCEI — {disease}")
        ax.set_xlabel("Year")
        ax.set_ylabel("wCEI (Z-weighted)")
        ax.grid(True, axis="y", ls=":", alpha=0.45)

        # B: 2000=1 指数（用平移后正值：wCEI - min_country + eps，再相对2000）
        ax = axes[1]
        for name in focus:
            sub = d[d["GBD_NAME"] == name].sort_values("Year")
            sub = sub[sub["Year"] >= 2000].copy()
            if sub.empty or 2000 not in set(sub["Year"]):
                continue
            # 指数用“相对 2000 的增量轨迹”：1 + (wCEI - wCEI_2000) / (|wCEI_2000|+1)
            base = float(sub.loc[sub["Year"] == 2000, "wCEI"].iloc[0])
            sub["idx"] = 1.0 + (sub["wCEI"] - base) / (abs(base) + 1.0)
            lw = 3.2 if name == "China" else 1.4 if name in ("United States", "Germany") else 2.2
            ls = (0, (3, 2)) if name in ("United States", "Germany") else "-"
            ax.plot(sub["Year"], sub["idx"], color=colors.get(name, "#333"), lw=lw, ls=ls, label=name)
        ax.axhline(1.0, color="#aaa", ls=":", lw=0.8)
        ax.set_title(f"B  Momentum index (2000 baseline) — {disease}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Relative momentum")
        ax.grid(True, axis="y", ls=":", alpha=0.45)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=8)
        fig.suptitle(
            f"Early-warning trajectories (standardized national wCEI): {disease}",
            fontsize=12,
            fontweight="bold",
            y=1.03,
        )
        fig.savefig(FIG_DIR / f"Fig_early_warning_{disease}.png")
        fig.savefig(FIG_DIR / f"Fig_early_warning_{disease}.pdf")
        plt.close(fig)


def plot_lag_results(res: pd.DataFrame) -> None:
    d = res[(res["group"] == "global_south") & (res["Disease"] == "Circulation")]
    d = d[d["match_quality"] != "weak"].sort_values("time_lag_years")
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, max(5, 0.2 * len(d) + 1.5)))
    ax.barh(d["Country"], d["time_lag_years"], color="#fc8d59", height=0.75)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Time lag vs China (years)")
    ax.set_title("Global South trajectory lag — Circulation (standardized wCEI)")
    fig.savefig(FIG_DIR / "Fig_timelag_bar_Circulation.png")
    fig.savefig(FIG_DIR / "Fig_timelag_bar_Circulation.pdf")
    plt.close(fig)

    # 政策窗口最短 Top20（≤0 记为已不能干预，条形画在 0，标签标 "-"）
    top = d.sort_values(["policy_window_years_from_now", "r_shape"], ascending=[True, False]).head(20).copy()
    closed = top["policy_window_years_from_now"] <= 0
    bar_vals = top["policy_window_years_from_now"].clip(lower=0)
    colors = np.where(closed, "#7f7f7f", "#b2182b")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["Country"], bar_vals, color=colors, height=0.75)
    for y, (_, row) in enumerate(top.iterrows()):
        lab = row.get("policy_window_label", str(int(row["policy_window_years_from_now"])))
        ax.text(
            max(float(bar_vals.iloc[y]), 0) + 0.15,
            y,
            lab if lab != "-" else "- (closed)",
            va="center",
            fontsize=7.5,
            color="#444",
        )
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Policy window (years; '-' = already past, no longer intervenable)")
    ax.set_title("Most urgent policy windows — Circulation")
    fig.savefig(FIG_DIR / "Fig_policy_window_top20_Circulation.png")
    fig.savefig(FIG_DIR / "Fig_policy_window_top20_Circulation.pdf")
    plt.close(fig)

    # 核心病种热图：展示政策窗口；≤0 或弱匹配显示 "-"（已不能干预 / 不可靠）
    focus_countries = (
        d.sort_values("r_shape", ascending=False)["Country"].head(25).tolist()
    )
    for must in ["India", "Indonesia", "Vietnam", "Bangladesh", "Nigeria", "Brazil", "Mexico"]:
        if must in set(res["Country"]) and must not in focus_countries:
            focus_countries.append(must)
    heat = res[
        (res["group"] == "global_south")
        & (res["Country"].isin(focus_countries))
        & (res["Disease"].isin(CORE_DISEASES))
    ].copy()
    heat["plot_window"] = heat["policy_window_years_from_now"].astype(float)
    heat.loc[heat["match_quality"] == "weak", "plot_window"] = np.nan
    heat.loc[heat["policy_window_years_from_now"] <= 0, "plot_window"] = np.nan
    mat = heat.pivot(index="Country", columns="Disease", values="plot_window")
    mat = mat.reindex(columns=[c for c in CORE_DISEASES if c in mat.columns])
    # 排序用原始政策窗口（含负数），紧迫的在上
    circ_order = heat[heat["Disease"] == "Circulation"].set_index("Country")[
        "policy_window_years_from_now"
    ]
    mat = mat.reindex(circ_order.sort_values(na_position="last").index.intersection(mat.index))

    fig, ax = plt.subplots(figsize=(7.5, max(5.5, 0.28 * len(mat) + 1)))
    im = ax.imshow(mat.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=0, vmax=20)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=25, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iloc[i, j]
            ax.text(j, i, f"{int(v)}" if pd.notna(v) else "-", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Policy window (years)")
    ax.set_title("Policy window heatmap (core diseases)\n'-' = already past / no longer intervenable")
    fig.savefig(FIG_DIR / "Fig_timelag_heatmap_core.png")
    fig.savefig(FIG_DIR / "Fig_timelag_heatmap_core.pdf")
    plt.close(fig)

    # 另存：时滞年数热图（弱匹配为 "-"）
    heat2 = res[
        (res["group"] == "global_south")
        & (res["Country"].isin(focus_countries))
        & (res["Disease"].isin(CORE_DISEASES))
    ].copy()
    heat2.loc[heat2["match_quality"] == "weak", "time_lag_years"] = np.nan
    mat2 = heat2.pivot(index="Country", columns="Disease", values="time_lag_years")
    mat2 = mat2.reindex(columns=[c for c in CORE_DISEASES if c in mat2.columns])
    if "Circulation" in mat2.columns:
        mat2 = mat2.reindex(mat2["Circulation"].sort_values(na_position="last").index)
    fig, ax = plt.subplots(figsize=(7.5, max(5.5, 0.28 * len(mat2) + 1)))
    im = ax.imshow(mat2.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=0, vmax=25)
    ax.set_xticks(range(mat2.shape[1]))
    ax.set_xticklabels(mat2.columns, rotation=25, ha="right")
    ax.set_yticks(range(mat2.shape[0]))
    ax.set_yticklabels(mat2.index, fontsize=7)
    for i in range(mat2.shape[0]):
        for j in range(mat2.shape[1]):
            v = mat2.iloc[i, j]
            ax.text(j, i, f"{int(v)}" if pd.notna(v) else "-", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Lag (years)")
    ax.set_title("Time-lag heatmap (core diseases)\n'-' = weak match (unreliable)")
    fig.savefig(FIG_DIR / "Fig_timelag_years_heatmap_core.png")
    fig.savefig(FIG_DIR / "Fig_timelag_years_heatmap_core.pdf")
    plt.close(fig)


def plot_alignment_overlays(cei: pd.DataFrame, res: pd.DataFrame, n: int = 8) -> None:
    d = res[(res["group"] == "global_south") & (res["match_quality"] == "strong")].copy()
    d = d[d["Disease"].isin(CORE_DISEASES)].sort_values("r_shape", ascending=False)
    picked = d.groupby("Country", as_index=False).head(1).head(n)
    if picked.empty:
        return
    n = len(picked)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, row) in zip(axes, picked.iterrows()):
        c = cei[
            (cei["GBD_NAME"] == "China")
            & (cei["Disease"] == row["Disease"])
            & (cei["Year"].between(row["china_match_start"], row["china_match_end"]))
        ].sort_values("Year")
        o = cei[
            (cei["GBD_NAME"] == row["Country"])
            & (cei["Disease"] == row["Disease"])
            & (cei["Year"].between(row["sentinel_start"], row["sentinel_end"]))
        ].sort_values("Year")
        zc = _zscore(c["wCEI"].to_numpy())
        zo = _zscore(o["wCEI"].to_numpy())
        x = np.arange(len(zc))
        ax.plot(x, zc, color="#b2182b", lw=2.3, label=f"China {int(row['china_match_start'])}-{int(row['china_match_end'])}")
        ax.plot(x, zo, color="#e66101", lw=2.0, label=f"{row['Country']} {int(row['sentinel_start'])}-{int(row['sentinel_end'])}")
        ax.axhline(0, color="#ccc", lw=0.7)
        pw = row.get("policy_window_label", str(int(row["policy_window_years_from_now"])))
        pw_txt = "window=- (closed)" if pw == "-" else f"window={pw}y"
        ax.set_title(
            f"{row['Country']} | {row['Disease']} | lag={int(row['time_lag_years'])}y | r={row['r_shape']:.2f}",
            fontsize=12,
        )
        ax.legend(fontsize=12)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
    for j in range(n, len(axes)):
        axes[j].set_axis_off()
    fig.suptitle("Best trajectory alignments (standardized wCEI)", fontweight="bold", y=1.01)
    fig.savefig(FIG_DIR / "Fig_alignment_overlays.png")
    fig.savefig(FIG_DIR / "Fig_alignment_overlays.pdf")
    plt.close(fig)




# Update-specific selection and figures. The original computation functions
# above are embedded here, so this file has no dependency on either old script.
import geopandas as gpd
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

OUT = OUT_DIR
FIG = FIG_DIR
CLASS_XLSX = BASE_DIR / "data_raw" / "CLASS_2026_07_15.xlsx"
WORLD_SHP = BASE_DIR / "data_raw" / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"
GDP_PATH = BASE_DIR.parent / "global_national_cei" / "data_raw" / "IHME_GDP_1960_2050_Y2021M09D22.CSV"

SLOWER_INCOME_GROUPS = {"Low income", "Lower middle income"}

# ggsci NPG（与 code_10 一致）
DISEASE_COLORS = {
    "Circulation": "#E64B35",
    "Digestive": "#4DBBD5",
    "Respiration": "#00A087",
    "Blood": "#3C5488",
    "Endocrine": "#F39B7F",
    "Skin": "#8491B4",
    "Congenital": "#91D1C2",
}
DISEASE_ORDER = [
    "Circulation",
    "Respiration",
    "Endocrine",
    "Digestive",
    "Blood",
    "Skin",
    "Congenital",
]
REGION_ORDER = [
    "Sub-Saharan Africa",
    "South Asia",
    "Middle East, North Africa, Afghanistan & Pakistan",
    "Latin America & Caribbean",
    "Europe & Central Asia",
    "East Asia & Pacific",
]
REGION_SHORT = {
    "Sub-Saharan Africa": "Sub-Saharan\nAfrica",
    "South Asia": "South\nAsia",
    "Middle East, North Africa, Afghanistan & Pakistan": "MENA &\nAfg/Pak",
    "Latin America & Caribbean": "Lat. America\n& Caribbean",
    "Europe & Central Asia": "Europe &\nCentral Asia",
    "East Asia & Pacific": "East Asia\n& Pacific",
}

# WB Economy / 特殊写法 → GBD_NAME
WB_TO_GBD = {
    "Bolivia": "Bolivia (Plurinational State of)",
    "Congo, Dem. Rep.": "Democratic Republic of the Congo",
    "Congo, Rep.": "Congo",
    "Côte d’Ivoire": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Egypt, Arab Rep.": "Egypt",
    "Gambia, The": "Gambia",
    "Korea, Dem. People's Rep.": "Democratic People's Republic of Korea",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Lao PDR": "Lao People's Democratic Republic",
    "Sao Tome and Principe": "Sao Tome and Principe",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Somalia, Fed. Rep.": "Somalia",
    "Tanzania": "United Republic of Tanzania",
    "Venezuela, RB": "Venezuela (Bolivarian Republic of)",
    "Yemen, Rep.": "Yemen",
    "West Bank and Gaza": "Palestine",
    "Viet Nam": "Vietnam",
    "Vietnam": "Vietnam",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["times new roman", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}")


def build_iso_to_gbd() -> dict[str, str]:
    """ISO3 → GBD_NAME（优先 IHME Country 名，且须出现在排放表中）。"""
    emi_names = set(
        pd.read_csv(BASE_DIR / "code_11_results" / "1_global_emission_data.csv", usecols=["GBD_NAME"])[
            "GBD_NAME"
        ].unique()
    )
    iso_map: dict[str, str] = {}
    if GDP_PATH.exists():
        gdp = pd.read_csv(GDP_PATH, usecols=["location_name", "iso3", "level"])
        gdp = gdp[gdp["level"] == "Country"].drop_duplicates("iso3")
        for _, r in gdp.iterrows():
            if r["location_name"] in emi_names:
                iso_map[str(r["iso3"])] = r["location_name"]
    # 手工补丁
    manual = {
        "BOL": "Bolivia (Plurinational State of)",
        "COD": "Democratic Republic of the Congo",
        "COG": "Congo",
        "CIV": "Côte d'Ivoire",
        "EGY": "Egypt",
        "GMB": "Gambia",
        "PRK": "Democratic People's Republic of Korea",
        "KGZ": "Kyrgyzstan",
        "LAO": "Lao People's Democratic Republic",
        "STP": "Sao Tome and Principe",
        "SOM": "Somalia",
        "TZA": "United Republic of Tanzania",
        "VEN": "Venezuela (Bolivarian Republic of)",
        "YEM": "Yemen",
        "PSE": "Palestine",
        "VNM": "Vietnam",
    }
    for k, v in manual.items():
        if v in emi_names:
            iso_map[k] = v
    return iso_map


def select_slower_than_china(cei: pd.DataFrame) -> pd.DataFrame:
    """返回筛选表：Economy, Code, Income group, Region, GBD_NAME。"""
    if not CLASS_XLSX.exists():
        raise FileNotFoundError(CLASS_XLSX)
    cls = pd.read_excel(CLASS_XLSX, sheet_name="List of economies")
    china = cls[cls["Economy"].str.strip().eq("China")]
    china_ig = china["Income group"].iloc[0] if len(china) else "Upper middle income"

    slow = cls[cls["Income group"].isin(SLOWER_INCOME_GROUPS)].copy()
    iso2gbd = build_iso_to_gbd()
    emi_names = set(cei["GBD_NAME"].unique())

    rows = []
    for _, r in slow.iterrows():
        eco = str(r["Economy"]).strip()
        code = str(r["Code"]).strip()
        gbd = None
        if code in iso2gbd:
            gbd = iso2gbd[code]
        elif eco in WB_TO_GBD and WB_TO_GBD[eco] in emi_names:
            gbd = WB_TO_GBD[eco]
        elif eco in emi_names:
            gbd = eco
        rows.append(
            {
                "Economy": eco,
                "Code": code,
                "Region": r.get("Region"),
                "Income_group": r["Income group"],
                "Lending_category": r.get("Lending category"),
                "GBD_NAME": gbd,
                "matched": gbd is not None and gbd in emi_names,
            }
        )
    tab = pd.DataFrame(rows)
    tab["China_Income_group"] = china_ig
    return tab


def run_timelag_all(cei: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    """对筛选国家 × 全部病种做匹配（算法同原 code_12）。"""
    china = cei[cei["GBD_NAME"] == REFERENCE_NAME]
    if china.empty:
        raise ValueError("China not found in CEI panel")

    diseases = [d for d in DISEASE_ORDER if d in set(cei["Disease"])]
    wins = {d: build_china_windows(china, d, WINDOW) for d in diseases}

    rows = []
    for disease in diseases:
        win = wins[disease]
        for country in countries:
            one = align_one(win, cei, disease, country)
            if one:
                one["group"] = "global_south"
                rows.append(one)

    # 高收入对照（保留原分析对照思路）
    for ctrl in ["United States of America", "United States", "United Kingdom", "Japan", "Germany"]:
        if ctrl not in set(cei["GBD_NAME"]):
            continue
        for disease in CORE_DISEASES[:2]:
            one = align_one(wins[disease], cei, disease, ctrl)
            if one:
                one["group"] = "control"
                rows.append(one)

    return pd.DataFrame(rows)


def plot_world_lag_map(res: pd.DataFrame, meta: pd.DataFrame, disease: str = "Circulation") -> None:
    if not WORLD_SHP.exists():
        print("  [skip map] world shapefile missing")
        return

    world = gpd.read_file(WORLD_SHP)
    # 去掉南极，地图更干净
    if "CONTINENT" in world.columns:
        world = world[world["CONTINENT"] != "Antarctica"].copy()
    iso_col = "ISO_A3" if "ISO_A3" in world.columns else "ADM0_A3"

    d = res[(res["group"] == "global_south") & (res["Disease"] == disease)].copy()
    d.loc[d["match_quality"] == "weak", "time_lag_years"] = np.nan
    meta_cols = meta[["GBD_NAME", "Code", "Income_group"]].drop_duplicates("GBD_NAME")
    d = d.drop(columns=[c for c in ["Code", "Income_group", "Region", "GBD_NAME"] if c in d.columns], errors="ignore")
    d = d.merge(meta_cols, left_on="Country", right_on="GBD_NAME", how="left")

    gdf = world.merge(
        d[["Code", "time_lag_years", "r_shape", "Country", "Income_group"]].dropna(subset=["Code"]),
        left_on=iso_col,
        right_on="Code",
        how="left",
    )

    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    # 底图：全部国家 + 清晰国界线
    gdf.plot(ax=ax, color="#f2f2f2", edgecolor="#666666", linewidth=0.35)
    valid = gdf["time_lag_years"].notna()
    if valid.any():
        gdf[valid].plot(
            column="time_lag_years",
            ax=ax,
            cmap="YlOrRd",
            linewidth=0.35,
            edgecolor="#444444",
            legend=False,
            vmin=0,
            vmax=max(25, float(gdf.loc[valid, "time_lag_years"].quantile(0.95))),
        )
    china = gdf[gdf[iso_col] == "CHN"]
    if not china.empty:
        china.plot(ax=ax, facecolor="none", edgecolor="#08306b", linewidth=1.3)

    sm = plt.cm.ScalarMappable(
        cmap="YlOrRd",
        norm=mcolors.Normalize(
            vmin=0,
            vmax=max(25, float(gdf.loc[valid, "time_lag_years"].quantile(0.95)) if valid.any() else 25),
        ),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label(f"Time lag vs China (years) — {disease}", fontsize=9)

    ax.set_axis_off()
    ax.set_title(
        f"Trajectory lag behind China | {disease}\n"
        "Countries with income below China (Low + Lower-middle)",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    ax.text(
        0.01,
        0.02,
        "Grey = not in sample / weak match; dark outline = China; lines = national borders",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
    )
    savefig(fig, f"12_world_map_lag_{disease}")


def plot_world_lag_maps_multi(res: pd.DataFrame, meta: pd.DataFrame, diseases: list[str]) -> None:
    """7 病种全球滞后图（含国界线）。"""
    if not WORLD_SHP.exists():
        return
    world = gpd.read_file(WORLD_SHP)
    if "CONTINENT" in world.columns:
        world = world[world["CONTINENT"] != "Antarctica"].copy()
    iso_col = "ISO_A3" if "ISO_A3" in world.columns else "ADM0_A3"

    n = len(diseases)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 3.35 * nrows))
    axes = np.atleast_1d(axes).ravel()
    vmax = 25
    meta_cols = meta[["GBD_NAME", "Code"]].drop_duplicates("GBD_NAME")

    for ax, disease in zip(axes, diseases):
        d = res[(res["group"] == "global_south") & (res["Disease"] == disease)].copy()
        d.loc[d["match_quality"] == "weak", "time_lag_years"] = np.nan
        d = d.drop(columns=[c for c in ["Code", "GBD_NAME"] if c in d.columns], errors="ignore")
        d = d.merge(meta_cols, left_on="Country", right_on="GBD_NAME", how="left")
        gdf = world.merge(
            d[["Code", "time_lag_years"]].dropna(subset=["Code"]),
            left_on=iso_col,
            right_on="Code",
            how="left",
        )
        # 全部国家底图 + 国界线
        gdf.plot(ax=ax, color="#f2f2f2", edgecolor="#666666", linewidth=0.28)
        valid = gdf["time_lag_years"].notna()
        if valid.any():
            gdf[valid].plot(
                column="time_lag_years",
                ax=ax,
                cmap="YlOrRd",
                linewidth=0.28,
                edgecolor="#444444",
                legend=False,
                vmin=0,
                vmax=vmax,
            )
        china = gdf[gdf[iso_col] == "CHN"]
        if not china.empty:
            china.plot(ax=ax, facecolor="none", edgecolor="#08306b", linewidth=0.9)
        ax.set_title(disease, fontsize=10, fontweight="bold")
        ax.set_axis_off()

    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=mcolors.Normalize(0, vmax))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Lag (years)", fontsize=8)
    fig.suptitle(
        "Global lag maps by disease (slower-than-China sample; national borders shown)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(right=0.90, wspace=0.04, hspace=0.12, top=0.94, bottom=0.02)
    savefig(fig, "12_world_map_lag_multi_disease")


def plot_lag_dist_heatmap_combo(reliable: pd.DataFrame) -> None:
    """左：滞后分布直方图；右：病种×地区均值热图（红色）。"""
    diseases = [d for d in DISEASE_ORDER if d in set(reliable["Disease"])]
    if not diseases:
        return
    g = reliable.dropna(subset=["time_lag_years"]).copy()

    bins = np.arange(-0.5, 26.5, 1.0)
    counts_all = []
    ymax = 1
    for d in diseases:
        vals = g.loc[g["Disease"] == d, "time_lag_years"].to_numpy()
        counts, _ = np.histogram(vals, bins=bins)
        counts_all.append((d, vals, counts))
        ymax = max(ymax, int(counts.max()) if len(counts) else 0)

    regions = [r for r in REGION_ORDER if r in set(g["Region"].dropna())]
    # 补上未列在 REGION_ORDER 里的地区
    for r in sorted(set(g["Region"].dropna()) - set(regions)):
        regions.append(r)
    heat = g.groupby(["Disease", "Region"])["time_lag_years"].mean().unstack("Region")
    heat_n = g.groupby(["Disease", "Region"])["time_lag_years"].count().unstack("Region")
    heat = heat.reindex(index=diseases, columns=regions)
    heat_n = heat_n.reindex(index=diseases, columns=regions)
    annot = heat.copy().astype(object)
    for d in heat.index:
        for r in heat.columns:
            m, n = heat.loc[d, r], heat_n.loc[d, r]
            if pd.isna(m) or pd.isna(n):
                annot.loc[d, r] = ""
            else:
                annot.loc[d, r] = f"{m:.0f}\n(n={int(n)})"

    n_d = len(diseases)
    fig = plt.figure(figsize=(13.8, 0.95 * n_d + 1.6))
    gs = GridSpec(
        n_d,
        2,
        figure=fig,
        width_ratios=[1.15, 1.05],
        wspace=0.18,
        hspace=0.08,
        left=0.11,
        right=0.98,
        top=0.90,
        bottom=0.14,
    )
    axes_l = [fig.add_subplot(gs[i, 0]) for i in range(n_d)]
    ax_h = fig.add_subplot(gs[:, 1])

    centers = 0.5 * (bins[:-1] + bins[1:])
    for i, (ax, (d, vals, counts)) in enumerate(zip(axes_l, counts_all)):
        ax.bar(
            centers,
            counts,
            width=0.92,
            color=DISEASE_COLORS.get(d, "#888888"),
            edgecolor="white",
            linewidth=0.35,
            align="center",
        )
        if len(vals):
            med = float(np.median(vals))
            ax.axvline(med, color="#222222", lw=1.0, ls="--", zorder=3)
            ax.text(
                0.98,
                0.82,
                f"n={len(vals)}  med={med:.0f}y",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="white",
            )
        ax.set_ylabel(d, rotation=0, ha="right", va="center", fontsize=9, labelpad=6)
        ax.set_ylim(0, ymax * 1.15)
        ax.set_yticks([0, ymax])
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_xlim(-0.5, 25.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i < n_d - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0, labelbottom=False)
        else:
            ax.set_xlabel("Lag (years)", fontsize=9)
            ax.tick_params(axis="x", labelsize=8)
    axes_l[0].text(
        -0.02,
        1.18,
        "B",
        transform=axes_l[0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
    axes_l[0].set_title(
        "Distribution of trajectory lags",
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=6,
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "nature_reds",
        ["#FFF5F0", "#FCBBA1", "#FB6A4A", "#CB181D", "#67000D"],
    )
    vmax = max(14, float(np.nanmax(heat.values)) if np.isfinite(heat.values).any() else 14)
    hm =     sns.heatmap(
        heat,
        ax=ax_h,
        cmap=cmap,
        annot=annot,
        fmt="",
        annot_kws={"size": 10, "color": "white", "linespacing": 1.1},
        linewidths=0.55,
        linecolor="white",
        cbar=False,
        vmin=0,
        vmax=vmax,
    )
    ax_h.set_xlabel("")
    ax_h.set_ylabel("")
    ax_h.set_yticklabels(diseases, rotation=0, fontsize=9)
    ax_h.set_xticklabels(
        [REGION_SHORT.get(r, r) for r in regions],
        rotation=0,
        fontsize=7.5,
        ha="center",
    )
    ax_h.xaxis.set_ticks_position("top")
    ax_h.xaxis.set_label_position("top")
    ax_h.tick_params(axis="x", length=0, pad=2)
    ax_h.tick_params(axis="y", length=0, pad=2)
    # colorbar below heatmap
    cax = fig.add_axes([0.58, 0.045, 0.32, 0.018])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Mean lag (years)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax_h.set_title("Mean lag by region × disease", fontsize=10, fontweight="bold", pad=18)

    savefig(fig, "13_lag_distribution_heatmap_combo")


def plot_extra_figures(res: pd.DataFrame, meta: pd.DataFrame, cei: pd.DataFrame) -> None:
    slow = res[res["group"] == "global_south"].copy()
    meta_cols = meta[["GBD_NAME", "Income_group", "Region", "Code"]].drop_duplicates("GBD_NAME")
    slow = slow.drop(columns=[c for c in ["Code", "Income_group", "Region", "GBD_NAME"] if c in slow.columns], errors="ignore")
    slow = slow.merge(meta_cols, left_on="Country", right_on="GBD_NAME", how="left")
    reliable = slow[slow["match_quality"] != "weak"]

    # A: Circulation lag by income group
    circ = reliable[reliable["Disease"] == "Circulation"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    order = ["Low income", "Lower middle income"]
    data = [circ.loc[circ["Income_group"] == ig, "time_lag_years"].dropna() for ig in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.55)
    colors = ["#E64B35", "#4DBBD5"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_ylabel("Time lag vs China (years)")
    ax.set_title("Circulation lag by income group", fontsize=12, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    savefig(fig, "13_lag_by_income_group_Circulation")

    # C: top urgent (shortest policy window) Circulation
    circ2 = reliable[reliable["Disease"] == "Circulation"].sort_values(
        ["policy_window_years_from_now", "r_shape"], ascending=[True, False]
    ).head(20)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    closed = circ2["policy_window_years_from_now"] <= 0
    vals = circ2["policy_window_years_from_now"].clip(lower=0)
    colors = np.where(closed, "#8C8C8C", "#E64B35")
    ax.barh(circ2["Country"], vals, color=colors, height=0.72)
    for i, (_, r) in enumerate(circ2.iterrows()):
        lab = r["policy_window_label"]
        ax.text(float(vals.iloc[i]) + 0.15, i, "-" if lab == "-" else lab, va="center", fontsize=7)
    ax.set_xlabel("Policy window (years; '-' = closed)")
    ax.set_title("Most urgent policy windows — Circulation", fontsize=12, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    savefig(fig, "13_urgent_policy_windows_Circulation")

    # B: 左分布 + 右热图（病种行 × 地区列，红色色阶）拼接
    plot_lag_dist_heatmap_combo(reliable)


def write_readme(meta: pd.DataFrame, res: pd.DataFrame) -> None:
    matched = meta[meta["matched"]]
    slow = res[res["group"] == "global_south"]
    lines = [
        "# code_12_update",
        "",
        "Country selection: World Bank CLASS (List of economies)",
        f"- China income group: {meta['China_Income_group'].iloc[0] if len(meta) else 'UMIC'}",
        "- Slower than China: Low income + Lower middle income",
        f"- CLASS economies in scope: {len(meta)}",
        f"- Matched to emission/GBD panel: {int(meta['matched'].sum())}",
        f"- Timelag rows: {len(slow)} (all diseases × selected countries)",
        "",
        "Method: same as code_12_cal_lagtime.py (Z-weighted national wCEI + shape matching).",
        "Outputs: tables + world lag maps + income/region summaries.",
    ]
    (OUT / "README.txt").write_text("\n".join(lines), encoding="utf-8")




def main() -> None:
    set_style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print(">>> [1] standardized national wCEI ...")
    cei = build_standardized_national_wcei()
    # build_standardized 会写到 OUT_DIR（已改）

    print(">>> [2] select slower-than-China countries from CLASS ...")
    meta = select_slower_than_china(cei)
    meta.to_csv(OUT / "3_CLASS_slower_than_china_all.csv", index=False, encoding="utf-8-sig")
    matched = meta[meta["matched"]].copy()
    matched.to_csv(OUT / "3_CLASS_slower_than_china_matched.csv", index=False, encoding="utf-8-sig")
    countries = sorted(matched["GBD_NAME"].unique().tolist())
    print(f"    matched countries: {len(countries)} / {len(meta)}")

    print(">>> [3] timelag match ALL selected countries x diseases ...")
    res = run_timelag_all(cei, countries)
    # attach income meta
    res = res.merge(
        matched[["GBD_NAME", "Code", "Income_group", "Region"]].rename(columns={"GBD_NAME": "Country"}),
        on="Country",
        how="left",
    )
    res.to_csv(OUT / "4_timelag_alignment_results.csv", index=False, encoding="utf-8-sig")
    reliable = res[(res["group"] == "global_south") & (res["match_quality"] != "weak")]
    reliable.to_csv(OUT / "4_timelag_reliable_matches.csv", index=False, encoding="utf-8-sig")
    circ = reliable[reliable["Disease"] == "Circulation"].sort_values("policy_window_years_from_now")
    circ.to_csv(OUT / "4_timelag_Circulation_ranked.csv", index=False, encoding="utf-8-sig")
    print(f"    rows={len(res)}, reliable={len(reliable)}")

    print(">>> [4] figures ...")
    # Figures from the embedded original plotting functions
    try:
        plot_early_warning_trajectories(cei)
    except Exception as e:
        print(f"  early-warning skip: {e}")
    try:
        plot_lag_results(res)
    except Exception as e:
        print(f"  lag-results skip: {e}")
    try:
        plot_alignment_overlays(cei, res)
    except Exception as e:
        print(f"  overlays skip: {e}")

    plot_world_lag_map(res, matched, "Circulation")
    # 7 个病种全覆盖（与 code_12 DISEASE_ORDER 一致）
    all_diseases = [d for d in DISEASE_ORDER if d in set(res["Disease"])]
    plot_world_lag_maps_multi(res, matched, all_diseases)
    plot_extra_figures(res, matched, cei)
    write_readme(meta, res)

    # 摘要
    print("\n=== Circulation shortest policy windows (top 10) ===")
    cols = ["Country", "Income_group", "time_lag_years", "r_shape", "policy_window_label", "china_match_start", "china_match_end"]
    try:
        print(circ.head(10)[cols].to_string(index=False))
    except Exception:
        print(circ.head(10)[cols].astype(str).to_string(index=False).encode("ascii", "replace").decode())

    print(f"\ndone → {OUT}")



    # %%

if __name__ == "__main__":
    main()
