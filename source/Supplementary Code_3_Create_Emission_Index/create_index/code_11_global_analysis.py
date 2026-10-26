#%%
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(BASE_PATH)
LIU_POLS_PATH = BASE_DIR / 'data_raw' / 'df_emi_global.xlsx'
GEMS_PATH = BASE_DIR / 'data_raw' / 'PAH_country_annual_emission.csv'
UNIT_FILE = os.path.join(BASE_PATH, '..', '..', '..', '..', '4_emi_describe', 'data_raw', 'pollutants & source description.xlsx')
OUT_DIR = BASE_DIR / 'code_11_results'

data1 = pd.read_excel(LIU_POLS_PATH)
data2 = pd.read_csv(GEMS_PATH)

df_long = data1.melt(
    id_vars=['GBD_NAME', 'Year'],       # 保持不变的列（作为索引键）
    var_name='pollutant',           # 原列名变成的新列名（化学物类型）
    value_name='Value'                  # 原数值变成的新列名
)

df_long['sector'] = 'CK'
df_long['pollutant'] = [i[:-3] for i in df_long['pollutant']]
data2.rename(columns={'year':'Year'},inplace=True)

unit_multipliers = {
    'kg': 1000,
    'g': 1,
    'mg': 1e-3,
    'ug': 1e-6  
}

if os.path.exists(UNIT_FILE):
    df_units = pd.read_excel(UNIT_FILE)
    # 将单位字符串转换为小写并去除空格
    df_units['Unit'] = df_units['Unit'].astype(str).str.strip().str.lower()
    df_units['Multiplier'] = df_units['Unit'].map(unit_multipliers).fillna(1.0)
    
    # 与主表合并
    df_long.rename(columns={'pollutant':'Pollutant abbr.'},inplace=True)
    df_merged = df_long.merge(df_units[['Pollutant abbr.', 'Multiplier']], on='Pollutant abbr.', how='left')
    df_merged.dropna(inplace=True)
else:
    print(f"  [警告] 未找到单位映射文件 {UNIT_FILE}！")
    print("  [提示] 将默认采用 1.0 权重，如果需要精确换算，请准备该 CSV 文件。")
    df_long['Multiplier'] = 1.0

# 计算统一到 g 后的真实质量排放量
df_merged['emission_g'] = df_merged['Value'] * df_merged['Multiplier']
df_merged.rename(columns={'Pollutant abbr.':'pollutant'},inplace=True)
data = pd.concat([df_merged[data2.columns.tolist()],data2],axis=0).reset_index(drop=True)
data.sort_values(by=['GBD_NAME','pollutant','Year'],inplace=True)
data = data.reset_index(drop=True)
data.to_csv(OUT_DIR / '1_global_emission_data.csv', index=False)
#%% create global cei
import re

PAIR_PATH = os.path.join(BASE_DIR,'code_3_results/3_Final_Top50pct_Categorized_Targets.csv')

print(">>> 正在加载并解析 wCEI (混合暴露指数) 的靶标权重字典...")
df_priority = pd.read_csv(PAIR_PATH)
if 'Label' in df_priority.columns:
    disease_col = 'Label'
elif 'Disease_Name' in df_priority.columns:
    disease_col = 'Disease_Name'
else:
    raise KeyError("在 PAIR_PATH 中找不到表示疾病的列。")

cei_weights = {}
all_targets_to_extract = set()

for disease in df_priority[disease_col].unique():
    disease_df = df_priority[df_priority[disease_col] == disease]
    temp_dict = {}
    
    for idx, row in disease_df.iterrows():
        weight = row.get('Priority_Index', 1.0)
        expo_raw = str(row['Exposure']).strip()
        
        expos = re.split(r',\s*(?=X_)', expo_raw)
        expos = [e.strip() for e in expos if e.strip()]
        
        for e in expos:
            if e in temp_dict:
                temp_dict[e] = max(temp_dict[e], weight)
            else:
                temp_dict[e] = weight
                
            all_targets_to_extract.add(e)
            
    cei_weights[disease] = temp_dict

print(f"    - 解析出 {len(cei_weights)} 种结局疾病的靶标网络。")
print(f"    - 共有 {len(all_targets_to_extract)} 种唯一核心污染物质进入计算池。")

if 'Cancer' in cei_weights:
    del cei_weights['Cancer']
if 'Neural' in cei_weights:
    del cei_weights['Neural']


#%% 构建国家尺度疾病特异 CEI
def parse_exposure_token(token: str) -> tuple[str, str]:
    """X_Cd_CK / X_3,6-BCZ_MSWI / X_BaP_ico → (pollutant, sector)"""
    e = str(token).strip()
    if e.startswith("X_"):
        e = e[2:]
    if "_" not in e:
        raise ValueError(f"无法解析 Exposure: {token}")
    pollutant, sector = e.rsplit("_", 1)
    return pollutant, sector


# 1) 整理为长表权重: Disease × pollutant × sector × weight
weight_rows = []
for disease, wdict in cei_weights.items():
    for expo, w in wdict.items():
        try:
            pol, sec = parse_exposure_token(expo)
        except ValueError:
            print(f"  [跳过] 无法解析: {expo}")
            continue
        weight_rows.append(
            {
                "Disease": disease,
                "Exposure": expo,
                "pollutant": pol,
                "sector": sec,
                "weight": float(w),
            }
        )

df_w = pd.DataFrame(weight_rows)
# 同病×同物质×同部门若重复，取最大权重（与上方字典逻辑一致）
df_w = (
    df_w.groupby(["Disease", "pollutant", "sector", "Exposure"], as_index=False)["weight"]
    .max()
)
df_w.to_csv(OUT_DIR / "2_disease_pollutant_sector_weights.csv", index=False)
print(f">>> 权重长表: {len(df_w)} 条疾病-污染物-部门记录")

# 2) 读取全球排放（优先用已导出文件，便于重复运行）
emi_path = OUT_DIR / "1_global_emission_data.csv"
if emi_path.exists():
    df_emi = pd.read_csv(emi_path)
    print(f">>> 加载排放: {emi_path}  ({len(df_emi)} 行)")
else:
    df_emi = data.copy()
    print(">>> 使用内存中的 data 作为排放表")

need_cols = {"GBD_NAME", "Year", "pollutant", "sector", "emission_g"}
missing = need_cols - set(df_emi.columns)
if missing:
    raise KeyError(f"排放表缺列: {missing}")

df_emi = df_emi.copy()
df_emi["Year"] = df_emi["Year"].astype(int)
df_emi["pollutant"] = df_emi["pollutant"].astype(str).str.strip()
df_emi["sector"] = df_emi["sector"].astype(str).str.strip()
df_emi["emission_g"] = pd.to_numeric(df_emi["emission_g"], errors="coerce").fillna(0.0)

# 同国家同年同物质同部门求和
df_emi_agg = (
    df_emi.groupby(["GBD_NAME", "Year", "pollutant", "sector"], as_index=False)["emission_g"]
    .sum()
)

# 3) 对齐覆盖率检查
needed_pairs = df_w[["pollutant", "sector"]].drop_duplicates()
have_pairs = df_emi_agg[["pollutant", "sector"]].drop_duplicates()
needed_pairs = needed_pairs.assign(key=needed_pairs["pollutant"] + "|" + needed_pairs["sector"])
have_pairs = have_pairs.assign(key=have_pairs["pollutant"] + "|" + have_pairs["sector"])
cover = needed_pairs.merge(have_pairs[["key"]], on="key", how="left", indicator=True)
cover["available"] = cover["_merge"] == "both"
cover = cover.drop(columns=["_merge", "key"])
cover.to_csv(OUT_DIR / "2_weight_emission_match_report.csv", index=False)
n_ok = int(cover["available"].sum())
print(f">>> 权重对匹配排放: {n_ok}/{len(cover)}")
if n_ok < len(cover):
    miss = cover.loc[~cover["available"], ["pollutant", "sector"]]
    print("  缺失对:")
    print(miss.to_string(index=False))

# 4) 合并权重与排放 → 贡献，再汇总 CEI
df_contrib = df_emi_agg.merge(
    df_w[["Disease", "pollutant", "sector", "weight", "Exposure"]],
    on=["pollutant", "sector"],
    how="inner",
)
df_contrib["contribution"] = df_contrib["emission_g"] * df_contrib["weight"]

df_cei = (
    df_contrib.groupby(["GBD_NAME", "Year", "Disease"], as_index=False)
    .agg(
        CEI=("contribution", "sum"),
        n_targets_used=("Exposure", "nunique"),
    )
)

# 每个疾病的理论靶标数 → coverage
n_targets = df_w.groupby("Disease")["Exposure"].nunique().rename("n_targets_total")
df_cei = df_cei.merge(n_targets, on="Disease", how="left")
df_cei["coverage"] = df_cei["n_targets_used"] / df_cei["n_targets_total"]

# 宽表: 国家×年，各病 CEI 列
df_cei_wide = (
    df_cei.pivot_table(index=["GBD_NAME", "Year"], columns="Disease", values="CEI", aggfunc="sum")
    .reset_index()
)
df_cei_wide.columns = [
    c if c in ("GBD_NAME", "Year") else f"CEI_{c}" for c in df_cei_wide.columns
]

# 贡献份额
df_cei_tot = df_cei[["GBD_NAME", "Year", "Disease", "CEI"]].rename(columns={"CEI": "CEI_total"})
df_contrib = df_contrib.merge(df_cei_tot, on=["GBD_NAME", "Year", "Disease"], how="left")
df_contrib["share"] = np.where(
    df_contrib["CEI_total"].abs() > 0,
    df_contrib["contribution"] / df_contrib["CEI_total"],
    np.nan,
)

# 5) 导出
out_long = OUT_DIR / "3_national_CEI_long.csv"
out_wide = OUT_DIR / "3_national_CEI_wide.csv"
out_contrib = OUT_DIR / "3_national_CEI_contributions.csv"
df_cei.to_csv(out_long, index=False)
df_cei_wide.to_csv(out_wide, index=False)
df_contrib.to_csv(out_contrib, index=False)

print(f"✅ 国家 CEI 长表: {out_long}  ({len(df_cei)} 行)")
print(f"✅ 国家 CEI 宽表: {out_wide}  ({len(df_cei_wide)} 行)")
print(f"✅ 贡献分解: {out_contrib}  ({len(df_contrib)} 行)")
print(
    f"   国家数={df_cei['GBD_NAME'].nunique()}, "
    f"年份={df_cei['Year'].min()}-{df_cei['Year'].max()}, "
    f"疾病={sorted(df_cei['Disease'].unique().tolist())}"
)

# 6) 简要：最近一年各国 CEI 排名（各病 Top10）
latest = int(df_cei["Year"].max())
rank_rows = []
for disease in sorted(df_cei["Disease"].unique()):
    sub = (
        df_cei[(df_cei["Year"] == latest) & (df_cei["Disease"] == disease)]
        .sort_values("CEI", ascending=False)
        .head(10)
        .copy()
    )
    sub["rank"] = range(1, len(sub) + 1)
    rank_rows.append(sub)
df_rank = pd.concat(rank_rows, ignore_index=True)
df_rank.to_csv(OUT_DIR / "3_national_CEI_rank_latest_top10.csv", index=False)
print(f"✅ 最近年({latest}) Top10: {OUT_DIR / '3_national_CEI_rank_latest_top10.csv'}")

#%%
