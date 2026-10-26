#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
import os
from matplotlib_venn import venn2

# Set working directory (using relative path)
base_dir = os.path.dirname(os.path.abspath(__file__))
ebio_dir = os.path.join(os.path.dirname(base_dir),'ebio_network')
statistic_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)),'2_association','code_2_results')
output_path = os.path.join(base_dir, 'code_7_ur_results')
os.makedirs(output_path, exist_ok=True)

#pols data
id_corrected_df = pd.read_excel(os.path.join(ebio_dir, 'id_corrected.xlsx'))
id_corrected_df.rename(columns={'unified_id': 'Exposure_ID'}, inplace=True)

#statistics results
EWAS_df = pd.read_csv(os.path.join(statistic_dir, 'EWAS_Results_Significance_ratio0.3.csv'))

def get_abbr(name):
    parts = name.split('_')
    if len(parts) > 2:
        return '_'.join(parts[1:-1])
    else:
        return parts[-1]

EWAS_df['name_en_abbr'] = EWAS_df['Exposure'].apply(get_abbr)
EWAS_df.rename(columns={'Outcome': 'Disease_Name'}, inplace=True)
keep_exp = id_corrected_df[id_corrected_df['Exposure_ID'].str.startswith('EX:E')]
id_df = keep_exp.drop_duplicates(subset='Exposure_ID')

df_final = pd.merge(EWAS_df,id_df[['name_en_abbr','catogory_en','sub category_en','Exposure_ID']],on='name_en_abbr',how='left')
df_final.to_csv(os.path.join(output_path, '1_statistic_data_with_info.csv'), index=False)
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
import os

# Set working directory (using relative path)
base_dir = os.path.dirname(os.path.abspath(__file__))
ebio_dir = os.path.join(os.path.dirname(base_dir),'ebio_network')
PATHWAYs = ['EPPD','EGoD']
for PATHWAY in PATHWAYs:
    csv_file = os.path.join(ebio_dir, 'results_pds_score',f'path_diversity_score_{PATHWAY}','exposure_disease_path_diversity_scores.csv')
    df = pd.read_csv(csv_file)

    pols_data = pd.read_csv(os.path.join(base_dir,'code_7_ur_results','1_statistic_data_with_info.csv'))
    pols_data['Disease_Name'] = [name[2:] for name in pols_data['Disease_Name']]
    pols_data = pols_data[~(pols_data['Disease_Name']=='All_cause')].reset_index(drop=True)
    merged_df = pd.merge(pols_data, df, on=['Exposure_ID','Disease_Name'], how='left')

    df_without_ebio = merged_df[merged_df.isna().any(axis=1)]
    df_without_ebio['Exposure'].unique()
    merged_df.to_csv(os.path.join(output_path,f'2_{PATHWAY}_merged_data.csv'))

#%%合并两种路径的评分
import pandas as pd
import os
df_EPPD = pd.read_csv(os.path.join(output_path,f'2_EPPD_merged_data.csv'))
df_EGoD = pd.read_csv(os.path.join(output_path,f'2_EGoD_merged_data.csv'))
df_EPPD.rename(columns={'PDS_Weighted':'PDS_EPPD'},inplace=True)
df_EGoD.rename(columns={'PDS_Weighted':'PDS_EGoD'},inplace=True)
scores = pd.concat([df_EPPD['PDS_EPPD'],df_EGoD['PDS_EGoD']],axis=1)
df_final = pd.concat([df_EPPD,scores],axis=1)
df_final.to_csv(os.path.join(output_path,'3_combined_2_pds_data.csv'))

# %%
import pandas as pd
import os
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_path,'code_7_ur_results')

tot_path = os.path.join(base_path,'code_7_ur_results/3_combined_2_pds_data.csv')
carbazole_path = os.path.join(base_path,'code_2_results/Disease_Specific_DeepTox_Scores.csv')

df_tot = pd.read_csv(tot_path)
df_carbazole = pd.read_csv(carbazole_path)

id_cols = ['smiles', 'Name_Abbr']

df_long = df_carbazole.melt(
    id_vars=id_cols,          
    var_name='Disease_Name',       
    value_name='DTL_Score'      
)

df_long.rename(columns={'Name_Abbr':'name_en_abbr'},inplace=True)
df_final = pd.merge(df_tot,df_long.iloc[:,1:],on=['name_en_abbr','Disease_Name'],how='left')
df_final.to_csv(os.path.join(output_dir,'4_combined_all_method_score.csv'))
cols_to_check = ['PDS_EPPD', 'PDS_EGoD', 'DTL_Score']
nan_rows = df_final[df_final[cols_to_check].isna().all(axis=1)]
nan_rows['Exposure'].unique()
print(nan_rows)  #三个平台都没有数据

#%%
print(">>> 正在执行多维毒理学机制融合")
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path,'code_7_ur_results/4_combined_all_method_score.csv')
df_final = pd.read_csv(file_path)

print(f">>> 成功加载 df_final，共 {len(df_final)} 条关联记录。")

# 为了处理可能缺失的类别信息，填充为 'Unknown'
if 'sub category_en' in df_final.columns:
    df_final['sub category_en'] = df_final['sub category_en'].fillna('Unknown')

# ==========================================
# 2. 核心算法：三大维度【完全统一】的组内 Min-Max 归一化
# ==========================================
print(">>> 正在执行 [疾病+大类] 维度的完全对称归一化与联合概率融合...")

group_cols = ['Disease_Name', 'sub category_en']

# 定义一个安全的 Min-Max 归一化函数
def min_max_scale(series):
    # 忽略 NaN
    s_valid = series.dropna()
    if len(s_valid) == 0:
        return series
    
    s_min = s_valid.min()
    s_max = s_valid.max()
    
    # 防止分母为 0 (整个组内所有值都一样)
    if s_max == s_min:
        # 如果唯一的值大于 0，说明都有同等强度的证据，赋 1.0；如果是 0，赋 0.0
        return series.apply(lambda x: 1.0 if pd.notna(x) and x > 0 else (0.0 if pd.notna(x) else np.nan))
        
    return (series - s_min) / (s_max - s_min)

# A. 严格对称！将 EPPD, EGoD, DTL 三种得分，全部转化为组内的 0~1 标度 (保留真实量级差距)
df_final['S_eppd'] = df_final.groupby(group_cols)['PDS_EPPD'].transform(min_max_scale)
df_final['S_egod'] = df_final.groupby(group_cols)['PDS_EGoD'].transform(min_max_scale)
df_final['S_dtl']  = df_final.groupby(group_cols)['DTL_Score'].transform(min_max_scale)

# B. 填补结构性缺失：NaN 意味着“该方法在该维度未查到证据”，填补为 0
df_final['S_eppd'] = df_final['S_eppd'].fillna(0)
df_final['S_egod'] = df_final['S_egod'].fillna(0)
df_final['S_dtl']  = df_final['S_dtl'].fillna(0)

# C. 同源数据去重：EPPD 和 EGoD 都来自知识图谱，取两者的最大值作为文献证据的分数
df_final['S_KG'] = df_final[['S_eppd', 'S_egod']].max(axis=1)

# D. 异源交叉验证：联合概率累积 (Joint Probability OR)
# 公式：P_total = 1 - (1 - P_KG) * (1 - P_DTL)
df_final['Unified_Bio_Score'] = 1 - (1 - df_final['S_KG']) * (1 - df_final['S_dtl'])

# ==========================================
# 3. 计算最终优先管控指数 (Priority Index)
# ==========================================
print(">>> 正在结合流行病学证据计算 Priority Index...")

# 计算流行病学纯统计学得分 (Effect Size * Significance)
# 为防止 P 值为 0 导致 log 爆炸，加上 1e-300
df_final['Epi_Score_Urban'] = df_final['LogRR_Urban'] * (-np.log10(df_final['P_value_Urban'] + 1e-300))
df_final['Epi_Score_Rural'] = df_final['LogRR_Rural'] * (-np.log10(df_final['P_value_Rural'] + 1e-300))

# 剔除流行病学表现为“保护作用”的负值噪声 (LogRR < 0 即 RR < 1)
df_final.loc[df_final['Epi_Score_Urban'] < 0, 'Epi_Score_Urban'] = 0
df_final.loc[df_final['Epi_Score_Rural'] < 0, 'Epi_Score_Rural'] = 0

# 【终极融合】：机制折损 (流行病学得分 * 统一生物学证据得分)
df_final['Priority_Index_Rural'] = df_final['Epi_Score_Rural'] * df_final['Unified_Bio_Score']
df_final['Priority_Index_Urban'] = df_final['Epi_Score_Urban'] * df_final['Unified_Bio_Score']

# ==========================================
# 4. 排序、清洗与导出
# ==========================================
# 过滤掉得分为 0 的无效靶标
df_priority = df_final[df_final['Priority_Index_Urban'] > 0].copy()
df_priority = df_priority[df_priority['Priority_Index_Rural'] > 0].copy()
output_file = os.path.join(output_dir,'5_Integrated_Priority_Targets_Categorized_min_max.csv')
df_priority.to_csv(output_file, index=False)

print(f"\n✅ 严谨 Min-Max 融合计算完成！有效靶标数量：{len(df_priority)}")
print(f"✅ 结果已保存至：{output_file}")
#%%
import pandas as pd
import os

# ==========================================
# 1. 加载包含 Priority_Index 的数据
# ==========================================
file_path = os.path.join(output_dir,'5_Integrated_Priority_Targets_Categorized_min_max.csv')
df = pd.read_csv(file_path)
df_Urban = df[df.Significance_raw_Urban =='Significant Harm']
df_Rural = df[df.Significance_raw_Rural =='Significant Harm']

def filter_top50(df,name):
    # ==========================================
    # 2. 分层帕累托筛选 (每个大类内部选前 50%)
    # ==========================================
    print(">>> 正在按 [疾病 + 物质大类] 双维度执行帕累托累积贡献率筛选...")

    # A. 确保在每组内部按优先级降序排列
    df = df.sort_values(by=['Disease_Name', 'sub category_en', f'Priority_Index_{name}'], 
                        ascending=[True, True, False])

    # B. 计算每个分组的总风险得分
    group_cols = ['Disease_Name', 'sub category_en']
    df['Group_Total_Risk'] = df.groupby(group_cols)[f'Priority_Index_{name}'].transform('sum')

    # C. 计算单个物质在【该类别内部】的贡献百分比
    df['Contribution_Pct'] = df[f'Priority_Index_{name}'] / df['Group_Total_Risk']

    # D. 计算累积贡献百分比 (在每个类别内独立累积)
    df['Cum_Contribution_Pct'] = df.groupby(group_cols)['Contribution_Pct'].cumsum()

    # ==========================================
    # 3. 截断逻辑：每类选前 50%
    # ==========================================
    def extract_top_50_percent(group):
        # 只要之前的累积贡献还没超过 50%，就保留该物质
        # shift(1) 确保包含那个“跨越” 50% 阈值的物质
        mask = group['Cum_Contribution_Pct'].shift(1).fillna(0) < 0.50
        return group[mask]

    # 应用截断逻辑
    df_top50 = df.groupby(group_cols, as_index=False).apply(extract_top_50_percent).reset_index(drop=True)

    # ==========================================
    # 4. 导出结果与统计
    # ==========================================
    output_file = os.path.join(output_dir,f'6_Final_Top50pct_Categorized_Targets_{name}.csv')
    df_top50.to_csv(output_file, index=False)

    print(f"\n✅ 分层筛选完成！浓缩出 {len(df_top50)} 个核心优先管控物质。")
    print(f"✅ 白名单已保存至：{output_file}")

    # 统计一下每个大类被选中的情况，供您确认覆盖度
    summary = df_top50.groupby(['Disease_Name', 'sub category_en']).size().reset_index(name='Count')
    print("\n>>> 各疾病下各化学类别的筛选结果统计:")
    print(summary.head(10)) 
filter_top50(df_Urban,'Urban')
filter_top50(df_Rural,'Rural')
#%%
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# =========================================================
# 0. 环境设置与数据加载
# =========================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 输入文件路径 
names = ['Urban','Rural']

base_dir = os.path.dirname(os.path.abspath(__file__))
for name in names:
    input_file = os.path.join(base_dir,f'code_7_ur_results/6_Final_Top50pct_Categorized_Targets_{name}.csv')
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"未找到对应的输入文件：{input_file}")

    df_raw = pd.read_csv(input_file)

    # 统一核心列名映射
    disease_col = 'Disease_Name'
    category_col = 'sub category_en'
    index_sum_col = 'Priority_Index_Sum'
    total_sum_col = 'Group_Total_Sum'
    sectors_col = 'Top3_Unique_Sectors'
    pollutants_col = 'Top3_Unique_Pollutants'

    # =========================================================
    # 1. 核心改进：明细数据动态聚合 (Data Aggregation)
    # =========================================================
    print(">>> 正在将明细靶标聚合为矩阵图所需的分类汇总格式...")

    # 确保排序正确，以便优先提取高风险污染物名称
    df_raw = df_raw.sort_values(by=[disease_col, category_col, f'Priority_Index_{name}'], ascending=[True, True, False])

    # 自定义聚合函数：提取排名前3的非重复污染物和主导部门
    def get_top_pollutants(x):
        # 去掉 X_ 前缀使其更易读
        clean_names = x.str.replace('^X_', '', regex=True)
        clean_names = clean_names.str.replace(r'_[^_\\]+(?=\\n|$)', '', regex=True)
        return '\n'.join(clean_names.unique()[:3])

    def get_primary_sector(x):
        # 提取出现次数最多的部门作为主导部门
        sectors = x.apply(lambda s: str(s).split('_')[-1] if '_' in str(s) else 'Other')
        return sectors.mode()[0] if not sectors.mode().empty else 'Other'

    # 执行聚合
    df_summary = df_raw.groupby([disease_col, category_col]).agg(
        Priority_Index_Sum=(f'Priority_Index_{name}', 'sum'),
        Contribution_Pct=('Contribution_Pct', 'sum'), # 由于筛选的是前50%，这里的和就是实际入选物质的贡献总和
        Top3_Unique_Pollutants=('Exposure', get_top_pollutants),
        Primary_Sector=('Exposure', get_primary_sector)
    ).reset_index()

    # 计算每种疾病的全局总风险分，用于计算气泡的相对大小 (在当前行的占比)
    disease_totals = df_raw.groupby(disease_col)[f'Priority_Index_{name}'].sum().reset_index(name=total_sum_col)
    df_summary = pd.merge(df_summary, disease_totals, on=disease_col, how='left')

    # 重新计算在整个疾病行中的贡献百分比
    df_summary['Contribution_Pct'] = (df_summary['Priority_Index_Sum'] / df_summary[total_sum_col]) * 100

    # =========================================================
    # 2. 数据清洗与网格坐标映射
    # =========================================================
    # A. 污染物名称清洗
    # 删除最后一个下划线及之后的内容
    # 这里的 \\n 匹配的是你数据里的文本 "\n"
    # (?=...) 是前瞻断言，确保只删除后缀，不删掉后面的污染物名称
    df_summary[pollutants_col] = df_summary[pollutants_col].replace({'Hexachlorobutadiene': 'HCBD'})

    # B. 疾病(Y轴)排序：根据每种疾病的总分，让高风险疾病排在图表上方
    disease_order = df_summary.groupby(disease_col)[index_sum_col].sum().sort_values(ascending=False).index
    df_summary[disease_col] = pd.Categorical(df_summary[disease_col], categories=disease_order, ordered=True)

    # C. 物质大类(X轴)排序：根据该大类的总分进行降序，重要的大类靠左
    category_order = df_summary.groupby(category_col)[index_sum_col].sum().sort_values(ascending=False).index
    df_summary[category_col] = pd.Categorical(df_summary[category_col], categories=category_order, ordered=True)

    # D. 建立离散网格的数字坐标系
    cat_to_x = {cat: i for i, cat in enumerate(category_order)}
    dis_to_y = {dis: i for i, dis in enumerate(disease_order)}

    df_summary['x_coord'] = df_summary[category_col].map(cat_to_x)
    df_summary['y_coord'] = df_summary[disease_col].map(dis_to_y)

    # E. 部门字体颜色映射：为每个Primary Sector分配一个深色系高对比度颜色
    unique_sectors = df_summary['Primary_Sector'].unique()
    sector_palette = sns.color_palette("Dark2", n_colors=len(unique_sectors)).as_hex()
    sector_colors = dict(zip(unique_sectors, sector_palette))

    # =========================================================
    # 3. 高保真学术图表绘制 (自适应无重叠矩阵)
    # =========================================================
    print(">>> 正在渲染高级矩阵气泡图...")
    fig, ax = plt.subplots(figsize=(10, 12), dpi=300) 

    scatter = sns.scatterplot(
        data=df_summary,
        x='x_coord',
        y='y_coord',
        size='Contribution_Pct',
        hue=category_col,
        sizes=(800, 2500),    
        alpha=0.35,           
        palette='Set3',       
        edgecolor='gray',
        linewidth=1.2,
        ax=ax
    )

    ax.set_xticks(range(len(category_order)))
    ax.set_xticklabels(category_order, rotation=35, ha='right', fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(disease_order)))
    ax.set_yticklabels(disease_order, fontsize=11, fontweight='bold')
    ax.invert_yaxis()

    # =========================================================
    # 4. 气泡内部文本智能填充 (彩色部门名称 + 深灰污染物)
    # =========================================================
    for i, row in df_summary.iterrows():
        x = row['x_coord']
        y = row['y_coord']
        sector = row['Primary_Sector']
        pollutant = str(row[pollutants_col]).replace(', ', '\n')
        pct = row['Contribution_Pct']
        s_color = sector_colors.get(sector, '#000000')
        
        # 第一行文本：工业部门（专属彩色字体）
        ax.text(x, y - 0.15, f"[{sector}]", 
                fontdict={'size': 7.5, 'weight': 'bold', 'color': s_color},
                ha='center', va='center', zorder=10)
        
        # 第二/三行文本：污染物名称及贡献度
        ax.text(x, y + 0.15, f"{pollutant}\n({pct:.0f}%)", 
                fontdict={'size': 7.0, 'weight': 'bold', 'color': '#2c3e50'},
                ha='center', va='center', zorder=10)

    # =========================================================
    # 5. 出版级排版美化与定制图例
    # =========================================================
    ax.set_title('Exposome Matrix: Prioritized Chemical Classes and Industrial Drivers by Disease\n(Top 50% Cumulative Risk Targets)', 
                fontsize=17, fontweight='bold', pad=25)
    ax.set_xlabel('Chemical Sub-Categories (Columns)', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('Disease Outcomes (Rows Ranked by Risk)', fontsize=13, fontweight='bold', labelpad=12)

    ax.grid(True, axis='both', linestyle=':', alpha=0.6, color='grey')
    ax.set_axisbelow(True)
    ax.set_xlim(-0.8, len(category_order) - 0.2)
    ax.set_ylim(len(disease_order) - 0.2, -0.8) 

    # --- 定制双图例系统 ---
    if ax.legend_:
        ax.legend_.remove()

    # 图例 1：提取尺寸(占比)图例
    handles, labels = scatter.get_legend_handles_labels()
    if 'Contribution_Pct' in labels:
        size_idx = labels.index('Contribution_Pct')
        size_handles = handles[size_idx+1:]
        size_labels = [f"{float(l):.0f}%" for l in labels[size_idx+1:] if l.replace('.','',1).isdigit()]
        
        leg_size = ax.legend(size_handles, size_labels, title='Total Risk Contribution', 
                            bbox_to_anchor=(1.02, 0.5), loc='center left', frameon=True, fontsize=11, title_fontsize=12)
        ax.add_artist(leg_size)

    # 图例 2：工业部门字体颜色图例
    sector_legend_elements = [
        Patch(facecolor=col, edgecolor='white', label=sec) 
        for sec, col in sector_colors.items()
    ]
    leg_sector = ax.legend(handles=sector_legend_elements, title='Primary Source Sector', 
                        bbox_to_anchor=(1.02, 0.8), loc='center left', frameon=True, fontsize=10, title_fontsize=11)
    ax.add_artist(leg_sector)

    plt.subplots_adjust(right=0.75, bottom=0.15) 

    output_path = os.path.join(base_dir,f'code_7_ur_results/1_Matrix_Bubble_Chart_New_Targets_{name}.png')
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 完美！基于全新 Top50% 白名单的高级矩阵气泡图已保存至: {output_path}")

    # 导出汇总后的绘图数据供检查
    df_summary.to_csv(os.path.join(base_dir,f'code_7_ur_results/7_Aggregated_Matrix_Data_{name}.csv'))
# %%
"""
code_4_update.py
----------------
Nature 风格 Exposome 矩阵图（不覆盖原 code_4）：
- 圆圈大小统一
- 填充色深浅 = 病种内贡献百分比
- 气泡内显示污染物名称（不显示 %）
- 描边色 = 主导部门；图例含完整部门名称
输出 → code_4_results_update/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent
names = ['Rural', 'Urban']
for name in names:
    INPUT_FILE = BASE_DIR / "code_7_ur_results" / f"6_Final_Top50pct_Categorized_Targets_{name}.csv"
    OUT_DIR = BASE_DIR / "code_7_ur_results"

    # X 轴：短标签 + 全称（图注用）
    CAT_SHORT = {
        "Heavy metal": "Heavy metal",
        "Halogenated carbazole": "HCZ",
        "Metalloid": "Metalloid",
        "Mineral Dust": "Mineral dust",
        "Halogen": "Halogen",
        "Halogenated aliphatic hydrocarbon": "HAH",
        "Light metal": "Light metal",
        "Parent polycyclic aromatic hydrocarbon": "pPAH",
        "Methylated polycyclic aromatic hydrocarbon": "MePAH",
    }

    # 部门：描边色 + 完整图例名
    SECTOR_STYLE = {
        "CK": {"edge": "#2171b5", "label": "CK (coke oven)"},
        "ico": {"edge": "#238b45", "label": "ico (iron / industrial)"},
        "MSWI": {"edge": "#6a51a3", "label": "MSWI (waste incineration)"},
        "pow": {"edge": "#cb181d", "label": "pow (power plants)"},
        "ipr": {"edge": "#b35806", "label": "ipr (non-ferrous)"},
        "IOS": {"edge": "#636363", "label": "IOS"},
        "Other": {"edge": "#969696", "label": "Other"},
    }

    # Nature 风格顺序色：浅灰蓝 → 深蓝（色盲相对友好，避免刺眼黄红）
    NATURE_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "nature_contrib",
        ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    )


    def set_style() -> None:
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Times new roman"],
                "axes.unicode_minus": False,
                "axes.linewidth": 0.8,
                "xtick.major.width": 0.6,
                "ytick.major.width": 0.6,
                "xtick.major.size": 3.0,
                "ytick.major.size": 3.0,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )


    ABBR_FIX = {
        "Hexachlorobutadiene": "HCBD",
        "1,3,6,8-CCZ": "1368-CCZ",
        "1,3,6-BCZ": "136-BCZ",
    }


    def format_pollutant_label(names: list[str], max_n: int = 3) -> str:
        """气泡内污染物名：过长则截断条目数并缩短个别名称。"""
        cleaned = []
        for n in names:
            n = ABBR_FIX.get(n, n)
            if n not in cleaned:
                cleaned.append(n)
        # 名称过长时只保留前 2 个，避免挤爆圆圈
        if any(len(n) > 7 for n in cleaned):
            cleaned = cleaned[:2]
        else:
            cleaned = cleaned[:max_n]
        return "\n".join(cleaned)


    def get_primary_sector(x: pd.Series) -> str:
        sectors = x.astype(str).apply(lambda s: s.split("_")[-1] if "_" in s else "Other")
        mode = sectors.mode()
        return mode.iloc[0] if not mode.empty else "Other"


    def aggregate(df_raw: pd.DataFrame):
        disease_col = "Disease_Name"
        category_col = "sub category_en"

        df_raw = df_raw.sort_values(
            by=[disease_col, category_col, f"Priority_Index_{name}"],
            ascending=[True, True, False],
        )

        def _top_names(g: pd.DataFrame) -> str:
            # 优先用清单缩写；保持 Priority 降序
            if "name_en_abbr" in g.columns:
                names = g["name_en_abbr"].astype(str).tolist()
            else:
                names = (
                    g["Exposure"]
                    .astype(str)
                    .str.replace(r"^X_", "", regex=True)
                    .str.replace(r"_[^_]+$", "", regex=True)
                    .tolist()
                )
            return format_pollutant_label(names, 3)

        rows = []
        for (dis, cat), g in df_raw.groupby([disease_col, category_col], sort=False):
            rows.append(
                {
                    disease_col: dis,
                    category_col: cat,
                    "Priority_Index_Sum": g[f"Priority_Index_{name}"].sum(),
                    "Contribution_Pct": g["Contribution_Pct"].sum(),
                    "Top3_Unique_Pollutants": _top_names(g),
                    "Primary_Sector": get_primary_sector(g["Exposure"]),
                }
            )
        df_summary = pd.DataFrame(rows)

        disease_totals = (
            df_raw.groupby(disease_col)[f"Priority_Index_{name}"]
            .sum()
            .reset_index(name="Group_Total_Sum")
        )
        df_summary = df_summary.merge(disease_totals, on=disease_col, how="left")
        df_summary["Contribution_Pct"] = (
            df_summary["Priority_Index_Sum"] / df_summary["Group_Total_Sum"] * 100.0
        )

        disease_order = (
            df_summary.groupby(disease_col)["Priority_Index_Sum"]
            .sum()
            .sort_values(ascending=False)
            .index
        )
        category_order = (
            df_summary.groupby(category_col)["Priority_Index_Sum"]
            .sum()
            .sort_values(ascending=False)
            .index
        )

        df_summary[disease_col] = pd.Categorical(
            df_summary[disease_col], categories=disease_order, ordered=True
        )
        df_summary[category_col] = pd.Categorical(
            df_summary[category_col], categories=category_order, ordered=True
        )
        df_summary["x_coord"] = df_summary[category_col].map(
            {c: i for i, c in enumerate(category_order)}
        )
        df_summary["y_coord"] = df_summary[disease_col].map(
            {d: i for i, d in enumerate(disease_order)}
        )
        return df_summary, list(disease_order), list(category_order)


    def _text_color(rgba) -> str:
        """按填充亮度自动选黑/白字。"""
        r, g, b = rgba[:3]
        # 相对亮度
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "#111111" if lum > 0.55 else "#ffffff"


    def plot_matrix(df: pd.DataFrame, disease_order: list, category_order: list) -> None:
        n_x, n_y = len(category_order), len(disease_order)
        fig = plt.figure(figsize=(10, 8.6), dpi=300)
        # 主图 + 右侧图例区
        gs = fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[4.6, 1.15],
            wspace=0.08,
            left=0.10,
            right=0.98,
            top=0.92,
            bottom=0.16,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")

        ax.set_xlim(-0.72, n_x - 0.28)
        ax.set_ylim(n_y - 0.28, -0.72)
        ax.set_xticks(range(n_x))
        ax.set_yticks(range(n_y))
        ax.set_xticklabels(
            [CAT_SHORT.get(c, c) for c in category_order],
            rotation=35,
            ha="right",
            fontsize=9,
            color="#222222",
        )
        ax.set_yticklabels(disease_order, fontsize=10, color="#222222")
        ax.tick_params(length=0)
        ax.set_xlabel("Chemical class", fontsize=10, color="#333333", labelpad=8)
        ax.set_ylabel("Disease outcome", fontsize=10, color="#333333", labelpad=8)

        # 极淡参考线（几乎不可见，仅辅助对齐）
        ax.set_xticks(np.arange(-0.5, n_x, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_y, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.35, color="#efefef", zorder=0)
        ax.grid(False, which="major")
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.7)

        vmin, vmax = 0.0, 100.0
        cmap = NATURE_CMAP
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        bubble_size = 1000
        for _, row in df.iterrows():
            pct = float(row["Contribution_Pct"])
            face = cmap(norm(pct))
            sec = str(row["Primary_Sector"])
            edge = SECTOR_STYLE.get(sec, SECTOR_STYLE["Other"])["edge"]
            ax.scatter(
                float(row["x_coord"]),
                float(row["y_coord"]),
                s=bubble_size,
                c=[face],
                edgecolors=edge,
                linewidths=1.35,
                alpha=1.0,
                zorder=3,
            )
            labels = str(row["Top3_Unique_Pollutants"]).strip()
            if labels and labels.lower() != "nan":
                n_lines = labels.count("\n") + 1
                fs = 7.0 #if n_lines >= 3 else (6.6 if n_lines == 2 else 7.2)
                ax.text(
                    float(row["x_coord"]),
                    float(row["y_coord"]),
                    labels,
                    ha="center",
                    va="center",
                    fontsize=fs,
                    fontweight="semibold",
                    color=_text_color(face),
                    linespacing=1.12,
                    zorder=4,
                    clip_on=True,
                )

        # ---------- 右侧完整图例 ----------
        y0 = 0.98
        ax_leg.text(
            0.0,
            y0,
            "Contribution",
            fontsize=9,
            fontweight="bold",
            color="#222222",
            transform=ax_leg.transAxes,
            va="top",
        )
        # 竖向 colorbar
        cax = ax_leg.inset_axes([0.08, 0.52, 0.18, 0.38])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_ticks([0, 25, 50, 75, 100])
        cbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.6, colors="#333333")
        cbar.outline.set_linewidth(0.6)
        ax_leg.text(
            0.42,
            0.71,
            "(%)",
            fontsize=8,
            color="#444444",
            transform=ax_leg.transAxes,
            va="center",
        )

        ax_leg.text(
            0.0,
            0.44,
            "Primary sector\n(circle edge)",
            fontsize=9,
            fontweight="bold",
            color="#222222",
            transform=ax_leg.transAxes,
            va="top",
            linespacing=1.25,
        )

        # 保证图例覆盖数据中全部部门 + 常用部门顺序
        sector_order = ["CK", "ico", "MSWI", "pow", "ipr", "IOS", "Other"]
        present = set(df["Primary_Sector"].astype(str))
        sectors_show = [s for s in sector_order if s in present]
        for s in sorted(present):
            if s not in sectors_show:
                sectors_show.append(s)

        handles = []
        labels = []
        for sec in sectors_show:
            sty = SECTOR_STYLE.get(sec, SECTOR_STYLE["Other"])
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markersize=9,
                    markerfacecolor="#f7fbff",
                    markeredgecolor=sty["edge"],
                    markeredgewidth=2.0,
                )
            )
            labels.append(sty["label"])

        leg = ax_leg.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.0, 0.38),
            frameon=False,
            fontsize=7.5,
            handletextpad=0.5,
            borderaxespad=0.0,
            labelspacing=0.85,
        )
        for t in leg.get_texts():
            t.set_color("#222222")

        ax_leg.text(
            0.0,
            0.02,
            "Circle size equal.\nFill depth = contribution.\nLabels = top pollutants.",
            fontsize=6.5,
            color="#777777",
            transform=ax_leg.transAxes,
            va="bottom",
            linespacing=1.35,
        )

        fig.suptitle(
            "Exposome priority matrix (top 50% cumulative risk)",
            fontsize=12,
            fontweight="bold",
            color="#111111",
            y=0.97,
        )

        out_png = OUT_DIR / f"1_Matrix_EqualSize_ColorByPct_{name}.png"
        out_pdf = OUT_DIR / f"1_Matrix_EqualSize_ColorByPct_{name}.pdf"
        fig.savefig(out_png, dpi=450, bbox_inches="tight", facecolor="white")
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved: {out_png.name}")


    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    print(">>> aggregate ...")
    df_raw = pd.read_csv(INPUT_FILE)
    df_summary, disease_order, category_order = aggregate(df_raw)
    df_summary.to_csv(OUT_DIR / f"1_Aggregated_Matrix_Data_{name}.csv", index=False)

    print(">>> plot Nature-style matrix ...")
    plot_matrix(df_summary, disease_order, category_order)

    (OUT_DIR / f"README_{name}.txt").write_text(
        "\n".join(
            [
                "# code_7_ur_update",
                "- Equal-size circles; fill depth = contribution (%)",
                "- Bubble text = pollutant names (no %)",
                "- Edge colour = primary sector (full names in legend)",
                "- Nature-like sequential blue colormap",
            ]
        ),
        encoding="utf-8",
    )
    print(f"done → {OUT_DIR}")

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
import os
df = pd.read_csv(os.path.join(BASE_DIR,'code_7_ur_results/1_statistic_data_with_info.csv'))
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans', 'Liberation Sans']
category_col = 'sub category_en'  
exposure_col = 'Exposure'         
disease_col = 'Label'             
sig_col = 'Significance_raw_Rural'      
effect_col = 'LogRR_Rural'              

df['Plot_Value'] = np.where(
    df[sig_col].astype(str).str.contains('Not', case=False, na=True), 
    np.nan, 
    df[effect_col]
)

max_val = df['Plot_Value'].abs().quantile(0.90)
if np.isnan(max_val) or max_val == 0: max_val = 1.0 
df['Plot_Value'] = df['Plot_Value'].clip(-max_val, max_val)
df = df.sort_values(by='sub category_en',ascending=True)

exp_categories = df[category_col].dropna().unique()
diseases = df[disease_col].dropna().unique()
diseases.sort()
dict_exp = {cat: df[df[category_col] == cat][exposure_col].dropna().unique() for cat in exp_categories}
exp_counts = [len(dict_exp[c]) for c in exp_categories]
total_unique_exposures = sum(exp_counts)

# Y轴：为暴露物质大类分配颜色
y_colors = sns.color_palette("Set2", len(exp_categories))
dict_cat_color_y = dict(zip(exp_categories, y_colors))

# X轴：为疾病分配颜色 (如果你有疾病的大类列，可以把它替换为大类)
x_colors = sns.color_palette("husl", len(diseases))
dict_cat_color_x = dict(zip(diseases, x_colors))

# 4. 调整画布
fig_height = min(6, total_unique_exposures * 0.05) 
fig_width = 3

fig, axes = plt.subplots(
    nrows=len(exp_categories), 
    ncols=1, 
    figsize=(fig_width, fig_height),
    gridspec_kw={'height_ratios': exp_counts, 'hspace': 0.015} 
)
if len(exp_categories) == 1: axes = [axes]

# ==========================================
# 【核心修改 1：顶级深蓝与深红 Colorbar】
# ==========================================
# 提取类似原图中的深蓝(#3062A5)和深红(#D0332B)色值
custom_colors = ["#3062A5", "white", "#D0332B"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_RdBu", custom_colors)
cmap.set_bad(color='#e0e0e0') # 保持高级灰背景

norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)

# 5. 逐个区块绘制热图
for i, exp_cat in enumerate(exp_categories):
    ax = axes[i]
    
    subset = df[df[category_col] == exp_cat]
    pivot_df = subset.pivot(index=exposure_col, columns=disease_col, values='Plot_Value')
    pivot_df = pivot_df.reindex(index=dict_exp[exp_cat], columns=diseases)
    
    sns.heatmap(
        pivot_df, ax=ax, cmap=cmap, norm=norm, cbar=False,
        linewidths=0,       
        yticklabels=False,  
        xticklabels=(i == len(exp_categories) - 1),
        rasterized=True 
    )
    ax.get_xaxis().set_label_text('') 
   
    # 利用 ax.transAxes (0~1的相对坐标) 在热图外部画长方形
    # x=-0.04 (向左偏移), y=0 (从底部开始), width=0.03, height=1 (贯穿整个子图高度)
    rect_y = Rectangle((-0.04, 0), 0.03, 1, transform=ax.transAxes, 
                       color=dict_cat_color_y[exp_cat], clip_on=False)
    ax.add_patch(rect_y)
    
    # 设置大类标签文字，并使用 labelpad 将文字往左推，让出色带的空间
    ax.set_ylabel(exp_cat, fontsize=7, rotation=0, ha='right', va='center', labelpad=12)
    
   
    if i == len(exp_categories) - 1:
        # X 轴美化：使用 pad 将文字往下推，让出横向色带的空间
        ax.tick_params(axis='x', rotation=90, labelsize=7, pad=6) 
        ax.get_xaxis().set_label_text('') 
        # 混合坐标系：X 轴用数据坐标 (精准对齐每一列)，Y 轴用相对坐标 (往下平移)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for j, disease in enumerate(diseases):
            # x=j (当前列位置), y=-0.06 (向下偏移), width=1, height=0.04
            rect_x = Rectangle((j, -0.06), 1, 0.04, transform=trans, 
                               color=dict_cat_color_x[disease], clip_on=False)
            ax.add_patch(rect_x)

# 6. 全局排版优化
cbar_ax = fig.add_axes([0.91, 0.35, 0.02, 0.3]) # 让 colorbar 变修长
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Adjusted LogRR\n(Significant Only)', fontweight='bold', fontsize=7)
cbar.outline.set_visible(False) 
plt.subplots_adjust(left=0.35, right=0.88, top=0.88, bottom=0.15)

# 导出高分辨率图片
plt.savefig(os.path.join(OUT_DIR, 'plot1_Phenome_Exposome_Atlas_Rural.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
import os
df = pd.read_csv(os.path.join(BASE_DIR,'code_7_ur_results/1_statistic_data_with_info.csv'))
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans', 'Liberation Sans']
category_col = 'sub category_en'  
exposure_col = 'Exposure'         
disease_col = 'Label'             
sig_col = 'Significance_raw_Urban'      
effect_col = 'LogRR_Urban'              

df['Plot_Value'] = np.where(
    df[sig_col].astype(str).str.contains('Not', case=False, na=True), 
    np.nan, 
    df[effect_col]
)

max_val = df['Plot_Value'].abs().quantile(0.90)
if np.isnan(max_val) or max_val == 0: max_val = 1.0 
df['Plot_Value'] = df['Plot_Value'].clip(-max_val, max_val)
df = df.sort_values(by='sub category_en',ascending=True)

exp_categories = df[category_col].dropna().unique()
diseases = df[disease_col].dropna().unique()
diseases.sort()
dict_exp = {cat: df[df[category_col] == cat][exposure_col].dropna().unique() for cat in exp_categories}
exp_counts = [len(dict_exp[c]) for c in exp_categories]
total_unique_exposures = sum(exp_counts)

# Y轴：为暴露物质大类分配颜色
y_colors = sns.color_palette("Set2", len(exp_categories))
dict_cat_color_y = dict(zip(exp_categories, y_colors))

# X轴：为疾病分配颜色 (如果你有疾病的大类列，可以把它替换为大类)
x_colors = sns.color_palette("husl", len(diseases))
dict_cat_color_x = dict(zip(diseases, x_colors))

# 4. 调整画布
fig_height = min(6, total_unique_exposures * 0.05) 
fig_width = 3

fig, axes = plt.subplots(
    nrows=len(exp_categories), 
    ncols=1, 
    figsize=(fig_width, fig_height),
    gridspec_kw={'height_ratios': exp_counts, 'hspace': 0.015} 
)
if len(exp_categories) == 1: axes = [axes]

# ==========================================
# 【核心修改 1：顶级深蓝与深红 Colorbar】
# ==========================================
# 提取类似原图中的深蓝(#3062A5)和深红(#D0332B)色值
custom_colors = ["#3062A5", "white", "#D0332B"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_RdBu", custom_colors)
cmap.set_bad(color='#e0e0e0') # 保持高级灰背景

norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)

# 5. 逐个区块绘制热图
for i, exp_cat in enumerate(exp_categories):
    ax = axes[i]
    
    subset = df[df[category_col] == exp_cat]
    pivot_df = subset.pivot(index=exposure_col, columns=disease_col, values='Plot_Value')
    pivot_df = pivot_df.reindex(index=dict_exp[exp_cat], columns=diseases)
    
    sns.heatmap(
        pivot_df, ax=ax, cmap=cmap, norm=norm, cbar=False,
        linewidths=0,       
        yticklabels=False,  
        xticklabels=(i == len(exp_categories) - 1),
        rasterized=True 
    )
    ax.get_xaxis().set_label_text('') 
   
    # 利用 ax.transAxes (0~1的相对坐标) 在热图外部画长方形
    # x=-0.04 (向左偏移), y=0 (从底部开始), width=0.03, height=1 (贯穿整个子图高度)
    rect_y = Rectangle((-0.04, 0), 0.03, 1, transform=ax.transAxes, 
                       color=dict_cat_color_y[exp_cat], clip_on=False)
    ax.add_patch(rect_y)
    
    # 设置大类标签文字，并使用 labelpad 将文字往左推，让出色带的空间
    ax.set_ylabel(exp_cat, fontsize=7, rotation=0, ha='right', va='center', labelpad=12)
    
   
    if i == len(exp_categories) - 1:
        # X 轴美化：使用 pad 将文字往下推，让出横向色带的空间
        ax.tick_params(axis='x', rotation=90, labelsize=7, pad=6) 
        ax.get_xaxis().set_label_text('') 
        # 混合坐标系：X 轴用数据坐标 (精准对齐每一列)，Y 轴用相对坐标 (往下平移)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for j, disease in enumerate(diseases):
            # x=j (当前列位置), y=-0.06 (向下偏移), width=1, height=0.04
            rect_x = Rectangle((j, -0.06), 1, 0.04, transform=trans, 
                               color=dict_cat_color_x[disease], clip_on=False)
            ax.add_patch(rect_x)

# 6. 全局排版优化
cbar_ax = fig.add_axes([0.91, 0.35, 0.02, 0.3]) # 让 colorbar 变修长
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Adjusted LogRR\n(Significant Only)', fontweight='bold', fontsize=7)
cbar.outline.set_visible(False) 
plt.subplots_adjust(left=0.35, right=0.88, top=0.88, bottom=0.15)

# 导出高分辨率图片
plt.savefig(os.path.join(OUT_DIR, 'plot1_Phenome_Exposome_Atlas_Urban.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
# %%
