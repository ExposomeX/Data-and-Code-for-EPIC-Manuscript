#%%
import pandas as pd
import os
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CODE2_DIR = os.path.join(BASE_PATH, 'code_2_results')
CODE3_DIR = os.path.join(BASE_PATH, 'code_3_results')
gdp_data = pd.read_csv(os.path.join(CODE2_DIR, 'grid_level_gdp.csv'))
gdp_data.groupby('time')['GDP'].sum().plot(marker='o', title='Total GDP over Time')

SSPS = ['ssp1','ssp2','ssp3']
for ssp in SSPS:
    gdp_fut_data = pd.read_csv(os.path.join(CODE3_DIR, f'grid_level_gdp_{ssp}.csv'))
    gdp_fut_data.rename(columns={'GDP':f'GDP_{ssp}'}, inplace=True)
    gdp_fut_data.groupby('time')[f'GDP_{ssp}'].sum().plot(marker='o',legend=True, title=f'Total GDP over Time - {ssp}')
    
# %%
import pandas as pd
pop_data = pd.read_csv(os.path.join(CODE2_DIR, 'grid_level_population.csv'))
pop_data.groupby('time')['population'].sum().plot(marker='o', title='Total Population over Time')
SSPS = ['ssp126','ssp245','ssp370']
for ssp in SSPS:
    pop_fut_data = pd.read_csv(os.path.join(CODE3_DIR, f'grid_level_population_{ssp}.csv'))
    pop_fut_data.rename(columns={'population':f'population_{ssp}'}, inplace=True)
    pop_fut_data.groupby('time')[f'population_{ssp}'].sum().plot(marker='o',legend=True, title=f'Total Population over Time - {ssp}')
# %%import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. Load historical baseline data ----------------------
proxy_hist = pd.read_csv(os.path.join(CODE2_DIR, 'grid_level_proxy_6_pols.csv'))
# Extract all pollutant-emission columns.
emi_cols = [col for col in proxy_hist.columns if "emission" in col]
# Aggregate historical data by year.
hist_total = proxy_hist.groupby("time")[emi_cols].sum()

# ---------------------- 2. Load and store future data for all SSP scenarios ----------------------
SSPS = ["ssp126", "ssp245", "ssp370"]
fut_dict = {}
for ssp in SSPS:
    df = pd.read_csv(os.path.join(CODE3_DIR, f'grid_level_proxy_6_pols_fut_{ssp}.csv'))
    df.columns = df.columns.str.replace('.0', '', regex=False)
    df_sum = df.groupby("time")[emi_cols].sum()
    fut_dict[ssp] = df_sum

# ---------------------- 3. Create a panel with one subplot per pollutant ----------------------
n_pollutant = len(emi_cols)
# Arrange automatically, with up to three subplots per row.
n_row = int(np.ceil(n_pollutant / 5))
n_col = 5
fig, axes = plt.subplots(n_row, n_col, figsize=(16, 5 * n_row), squeeze=False)
axes = axes.flatten()  # Flatten the axes array for iteration.

# Plot each pollutant.
for idx, pollute_name in enumerate(emi_cols):
    ax = axes[idx]
    # Plot the historical baseline as a solid black line.
    ax.plot(hist_total.index, hist_total[pollute_name],
            marker="o", c="black", label="Historical 2017-2023")
    
    # Plot all SSP scenarios.
    color_map = {"ssp126": "#2ca02c", "ssp245": "#ff7f0e", "ssp370": "#d62728"}
    for ssp in SSPS:
        df_fut = fut_dict[ssp]
        ax.plot(df_fut.index, df_fut[pollute_name],
                marker="o", c=color_map[ssp], label=ssp)
    
    # Format the individual subplot.
    ax.set_title(f"Total Emissions: {pollute_name}", fontsize=12)
    ax.set_xlabel("Year (time)")
    ax.set_ylabel("Total Emission")
    if idx == 0:  # Show the legend only in the first subplot.
        ax.legend(loc="best")
    ax.grid(alpha=0.3)

# Hide unused subplots when there are fewer pollutants than grid slots.
for empty_idx in range(n_pollutant, len(axes)):
    axes[empty_idx].axis("off")

# Set the overall figure title and layout.
#fig.suptitle("Historical & Future Emissions Under Different SSP Scenarios", fontsize=16, y=0.98)
plt.tight_layout()
os.makedirs(CODE3_DIR, exist_ok=True)
plt.savefig(os.path.join(CODE3_DIR, 'Total_Emissions_SSP_Comparison.png'), dpi=300, bbox_inches='tight')
# %%
