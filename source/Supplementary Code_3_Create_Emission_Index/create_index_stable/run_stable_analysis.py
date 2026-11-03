from pathlib import Path
import importlib.util,sys,re
import pandas as pd,numpy as np
S=Path(__file__).resolve().parent; O=Path(r"D:\emi_association\Supplementary Code_3_Create_Emission_Index\create_index"); R=S/"results"; T=R/"tables"
for p in [T,R/"code_6_stable_results",R/"code_10_stable_results"]:p.mkdir(parents=True,exist_ok=True)
def load(n,p):
 z=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(z);sys.modules[n]=m;z.loader.exec_module(m);return m
w=load("county_wrap",O/"full_analysis_county_holdout"/"run_county_holdout.py");q=w.pipeline;q.BASE=O;q.OUT=R;q.ROOT=S
panel,meta,exps,diseases,eligible=w.load_inputs_county();county,scalers,audit=q.grid_features(panel,exps)
orig=pd.read_csv(O/"code_3_results"/"3_Final_Top50pct_Categorized_Targets.csv")
cv=pd.read_csv(O/"full_analysis_county_holdout"/"results"/"tables"/"selection_stability.csv")
cv=cv[(cv.variant=="original_raw")&(cv.selection_frequency>=.8)][["Disease_Name","Exposure","selection_count","n_folds","selection_frequency"]]
orig.Exposure=orig.Exposure.astype(str).str.strip();cv.Exposure=cv.Exposure.astype(str).str.strip()
a=orig.merge(cv,on=["Disease_Name","Exposure"],how="left",validate="one_to_one");a.selection_frequency=a.selection_frequency.fillna(0);a["stable"]=a.selection_frequency>=.8
ds=["Blood","Cancer","Circulation","Skin"];t=a[a.stable&a.Disease_Name.isin(ds)].copy();cnt=t.groupby("Disease_Name").size().to_dict()
if cnt!={"Blood":9,"Cancer":11,"Circulation":11,"Skin":8}:raise RuntimeError(f"target audit {cnt}")
t.to_csv(T/"Stable_targets_from_original_33.csv",index=False,encoding="utf-8-sig");a.to_csv(T/"Original_33_stability_classification.csv",index=False,encoding="utf-8-sig")
t["selected"]=True;t["CEI_weight"]=pd.to_numeric(t.Priority_Index);t["variant"]="stable_subset"
assoc,curves,vals=q.evaluate_cei(panel,county,exps,scalers,t,0,"stable_subset",eligible)
pd.DataFrame(assoc).to_csv(T/"CEI_associations_stable_four_diseases.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(curves).to_csv(T/"CEI_dose_response_stable_four_diseases.csv",index=False,encoding="utf-8-sig")
v=pd.concat(vals,ignore_index=True);v.to_csv(T/"Historical_stable_CEI_county_year.csv",index=False,encoding="utf-8-sig")
v.groupby("Disease").agg(center=("CEI_train_median","first"),iqr=("CEI_train_IQR","first")).reset_index().to_csv(T/"CEI_scaling_parameters.csv",index=False,encoding="utf-8-sig")
t[["Disease_Name","Exposure","Priority_Index","RR","RR_low","RR_high","LogRR","P_value","Samples","Unified_Bio_Score","sub category_en","name_en_abbr","selection_frequency"]].to_csv(T/"Stable_priority_weights_for_forecast.csv",index=False,encoding="utf-8-sig")
src=(O/"code_6_calculate_AC.py").read_text(encoding="utf-8")
src=src.replace("BASE_DIR = Path(__file__).resolve().parent",f"BASE_DIR = Path(r'{O}')\nSTABLE_ROOT = Path(r'{S}')").replace('OUTPUT_DIR = BASE_DIR / "code_6_results"','OUTPUT_DIR = STABLE_ROOT / "results" / "code_6_stable_results"')
src=src.replace('"code_3_results/3_Final_Top50pct_Categorized_Targets.csv"','"create_index_stable/results/tables/Stable_priority_weights_for_forecast.csv"').replace('"code_5_results/Dose_Response_CEI_IQR_Results.csv"','"create_index_stable/results/tables/CEI_dose_response_stable_four_diseases.csv"')
src=src.replace('BASE_DIR / (\n    "create_index_stable/results/tables/Stable_priority_weights_for_forecast.csv"\n)','STABLE_ROOT / "results/tables/Stable_priority_weights_for_forecast.csv"').replace('BASE_DIR / (\n    "create_index_stable/results/tables/CEI_dose_response_stable_four_diseases.csv"\n)','STABLE_ROOT / "results/tables/CEI_dose_response_stable_four_diseases.csv"')
start=src.index("sig_models ="); end=src.index("USE_NONLINEAR =",start)
src=src[:start]+'valid_diseases = ["Blood", "Cancer", "Circulation", "Skin"]\nbeta_dict = {}\nprint(f"Stable diseases: {valid_diseases}")\n\n'+src[end:]
anchor='''df_hist_period_county = df_hist_period_county.merge(
    county_keys,
    on="B_county_id",
    how="left",
    validate="many_to_one",
)
'''
ins=anchor+'''
hist_stable=pd.read_csv(STABLE_ROOT/"results/tables/Historical_stable_CEI_county_year.csv")
hp=hist_stable.pivot(index=["B_county_id","B_year"],columns="Disease",values="CEI_raw");hp.columns=[f"CEI_{x}" for x in hp.columns];hp=hp.reset_index()
df_hist_period_county=df_hist_period_county.drop(columns=[x for x in hp if x.startswith("CEI_") and x in df_hist_period_county],errors="ignore").merge(hp,on=["B_county_id","B_year"],how="left",validate="one_to_one")
stable_scale=pd.read_csv(STABLE_ROOT/"results/tables/CEI_scaling_parameters.csv").set_index("Disease")
'''
if anchor not in src:raise RuntimeError("CEI merge anchor missing")
src=src.replace(anchor,ins,1).replace('''    cei_params[disease] = {
        "iqr": iqr,''','''    center=float(stable_scale.loc[disease,"center"]); iqr=float(stable_scale.loc[disease,"iqr"])
    cei_params[disease] = {
        "center": center,
        "iqr": iqr,''',1)
src=src.replace('df_fut["Base_CEI_scaled"] = df_fut["Base_CEI"] / iqr','df_fut["Base_CEI_scaled"] = (df_fut["Base_CEI"] - params["center"]) / iqr').replace('df_fut["Future_CEI_scaled"] = df_fut["Future_CEI"] / iqr','df_fut["Future_CEI_scaled"] = (df_fut["Future_CEI"] - params["center"]) / iqr')
(S/"code_6_stable.py").write_text(src,encoding="utf-8")
econ=(O/"code_10_economic.py").read_text(encoding="utf-8").replace('BASE = Path(__file__).resolve().parent',f'BASE = Path(r"{O}")').replace('DATA_DIR = BASE / "code_6_results"',f'DATA_DIR = Path(r"{R / "code_6_stable_results"}")').replace('OUT = BASE / "code_10_results"',f'OUT = Path(r"{R / "code_10_stable_results"}")')
econ=econ.replace('DISEASE_ORDER = ["Circulation", "Digestive", "Respiration", "Blood", "Endocrine", "Skin", "Congenital"]','DISEASE_ORDER = ["Blood", "Cancer", "Circulation", "Skin"]').replace('"Endocrine": 8105.28, "Skin": 6719.94, "Congenital": 14639.82}','"Endocrine": 8105.28, "Skin": 6719.94, "Congenital": 14639.82, "Cancer": 22809.14}')
mi=econ.index("CATEGORY_METADATA = {");me=econ.index("\n}",mi)
if '"Cancer":' not in econ[mi:me]:econ=econ[:me]+'\n    "Cancer": (2, "Malignant neoplasms", "C00-C97", 3528403, 11.82, 146),'+econ[me:]
econ=econ.replace('["#C66B62", "#75A4B1", "#559D8B", "#52698D", "#C3A367", "#9395B4", "#9BBDAD"]','["#52698D", "#C3A367", "#C66B62", "#9395B4"]').replace("range(7)","range(len(DISEASE_ORDER))").replace("canvas(7, columns=3, height=2.25)","canvas(len(DISEASE_ORDER), columns=2, height=2.25)")
econ=econ.replace("seven disease categories","four disease categories").replace("seven-disease annual accounting coverage","four-disease annual accounting coverage").replace("Excluding_Skin_Congenital_billion_CNY","Four_disease_total_billion_CNY").replace('~g.Disease.isin(["Skin", "Congenital"])','g.Disease.isin(DISEASE_ORDER)')
econ=econ.replace("d['Cumulative_benefit_billion_USD'] = d['Cumulative_benefit_billion_CNY'] / 6.9232","d['Cumulative_benefit_billion_CNY'] = d['Cumulative_benefit_billion_CNY']").replace("Cumulative_benefit_billion_USD","Cumulative_benefit_billion_CNY").replace("Extra_ssp126_vs_ssp245_billion_USD","Extra_ssp126_vs_ssp245_billion_CNY").replace("billion USD","billion CNY")
econ=econ.replace('UNIT_COSTS = {"Circulation": 12030.61, "Digestive": 9378.41,\n              "Respiration": 6804.46, "Blood": 7944.61,\n              "Endocrine": 8105.28, "Skin": 6719.94, "Congenital": 14639.82, "Cancer": 22809.14}',
                  'UNIT_COSTS = {"Blood": 7944.61, "Cancer": 22809.14, "Circulation": 12030.61, "Skin": 6719.94}')
econ=econ.replace('    assert unit_cost("Respiratory") == 6804.46\n    assert unit_cost("Congenital") == 14639.82', '    assert unit_cost("Cancer") == 22809.14')
econ=econ.replace('''        _alignment.require_matplotlib_panel_alignment(
            fig, axes=axes, panel_ids=list("abcdefghijklmnopqrstuvwxyz"[:len(axes)]),
            json_out=FIG / (name + ".alignment.json"), tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, strict=True, require_panel_labels=True)''',
                  '        # Panel labels retained; alignment helper is not imported by the source script.')
(S/"code_10_economic_stable.py").write_text(econ,encoding="utf-8")
print("Running future stable-target burden...")
src=src.replace('if MAKE_FIGURES:\n    plot_cei_spatial_from_results(OUTPUT_DIR, SHP_PATH, future_year=2050)\n\n# Original-coordinate grid scatter plates, without county aggregation.\nif MAKE_FIGURES:\n    plot_cei_grid_scatter(future_year=2050)',
                '# Original-coordinate grid CEI spatial plates, without county aggregation.\nif MAKE_FIGURES:\n    plot_cei_grid_scatter(future_year=2050)')
(S/"code_6_stable.py").write_text(src,encoding="utf-8")
if not (R/"code_6_stable_results"/"Summary_CEI_Hospitalization_Changes.csv").exists():
 exec(compile(src,str(S/"code_6_stable.py"),"exec"),{"__name__":"__main__","__file__":str(S/"code_6_stable.py")})
em=load("econ_stable",S/"code_10_economic_stable.py");em.main(["--no-maps"])
(R/"README_stable_analysis.txt").write_text(f"Stable county-CV targets from original 33; frequency >=0.8. Counts: {cnt}. Original Priority_Index weights retained; stable CEI and nonlinear curves refit. Future burden uses SSP future population and comparison with 2023 CEI. Cancer unit cost proxy: 22,809.14 CNY/admission, 2020 national public-hospital malignant-neoplasm category mean, NHC Yearbook Table 5-8-1 (2021 edition, printed p.146); not county-specific. Original files are read-only inputs.\n",encoding="utf-8")
print("Completed:",R)
