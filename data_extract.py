# --------------------------------
# 0. IMPORT LIBRARIES
# --------------------------------
import pandas as pd
import numpy as np
from io import BytesIO
import requests
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# # =====================================================
# # Helper function to load XPT files
# # =====================================================
# def load_xpt(url, columns=None):
#     try:
#         r = requests.get(url)
#         r.raise_for_status()
#         df = pd.read_sas(BytesIO(r.content), format="xport")
#         if columns:
#             df = df[[c for c in columns if c in df.columns]]
#         return df
#     except Exception as e:
#         print(f"Warning: Could not load {url}: {e}")
#         return pd.DataFrame()

# # =====================================================
# # Cap PA days between 0–7
# # =====================================================
# def cap_range(series, max_val=7):
#     return series.clip(lower=0, upper=max_val)

# # =====================================================
# # PA category
# # =====================================================
# def pa_category(x):
#     if pd.isna(x):
#         return np.nan
#     elif x <= 2:
#         return "Low"
#     elif x <= 4:
#         return "Moderate"
#     return "High"


# # =====================================================
# # Cycle → File → Variables
# # =====================================================

# CYCLE_FILE_VARS = {
#     "2001": {
#         "DEMO_B": ["SEQN","RIAGENDR","RIDAGEYR","RIDRETH1","INDFMPIR","DMDEDUC2","DMDEDUC3","DMDHHSIZ","WTMEC2YR"],
#         "BMX_B": ["SEQN","BMXBMI"],
#         "SMQ_B": ["SEQN","SMQ020","SMQ040"],
#         "ALQ_B": ["SEQN","ALD100"],
#         "DBQ_B": ["SEQN","DBD229"],
#         "DRXTOT_B": ["SEQN","DRXTCALC"],
#         "DIQ_B": ["SEQN","DIQ010"],
#         "DEQ_B": ["SEQN","DED120","DED125","DEQ034A","DEQ034C","DEQ034D"],
#         "PAQ_B": ["SEQN","PAD020","PAQ050Q","PAQ050U"],
#         "L40_B": ["SEQN","LBXSCA"],
#         "VID_B": ["SEQN","LBDVIDMS"]
#     },
#     "2003": {
#         "DEMO_C": ["SEQN","RIAGENDR","RIDAGEYR","RIDRETH1","INDFMPIR","DMDEDUC2","DMDEDUC3","DMDHHSIZ","WTMEC2YR"],
#         "BMX_C": ["SEQN","BMXBMI"],
#         "SMQ_C": ["SEQN","SMQ020","SMQ040"],
#         "ALQ_C": ["SEQN","ALQ101","ALQ120Q","ALQ120U"],
#         "DBQ_C": ["SEQN","DBQ229"],
#         "DR1TOT_C": ["SEQN","DR1TCALC"],
#         "DR2TOT_C": ["SEQN","DR2TCALC"],
#         "DIQ_C": ["SEQN","DIQ010"],
#         "DEQ_C": ["SEQN","DED120","DED125","DEQ034A","DEQ034C","DEQ034D"],
#         "PAQ_C": ["SEQN","PAD020","PAQ050Q","PAQ050U"],
#         "L40_C": ["SEQN","LBXSCA"],
#         "VID_C": ["SEQN","LBDVIDMS"]
#     },
#     "2005": {
#         "DEMO_D": ["SEQN","RIAGENDR","RIDAGEYR","RIDRETH1","INDFMPIR","DMDEDUC2","DMDEDUC3","DMDHHSIZ","WTMEC2YR"],
#         "BMX_D": ["SEQN","BMXBMI"],
#         "SMQ_D": ["SEQN","SMQ020","SMQ040"],
#         "ALQ_D": ["SEQN","ALQ101","ALQ120Q","ALQ120U"],
#         "DBQ_D": ["SEQN","DBQ229"],
#         "DR1TOT_D": ["SEQN","DR1TCALC"],
#         "DR2TOT_D": ["SEQN","DR2TCALC"],
#         "DIQ_D": ["SEQN","DIQ010"],
#         "DEQ_D": ["SEQN","DED120","DED125","DEQ034A","DEQ034C","DEQ034D"],
#         "PAQ_D": ["SEQN","PAD020","PAQ050Q","PAQ050U"],
#         "BIOPRO_D": ["SEQN","LBXSCA"],
#         "VID_D": ["SEQN","LBDVIDMS"]
#     },
#     "2007": {
#         "DEMO_E": ["SEQN","RIAGENDR","RIDAGEYR","RIDRETH1","INDFMPIR","DMDEDUC2","DMDEDUC3","DMDHHSIZ","WTMEC2YR"],
#         "BMX_E": ["SEQN","BMXBMI"],
#         "SMQ_E": ["SEQN","SMQ020","SMQ040"],
#         "ALQ_E": ["SEQN","ALQ101","ALQ120Q","ALQ120U"],
#         "DBQ_E": ["SEQN","DBQ229"],
#         "DR1TOT_E": ["SEQN","DR1TCALC"],
#         "DR2TOT_E": ["SEQN","DR2TCALC"],
#         "DIQ_E": ["SEQN","DIQ010"],
#         # "DEQ_E": ["SEQN","DED120","DED125","DEQ034A","DEQ034C","DEQ034D"], # File Not Present
#         "PAQ_E": ["SEQN","PAQ640","PAQ655","PAQ670"],
#         "BIOPRO_E": ["SEQN","LBXSCA"],
#         "VID_E": ["SEQN","LBXVIDMS"]
#     },
#     "2009": {
#         "DEMO_F": ["SEQN","RIAGENDR","RIDAGEYR","RIDRETH1","INDFMPIR","DMDEDUC2","DMDEDUC3","DMDHHSIZ","WTMEC2YR"],
#         "BMX_F": ["SEQN","BMXBMI"],
#         "SMQ_F": ["SEQN","SMQ020","SMQ040"],
#         "ALQ_F": ["SEQN","ALQ101","ALQ120Q","ALQ120U"],
#         "DBQ_F": ["SEQN","DBQ229"],
#         "DR1TOT_F": ["SEQN","DR1TCALC"],
#         "DR2TOT_F": ["SEQN","DR2TCALC"],
#         "DIQ_F": ["SEQN","DIQ010"],
#         "DEQ_F": ["SEQN","DED120","DED125","DEQ034A","DEQ034C","DEQ034D"],
#         "PAQ_F": ["SEQN","PAQ640","PAQ655","PAQ670"],
#         "BIOPRO_F": ["SEQN","LBXSCA"],
#         "VID_F": ["SEQN","LBXVIDMS"]
#     },
#     "2011": {
#         "DEMO_G": ["SEQN","RIAGENDR","RIDAGEYR","RIDRETH1","INDFMPIR","DMDEDUC2","DMDEDUC3","DMDHHSIZ","WTMEC2YR"],
#         "BMX_G": ["SEQN","BMXBMI"],
#         "SMQ_G": ["SEQN","SMQ020","SMQ040"],
#         "ALQ_G": ["SEQN","ALQ101","ALQ120Q","ALQ120U"],
#         "DBQ_G": ["SEQN","DBQ229"],
#         "DR1TOT_G": ["SEQN","DR1TCALC"],
#         "DR2TOT_G": ["SEQN","DR2TCALC"],
#         "DIQ_G": ["SEQN","DIQ010"],
#         "DEQ_G": ["SEQN","DED120","DED125","DEQ034A","DEQ034C","DEQ034D"],
#         "PAQ_G": ["SEQN","PAQ640","PAQ655","PAQ670"],
#         "BIOPRO_G": ["SEQN","LBXSCA"],
#         "VID_G": ["SEQN","LBXVIDMS"]
#     },
#     "2013": {
#         "DEMO_H": ["SEQN","RIAGENDR","RIDAGEYR","RIDRETH1","INDFMPIR","DMDEDUC2","DMDEDUC3","DMDHHSIZ","WTMEC2YR"],
#         "BMX_H": ["SEQN","BMXBMI"],
#         "SMQ_H": ["SEQN","SMQ020","SMQ040"],
#         "ALQ_H": ["SEQN","ALQ101","ALQ120Q","ALQ120U"],
#         "DBQ_H": ["SEQN","DBQ229"],
#         "DR1TOT_H": ["SEQN","DR1TCALC"],
#         "DR2TOT_H": ["SEQN","DR2TCALC"],
#         "DIQ_H": ["SEQN","DIQ010"],
#         "DEQ_H": ["SEQN","DED120","DED125","DEQ034A","DEQ034C","DEQ034D"],
#         "PAQ_H": ["SEQN","PAQ640","PAQ655","PAQ670"],
#         "BIOPRO_H": ["SEQN","LBXSCA"],
#         "VID_H": ["SEQN","LBXVIDMS"]
#     },
#     "2015": {
#         "DEMO_I": ["SEQN","RIAGENDR","RIDAGEYR","RIDRETH1","INDFMPIR","DMDEDUC2","DMDEDUC3","DMDHHSIZ","WTMEC2YR"],
#         "BMX_I": ["SEQN","BMXBMI"],
#         "SMQ_I": ["SEQN","SMQ020","SMQ040"],
#         "ALQ_I": ["SEQN","ALQ101","ALQ120Q","ALQ120U"],
#         "DBQ_I": ["SEQN","DBQ229"],
#         "DR1TOT_I": ["SEQN","DR1TCALC"],
#         "DR2TOT_I": ["SEQN","DR2TCALC"],
#         "DIQ_I": ["SEQN","DIQ010"],
#         "DEQ_I": ["SEQN","DED120","DED125","DEQ034A","DEQ034C","DEQ034D"],
#         "PAQ_I": ["SEQN","PAQ640","PAQ655","PAQ670"],
#         "BIOPRO_I": ["SEQN","LBXSCA"],
#         "VID_I": ["SEQN","LBXVIDMS"]
#     },
#     "2017": { 
#         "DEMO_J": ["SEQN","RIAGENDR","RIDAGEYR","RIDRETH1","INDFMPIR","DMDEDUC2","DMDEDUC3","DMDHHSIZ","WTMEC2YR"],
#         "BMX_J": ["SEQN","BMXBMI"], 
#         "SMQ_J": ["SEQN","SMQ020","SMQ040"], 
#         "ALQ_J": ["SEQN","ALQ121"], 
#         "DBQ_J": ["SEQN","DBQ229"], 
#         "DR1TOT_J": ["SEQN","DR1TCALC"], 
#         "DR2TOT_J": ["SEQN","DR2TCALC"], 
#         "DIQ_J": ["SEQN","DIQ010"], 
#         "DEQ_J": ["SEQN","DED120","DED125","DEQ034A","DEQ034C","DEQ034D"], 
#         "PAQ_J": ["SEQN","PAQ640","PAQ655","PAQ670"], 
#         "BIOPRO_J": ["SEQN","LBXSCA"],
#         "VID_J": ["SEQN","LBXVIDMS"]
#     },
#     "2021": {
#         "DEMO_L": ["SEQN","RIAGENDR","RIDAGEYR","RIDRETH1","INDFMPIR","DMDEDUC2","DMDEDUC3","DMDHHSIZ","WTMEC2YR"],
#         "BMX_L": ["SEQN","BMXBMI"],
#         "SMQ_L": ["SEQN","SMQ020","SMQ040"],
#         "ALQ_L": ["SEQN","ALQ121"],
#         "DBQ_L": ["SEQN","PAD790Q","PAD790U","PAD810Q","PAD810U"],
#         "DR1TOT_L": ["SEQN","DR1TCALC"],
#         "DR2TOT_L": ["SEQN","DR2TCALC"],
#         "DIQ_L": ["SEQN","DIQ010"],
#         "DEQ_L": ["SEQN","DED120","DED125","DEQ034A","DEQ034C","DEQ034D"],
#         "PAQ_L": ["SEQN","PAD790Q","PAD790U","PAD810Q","PAD810U"],
#         "BIOPRO_L": ["SEQN","LBXSCA"],
#         "VID_L": ["SEQN","LBXVIDMS"]  
#     }
# }



# # =====================================================
# # Derived functions
# # =====================================================
# def age_cat(x):
#     if pd.isna(x): return np.nan
#     if x < 18: return "<18"
#     if x <= 44: return "18–44"
#     if x <= 65: return "45–65"
#     return ">65"

# def pir_cat(x):
#     if pd.isna(x): return np.nan
#     if x <= 1.3: return "Low"
#     if x <= 3.5: return "Middle"
#     return "High"

# def bmi_cat(x):
#     if pd.isna(x): return np.nan
#     if x < 18.5: return "Underweight"
#     if x < 25: return "Normal"
#     if x < 30: return "Overweight"
#     if x > 30: return "Obese"
#     else: return np.nan

# def sun_category(x):
#     if pd.isna(x): return np.nan
#     if x == 0: return "Never"
#     if x <= 14: return "Rare"
#     if x <= 480: return "Sometimes"
#     if x <= 1440: return "Frequent"
#     return "Regular"

# # =====================================================
# # Alcohol harmonization (leave as-is)
# # =====================================================
# def alcohol_harmonize(row, cycle):
#     alc_yesno = np.nan
#     drinks_per_year = np.nan

#     if cycle == "2001":
#         val = row.get("ALD100")
#         if pd.notna(val):
#             alc_yesno = "Yes" if val == 1 else ("No" if val == 2 else np.nan)

#     elif cycle in ["2003","2005","2007","2009","2011","2013","2015"]:
#         val = row.get("ALQ101")
#         alc_yesno = "Yes" if val == 1 else ("No" if val == 2 else np.nan)

#         freq = row.get("ALQ120Q")
#         unit = row.get("ALQ120U")
#         if pd.notna(freq) and pd.notna(unit):
#             if unit == 1:  # per week
#                 drinks_per_year = freq * 52
#             elif unit == 2:  # per month
#                 drinks_per_year = freq * 12
#             elif unit == 3:  # per year
#                 drinks_per_year = freq

#     elif cycle in ["2017","2021"]:
#         val = row.get("ALQ121")
#         alc_yesno = "Yes" if (pd.notna(val) and ( val > 0 and val <=10 )) else ("No" if val == 0 else np.nan)

#         code_to_drinks = {0:0,1:365,2:330,3:182,4:104,5:52,6:30,7:12,8:9,9:4,10:1}
#         if pd.notna(val):
#             drinks_per_year = code_to_drinks.get(int(val), np.nan)

#     return pd.Series([alc_yesno, drinks_per_year])

# # =====================================================
# # Education harmonization
# # =====================================================
# def harmonize_education(educ2, educ3, age):

#     if age < 20:
#         if pd.isna(educ3) or educ3 in [77,99]: return np.nan
#         if educ3 in [13, 14, 15]:
#             return "High school or above"
#         elif educ3 >= 4:
#             return "High school or below"
#         elif educ3 >= 0:
#             return "Low education"
#         else:
#             return np.nan
#     else:
#         if pd.isna(educ2) or educ2 in [77,99]: return np.nan
#         if educ2 in [4, 5]:
#             return "College"
#         elif educ2 == 3:
#             return "High school"
#         elif educ2 >= 1:
#             return "Less than high school"
#         else:
#             return np.nan

# # =====================================================
# # Sun helpers
# # =====================================================
# def clean_sun_minutes(series):
#     return series.replace({3333: np.nan, 7777: np.nan, 9999: np.nan})

# def map_sun_behavior(series, shade=False):
#     if shade:
#         return series.map({
#             1: "Always",
#             2: "Most",
#             3: "Sometimes",
#             4: "Rarely",
#             5: "Never",
#             6: "No outdoor exposure"
#         })
#     else:
#         return series.map({
#             1: "Always",
#             2: "Most",
#             3: "Sometimes",
#             4: "Rarely",
#             5: "Never"
#         })

# # =====================================================
# # Load and merge raw files per cycle
# # =====================================================
# def load_cycle(cycle):
#     base = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle}/DataFiles/"
#     dfs = {}

#     for file, vars_ in CYCLE_FILE_VARS[cycle].items():
#         url = f"{base}{file}.XPT"
#         df = load_xpt(url, vars_)
#         if not df.empty:
#             dfs[file] = df

#     if not dfs:
#         return pd.DataFrame()

#     out = next(iter(dfs.values()))
#     for df in list(dfs.values())[1:]:
#         out = out.merge(df, on="SEQN", how="outer")

#     return out


# # =====================================================
# # Harmonize one cycle
# # =====================================================
# def harmonize_cycle(cycle):
#     print(f"Processing {cycle}")
#     df = load_cycle(cycle)
#     if df.empty:
#         return df
#     h = pd.DataFrame({"SEQN": df["SEQN"]})
#     h["cycle"] = cycle

#     # MEC weight
#     h["MEC_weight"] = df.get("WTMEC2YR", np.nan)

#     # Demographics
#     h["gender"] = df.get("RIAGENDR").map({1:"Male",2:"Female"})
#     h["age_cat"] = df.get("RIDAGEYR").apply(age_cat)
#     h["race_ethnicity"] = df.get("RIDRETH1").map({
#         1:"Mexican American",2:"Other Hispanic",
#         3:"Non-Hispanic White",4:"Non-Hispanic Black",5:"Other"
#     }) if df.get("RIDRETH1") is not None else np.nan

#     h["PIR_cat"] = df.get("INDFMPIR").apply(pir_cat)

#     # Education
#     h["education"] = df.apply(lambda row: harmonize_education(row.get("DMDEDUC2"), row.get("DMDEDUC3"), row.get("RIDAGEYR")), axis=1)

#     # Household size
#     h["household_size"] = df.get("DMDHHSIZ")

#     # BMI
#     h["BMI_cat"] = df.get("BMXBMI").apply(bmi_cat) if "BMXBMI" in df.columns else np.nan

#     # Smoking
#     smq020 = df.get("SMQ020")
#     smq040 = df.get("SMQ040")
#     h["smoker_ever"] = smq020.map({1:"Yes",2:"No"}) if smq020 is not None else np.nan
#     h["smoker_current"] = smq040.map({1:"Every day",2:"Some days",3:"Not at all"}) if smq040 is not None else np.nan

#     # Alcohol
#     alc = df.apply(lambda row: alcohol_harmonize(row, cycle), axis=1)
#     h["alcohol_yesno"] = alc[0]
#     h["alcohol_drinks_per_year"] = alc[1]

#     # Milk intake
#     if "DBQ229" in df.columns:
#         milk = df["DBQ229"]
#     elif "DBD229" in df.columns:
#         milk = df["DBD229"]
#     else:
#         milk = pd.Series(np.nan, index=df.index)
#     h["milk_intake"] = milk.map({1:"Regular", 2:"Never", 3:"Sometimes"})

#     # Calcium intake
#     if "DRXTCALC" in df.columns:
#         h["calcium_intake"] = df["DRXTCALC"]
#     elif all(col in df.columns for col in ["DR1TCALC","DR2TCALC"]):
#         h["calcium_intake"] = df[["DR1TCALC","DR2TCALC"]].mean(axis=1, skipna=True)
#     else:
#         h["calcium_intake"] = np.nan

#     # Initialize total_PA_days
#     total_PA_days = pd.Series(np.nan, index=df.index)

#     if cycle in ["2001","2003","2005"]:
#         # Get variables
#         pad = df.get("PAD020", pd.Series(np.nan, index=df.index))
#         q = df.get("PAQ050Q", pd.Series(np.nan, index=df.index))
#         u = df.get("PAQ050U", pd.Series(np.nan, index=df.index))

#         # Replace special codes with NaN
#         pad = pad.replace({2:0, 3:0, 7:np.nan, 9:np.nan})  # 1=Yes, 2=No, 3=Unable
#         q = q.replace({77777:np.nan, 99999:np.nan})
#         u = u.replace({7:np.nan, 9:np.nan})

#         # Convert units to days
#         conv = q.copy()
#         conv[u==1] *= 1      # per day → keep as is
#         conv[u==2] *= 7      # per week → multiply by 7
#         conv[u==3] /= 4      # per month → divide by ~4 weeks
#         conv[u.isna()] = np.nan

#         # Combine with PAD020 contribution
#         total_PA_days = pad + conv
#         total_PA_days = total_PA_days.clip(upper=7)  # cap at 7

#     elif cycle in ["2007","2009","2011","2013","2015","2017"]:
#         # Get variables
#         pa1 = df.get("PAQ640", pd.Series(np.nan, index=df.index))
#         pa2 = df.get("PAQ655", pd.Series(np.nan, index=df.index))
#         pa3 = df.get("PAQ670", pd.Series(np.nan, index=df.index))

#         # Replace special codes with NaN
#         pa1 = pa1.replace({77:np.nan, 99:np.nan})
#         pa2 = pa2.replace({77:np.nan, 99:np.nan})
#         pa3 = pa3.replace({77:np.nan, 99:np.nan})

#         # Sum and cap
#         total_PA_days = pa1 + pa2 + pa3
#         total_PA_days = total_PA_days.clip(upper=7)

#     elif cycle == "2021":
#         # Get variables
#         mod = df.get("PAD790Q", pd.Series(np.nan, index=df.index)).copy()
#         vig = df.get("PAD810Q", pd.Series(np.nan, index=df.index)).copy()
#         u_mod = df.get("PAD790U", pd.Series("W", index=df.index))
#         u_vig = df.get("PAD810U", pd.Series("W", index=df.index))

#         # Replace special codes with NaN
#         mod.replace({7777:np.nan, 9999:np.nan}, inplace=True)
#         vig.replace({7777:np.nan, 9999:np.nan}, inplace=True)

#         # Convert units to days/week
#         for arr, unit in [(mod,u_mod),(vig,u_vig)]:
#             arr.loc[unit=="D"] *= 7
#             arr.loc[unit=="M"] /= 4
#             arr.loc[unit=="Y"] /= 52
#             arr[arr < 0] = np.nan

#         # Sum moderate + vigorous
#         total_PA_days = mod + vig
#         total_PA_days = total_PA_days.clip(upper=7)

#     # Assign to harmonized dict
#     h["total_PA_days"] = total_PA_days
#     h["PA_category"] = h["total_PA_days"].apply(pa_category)

#     # Sun exposure
#     ded120 = clean_sun_minutes(df.get("DED120", pd.Series(np.nan, index=df.index)))
#     ded125 = clean_sun_minutes(df.get("DED125", pd.Series(np.nan, index=df.index)))
#     h["sun_exposure"] = (ded120 + ded125).apply(sun_category)

#     # Sun protective behaviors
#     h["sun_shade"] = map_sun_behavior(df.get("DEQ034A", pd.Series(np.nan, index=df.index)), shade=True)
#     h["sun_shirt"] = map_sun_behavior(df.get("DEQ034C", pd.Series(np.nan, index=df.index)))
#     h["sun_sunscreen"] = map_sun_behavior(df.get("DEQ034D", pd.Series(np.nan, index=df.index)))


#     # Diabetes
#     h["diabetes"] = df.get("DIQ010").map({1:"Yes",2:"No",3:"Borderline"}) if "DIQ010" in df.columns else np.nan

#     # Lab markers
#     h["calcium_lab"] = df.get("LBXSCA")

#     if "LBXVIDMS" in df.columns:
#         h["vitamin_D"] = df["LBXVIDMS"]
#     elif "LBDVIDMS" in df.columns:
#         h["vitamin_D"] = df["LBDVIDMS"]
#     else:
#         h["vitamin_D"] = pd.Series(np.nan, index=df.index)

#     h["vitamin_D_deficiency"] = h["vitamin_D"].apply(lambda x: 1 if pd.notna(x) and x < 50 else (0 if pd.notna(x) else np.nan))

#     return h

# # =====================================================
# # Process all cycles
# # =====================================================
# all_cycles = ["2001","2003","2005","2007","2009","2011","2013","2015","2017","2021"]

# all_dfs = []

# for cycle in all_cycles:
#     df_cycle = harmonize_cycle(cycle)

#     if df_cycle.empty:
#         print(f"Skipping {cycle} (empty)")
#         continue

#     # Save per-cycle file
#     df_cycle.to_csv(f"harmonised_{cycle}.csv", index=False)

#     all_dfs.append(df_cycle)

# # Master file
# harmonised_master = pd.concat(all_dfs, ignore_index=True)

# # Adjust weight for pooled cycles
# harmonised_master["MEC_weight_adj"] = harmonised_master["MEC_weight"] / len(all_cycles)

# harmonised_master.to_csv("harmonised_all_cycles.csv", index=False)
# print("Harmonization complete. Master file saved.")

# ================================
# Load CSV safely
# ================================
df = pd.read_csv("harmonised_all_cycles.csv", low_memory=False)

# Filter rows missing vitamin D status
df = df[df["vitamin_D_deficiency"].notna()]
# Count missing values per column
missing_summary = df.isna().sum().sort_values(ascending=False)

# Optional: percentage of missing
missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)

# Combine into one table
missing_df = pd.DataFrame({
    "missing_count": missing_summary,
    "missing_percent": missing_percent
})

print("Missing Values for each Variable in dataset:")
print(missing_df)

sun_vars = ["sun_exposure", "sun_shade", "sun_shirt", "sun_sunscreen"]

for col in sun_vars:
    print(f"\n{col} by age_cat:")
    print(
        df
        .groupby("age_cat")[col]
        .apply(lambda x: x.value_counts(dropna=False))
    )

# ================================
# NHANES MICE IMPUTATION PIPELINE (FINAL WITH NA CATEGORY)
# ================================

# --------------------------------
# 1. LOAD DATA
# --------------------------------
# df = pd.read_csv("harmonised_all_cycles.csv", low_memory=False)

# --------------------------------
# 2. VARIABLE GROUPS
# --------------------------------
NUMERIC_VARS = [
    "total_PA_days",
    "alcohol_drinks_per_year",
    "calcium_intake",
    "calcium_lab",
    "household_size"
]

ORDINAL_VARS = [
    "BMI_cat",
    # "PA_category",          
    "sun_exposure",
    "sun_shade",
    "sun_shirt",
    "sun_sunscreen",
    "milk_intake",
    "smoker_current",
    "diabetes"
]

BINARY_VARS = [
    "smoker_ever",
    "alcohol_yesno"
]


STRUCTURAL_VARS = [
    "milk_intake",
    "sun_exposure",
    "sun_shade",
    "sun_shirt",
    "sun_sunscreen",
    "PA_category"
]


df_mice = df.copy()

# --------------------------------
# 3. ADD MISSINGNESS INDICATORS
# --------------------------------
for col in NUMERIC_VARS + ORDINAL_VARS + BINARY_VARS:
    df_mice[f"{col}_missing"] = df_mice[col].isna().astype(int)

# --------------------------------
# 4. MAP BINARY VARIABLES → 0/1
# --------------------------------
binary_maps = {
    "smoker_ever": {"No": 0, "Yes": 1},
    "alcohol_yesno": {"No": 0, "Yes": 1}
}


for col, mapping in binary_maps.items():
    df_mice[col] = df_mice[col].map(mapping)


# ======================================================
# 4. STRUCTURAL MISSING RULES
# ======================================================

SUN_ELIGIBLE = ["18–44", "45–65"]
mask_under18 = df_mice["age_cat"] == "<18"
mask_sun_eligible = df_mice["age_cat"].isin(SUN_ELIGIBLE)
mask_sun_ineligible = ~mask_sun_eligible

# Sun vars: outside 18–65 → -1, inside 18–65 NaN stays for imputation
# Sun variables: eligible 18–65, others → -1
sun_vars = ["sun_exposure", "sun_shade", "sun_shirt", "sun_sunscreen"]
mask_ineligible = ~df_mice["age_cat"].isin(SUN_ELIGIBLE)
for col in sun_vars:
    df_mice.loc[mask_ineligible, col] = -1
    df_mice.loc[mask_ineligible, f"{col}_missing"] = 0


# Alcohol & smoking <18 → 0
alc_smoke_vars = ["alcohol_drinks_per_year", "alcohol_yesno", "smoker_current", "smoker_ever"]

for col in alc_smoke_vars:
    df_mice.loc[mask_under18, col] = 0
    df_mice.loc[mask_under18, f"{col}_missing"] = 0

# Milk intake: <18 → -1
df_mice.loc[mask_under18, "milk_intake"] = -1
df_mice.loc[mask_under18, "milk_intake_missing"] = 0

# PA_category: will be derived from total_PA_days later
df_mice.loc[mask_under18, "PA_category"] = -1
df_mice.loc[mask_under18, "PA_category_missing"] = 0


ordinal_maps = {
    # Age-independent ordinals — no -1
    "BMI_cat": {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3},
    "smoker_current": {"Not at all": 0, "Some days": 1, "Every day": 2},
    "diabetes": {"No": 0, "Borderline": 1, "Yes": 2},

    # Structural vars — include -1
    "milk_intake": {"Never": 0, "Sometimes": 1, "Regular": 2, -1: -1},
    "sun_exposure": {"Never": 0, "Rare": 1, "Sometimes": 2, "Frequent": 3, "Regular": 4, -1: -1},
    "sun_shade": {"No outdoor exposure": 0, "Never": 1, "Rarely": 2, "Sometimes": 3, "Most": 4, "Always": 5, -1: -1},
    "sun_shirt": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Most": 3, "Always": 4, -1: -1},
    "sun_sunscreen": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Most": 3, "Always": 4, -1: -1}
}


for col, mapping in ordinal_maps.items():
    # Map values; leave NaN as-is for MICE
    df_mice[col] = df_mice[col].map(mapping)

# PRESERVE <18 smoker_current = 0
df_mice.loc[mask_under18, "smoker_current"] = 0

# One-hot encode cycle
df_mice = pd.get_dummies(df_mice, columns=["cycle"], drop_first=True)


# ======================================================
# 7. MICE IMPUTATION (COLUMN-WISE, SAFE)
# ======================================================
def impute_with_mask(df, cols, imputer, eligible_mask):
    df_sub = df.loc[eligible_mask, cols]
    nan_mask = df_sub.isna()
    if nan_mask.sum().sum() == 0:
        return df
    imputed = imputer.fit_transform(df_sub)
    for i, col in enumerate(cols):
        df.loc[eligible_mask & nan_mask[col], col] = imputed[nan_mask[col], i]
    return df

# ---------------- MICE imputers ----------------
# Numeric and ordinal imputers
num_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
    max_iter=10, random_state=42
)
ordinal_imputer = IterativeImputer(
    estimator=RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
    max_iter=10, random_state=42
)
bin_imputer = IterativeImputer(
    estimator=RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
    max_iter=10, random_state=42
)

# ================================
# 10. RUN MICE
# ================================
# Numeric imputation
NUMERIC_IMPUTE = [
        "total_PA_days", 
        "alcohol_drinks_per_year", 
        "calcium_intake", 
        "calcium_lab", 
        "household_size"]
for col in NUMERIC_IMPUTE:
    df_mice[col] = pd.to_numeric(df_mice[col], errors='coerce')
df_mice = impute_with_mask(df_mice, NUMERIC_IMPUTE, num_imputer, pd.Series(True, index=df_mice.index))

# mask_pa_eligible_days = df_mice["age_cat"] != "<18"

# df_mice["total_PA_days"] = pd.to_numeric(
#     df_mice["total_PA_days"], errors="coerce"
# )

# df_mice = impute_with_mask(
#     df_mice,
#     cols=["total_PA_days"],
#     imputer=num_imputer,
#     eligible_mask=mask_pa_eligible_days
# )

# Ordinal imputation (only age-eligible for structural vars)
mask_milk_eligible = df_mice["age_cat"] != "<18"
mask_smoker_current_eligible = df_mice["age_cat"] != "<18"
mask_bmi_eligible = pd.Series(True, index=df_mice.index)
mask_diabetes_eligible = pd.Series(True, index=df_mice.index)

# BMI_cat
df_mice = impute_with_mask(df_mice, ["BMI_cat"], ordinal_imputer, mask_bmi_eligible)
# Diabetes
df_mice = impute_with_mask(df_mice, ["diabetes"], ordinal_imputer, mask_diabetes_eligible)
# Milk intake >=18
df_mice = impute_with_mask(df_mice, ["milk_intake"], ordinal_imputer, mask_milk_eligible)
# Smoker_current >=18
df_mice = impute_with_mask(df_mice, ["smoker_current"], ordinal_imputer, mask_smoker_current_eligible)
# Sun variables 18–65
df_mice = impute_with_mask(df_mice, sun_vars, ordinal_imputer, mask_sun_eligible)

# Binary imputation
df_mice = impute_with_mask(df_mice, ["smoker_ever", "alcohol_yesno"], bin_imputer, pd.Series(True, index=df_mice.index))


# --------------------------------
# 7. POST-IMPUTATION CLEANUP
# --------------------------------
# Binary rounding
df_mice["smoker_ever"] = (df_mice["smoker_ever"] >= 0.5).astype("Int64")
df_mice["alcohol_yesno"] = (df_mice["alcohol_yesno"] >= 0.5).astype("Int64")

# Ordinal rounding & clipping
ORDINAL_MAX = {
    "BMI_cat": 3,
    "sun_exposure": 4,
    "sun_shade": 5,
    "sun_shirt": 4,
    "sun_sunscreen": 4,
    "milk_intake": 2,
    "smoker_current": 2,
    "diabetes": 2
}

def round_and_safe_int64(series, min_val=None, max_val=None, structural_missing=-1):
    series_safe = series.copy()
    mask_struct = series_safe == structural_missing
    series_safe = series_safe.round()
    if min_val is not None:
        series_safe = series_safe.clip(lower=min_val)
    if max_val is not None:
        series_safe = series_safe.clip(upper=max_val)
    series_safe[mask_struct] = structural_missing
    return series_safe.astype("Int64")

for col, max_val in ORDINAL_MAX.items():
    struct_missing = -1 if col in STRUCTURAL_VARS else None
    df_mice[col] = round_and_safe_int64(df_mice[col], min_val=0, max_val=max_val, structural_missing=struct_missing if struct_missing else 0)

# Numeric cleanup
df_mice["total_PA_days"] = df_mice["total_PA_days"].round().clip(0, 7).astype("Int64")
df_mice["alcohol_drinks_per_year"] = df_mice["alcohol_drinks_per_year"].round().clip(lower=0).astype("Int64")
df_mice["calcium_intake"] = df_mice["calcium_intake"].round().astype("Int64")
df_mice["calcium_lab"] = df_mice["calcium_lab"].round(1).astype(float)
df_mice["household_size"] = df_mice["household_size"].round().astype("Int64")


# --------------------------------
# 8. DERIVE PA_CATEGORY
# --------------------------------
def derive_pa_category(days):
    if pd.isna(days):
        return -1  # map missing to NA
    elif days <= 1:
        return 0  # Low
    elif 2 <= days <= 4:
        return 1  # Moderate
    else:
        return 2  # High

df_mice["PA_category"] = df_mice["total_PA_days"].apply(derive_pa_category).astype("Int64")
df_mice["PA_category_missing"] = (df_mice["PA_category"] == -1).astype(int)

df_mice["total_PA_days_missing"] = df_mice["total_PA_days"].isna().astype(int)

# Step 3: Fill missing with -1 after clipping
df_mice["total_PA_days"] = df_mice["total_PA_days"].fillna(-1).astype("Int64")

# --------------------------------
# 9. FINAL VALIDATION
# --------------------------------
print("Remaining missing values after MICE:")
print(df_mice[NUMERIC_VARS + ORDINAL_VARS + BINARY_VARS].isna().sum())


# ---------------------------
# 1. Handle missing for protected vars
# ---------------------------
PROTECTED_VARS = ["age_cat", "gender", "race_ethnicity", "education", "PIR_cat"]

for col in PROTECTED_VARS:
    df_mice[col] = df_mice[col].fillna("Missing")  # replace None/NaN with 'Missing'

# # If you want numeric mapping for education / PIR:
# edu_map = {
#     "Less than high school": 0,
#     "High school": 1,
#     "College": 2,
#     "Missing": -1
# }
# pir_map = {
#     "<1.3": 0,
#     "1.3–3.5": 1,
#     "3.5–5": 2,
#     "5+": 3,
#     "Missing": -1
# }

# df_mice["education_num"] = df_mice["education"].map(edu_map)
# df_mice["pir_num"] = df_mice["pir"].map(pir_map)

# ---------------------------
# 2. One-hot encode protected vars (including missing)
# ---------------------------
df_mice = pd.get_dummies(df_mice, columns=PROTECTED_VARS, drop_first=False, dtype=int)

df_mice.columns = df_mice.columns.str.strip()           # remove leading/trailing spaces
df_mice.columns = df_mice.columns.str.replace('–', '-', regex=False)
df_mice.columns = df_mice.columns.str.replace(' ', '_', regex=False)

# Save imputed dataset
df_mice.to_csv("harmonised_all_cycles_imputed.csv", index=False)

# Updated list of columns to keep for ML models (vitamin_D removed)
feature_cols = [
    # Demographics / protected attributes (one-hot encoded)
    'age_cat_18-44', 'age_cat_45-65', 'age_cat_<18', 'age_cat_>65',
    'gender_Female', 'gender_Male',
    'race_ethnicity_Mexican_American', 'race_ethnicity_Non-Hispanic_Black',
    'race_ethnicity_Non-Hispanic_White', 'race_ethnicity_Other',
    'race_ethnicity_Other_Hispanic',
    'education_College', 'education_High_school', 'education_High_school_or_above',
    'education_High_school_or_below', 'education_Less_than_high_school',
    'education_Low_education', 'education_Missing',
    'PIR_cat_High', 'PIR_cat_Low', 'PIR_cat_Middle', 'PIR_cat_Missing',

    # Health / lifestyle / nutrition
    'household_size', 'BMI_cat', 'smoker_ever', 'smoker_current',
    'alcohol_yesno', 'alcohol_drinks_per_year', 'milk_intake', 'calcium_intake',
    'total_PA_days', 'PA_category', 'sun_exposure', 'sun_shade', 'sun_shirt',
    'sun_sunscreen', 'diabetes', 'calcium_lab', 'vitamin_D_deficiency'  # Target variable is separate
]

# Create subset dataframe
df_ml = df_mice[feature_cols + ["MEC_weight_adj"]].copy()

# Save imputed dataset
df_ml.to_csv("harmonised_all_cycles_ml_dateset.csv", index=False)

df = pd.read_csv("harmonised_all_cycles_ml_dateset.csv", low_memory=False)

# Separate features and target
X = df.drop('vitamin_D_deficiency', axis=1)
y = df['vitamin_D_deficiency']

# Check
print(X.shape, y.shape)
print(X.columns.tolist())
