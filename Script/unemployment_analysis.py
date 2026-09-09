"""
UNEMPLOYMENT ANALYSIS WITH PYTHON
------------------------------------
Goal: Explore unemployment rate trends over time, understand the impact
of Covid-19, and spot seasonal / demographic patterns.

Dataset link: https://www.kaggle.com/datasets/aniruddhasshirahatti/us-unemployment-dataset-2010-2020?resource=download
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------
NATIONAL_DATA_PATH = "unemployment_data_us.csv"        
STATE_DATA_PATH = "unemployment_data_us_state.csv"      
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")


# =================================================================
# NATIONAL DATA (unemployment_data_us.csv)
# =================================================================

# ---------------------------------------------------------------
# STEP 1: LOAD AND TAKE A FIRST LOOK
# ---------------------------------------------------------------
print("STEP 1: Loading the national dataset...")
national = pd.read_csv(NATIONAL_DATA_PATH)

print(f"Shape: {national.shape}")
print("\nColumns:", national.columns.tolist())
print("\nFirst 5 rows:")
print(national.head())

print("\nMissing values per column:")
print(national.isnull().sum())


# ---------------------------------------------------------------
# STEP 2: CLEAN THE NATIONAL DATA
# ---------------------------------------------------------------
print("\nSTEP 2: Cleaning the national data...")

national["Date"] = pd.to_datetime(national["Date"], format="%b-%Y")
national = national.sort_values("Date").reset_index(drop=True)

missing_rows = national[national.isnull().any(axis=1)]
print("Rows with missing data:")
print(missing_rows[["Year", "Month", "Date"]])

# Finding: every missing row is April 2020 through December 2020.
# In other words, this file's education/race/gender breakdown simply
# stops being reported right as Covid-19 hit the US in March 2020.
# We can't invent 9 months of demographic data, so we drop these rows --
# but we keep this fact in mind: this file alone CANNOT show us the
# pandemic's demographic impact, only its very first signs (Jan-Mar 2020).
national_clean = national.dropna().reset_index(drop=True)
print(f"\nDropped {len(national) - len(national_clean)} rows "
      "(Apr-Dec 2020 -- demographic data not reported after Covid hit).")

print("\nDuplicate rows:", national_clean.duplicated().sum())
national_clean = national_clean.drop_duplicates()

education_cols = ["Primary_School", "High_School", "Associates_Degree", "Professional_Degree"]
race_cols = ["White", "Black", "Asian", "Hispanic"]
gender_cols = ["Men", "Women"]


# ---------------------------------------------------------------
# LONG-TERM TREND (2010 - MARCH 2020)
# ---------------------------------------------------------------
print("\nSTEP 3: Plotting long-term unemployment trends by education level...")

plt.figure(figsize=(12, 6))
for col in education_cols:
    plt.plot(national_clean["Date"], national_clean[col], label=col.replace("_", " "))
plt.title("Unemployment Rate by Education Level (2010 - Mar 2020)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/trend_by_education.png")
plt.close()

plt.figure(figsize=(12, 6))
for col in race_cols:
    plt.plot(national_clean["Date"], national_clean[col], label=col)
plt.title("Unemployment Rate by Race (2010 - Mar 2020)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/trend_by_race.png")
plt.close()

plt.figure(figsize=(12, 6))
for col in gender_cols:
    plt.plot(national_clean["Date"], national_clean[col], label=col)
plt.title("Unemployment Rate by Gender (2010 - Mar 2020)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/trend_by_gender.png")
plt.close()

print("Saved: trend_by_education.png, trend_by_race.png, trend_by_gender.png")


# ---------------------------------------------------------------
# EARLY COVID SIGNAL IN THE NATIONAL DATA (Jan-Mar 2019 vs 2020)
# ---------------------------------------------------------------
print("\nSTEP 4: Comparing Jan-Mar 2019 vs Jan-Mar 2020 by demographic group...")

early_2019 = national_clean[(national_clean["Year"] == 2019) & (national_clean["Month"].isin(["Jan", "Feb", "Mar"]))]
early_2020 = national_clean[(national_clean["Year"] == 2020) & (national_clean["Month"].isin(["Jan", "Feb", "Mar"]))]

comparison = pd.DataFrame({
    "2019 (Jan-Mar avg)": early_2019[education_cols + race_cols + gender_cols].mean(),
    "2020 (Jan-Mar avg)": early_2020[education_cols + race_cols + gender_cols].mean(),
})
comparison["Change"] = comparison["2020 (Jan-Mar avg)"] - comparison["2019 (Jan-Mar avg)"]
comparison = comparison.sort_values("Change", ascending=False)

print(comparison)

plt.figure(figsize=(10, 7))
sns.barplot(x=comparison["Change"], y=comparison.index, color="steelblue")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Change in Unemployment Rate: Jan-Mar 2019 vs Jan-Mar 2020")
plt.xlabel("Percentage Point Change")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/early_covid_change_by_group.png")
plt.close()


# ---------------------------------------------------------------
# SEASONAL PATTERN (USING FULL HISTORY, 2010-2019 ONLY)
# ---------------------------------------------------------------
print("\nSTEP 5: Checking seasonal patterns (using stable pre-2020 years)...")

# We use 2010-2019 here (not 2020) so a single unusual year doesn't
# distort what a "normal" seasonal pattern looks like.
pre_2020 = national_clean[national_clean["Year"] < 2020].copy()
pre_2020["Overall_Rate"] = pre_2020[race_cols].mean(axis=1)  # rough overall proxy

month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
seasonal = pre_2020.groupby("Month")["Overall_Rate"].mean().reindex(month_order)

plt.figure(figsize=(10, 6))
sns.barplot(x=seasonal.index, y=seasonal.values, color="steelblue")
plt.title("Average Unemployment Rate by Month (2010-2019, Seasonal Pattern)")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/seasonal_pattern.png")
plt.close()

print("Saved: seasonal_pattern.png")


# =================================================================
# B: STATE-LEVEL DATA (unemployment_data_us_state.csv)
# =================================================================

# ---------------------------------------------------------------
# LOAD AND CLEAN THE STATE-LEVEL DATA
# ---------------------------------------------------------------
print("\nSTEP 6: Loading the state-level dataset...")
state_df = pd.read_csv(STATE_DATA_PATH)

print(f"Shape: {state_df.shape}")
print("Columns:", state_df.columns.tolist())
print("\nMissing values:")
print(state_df.isnull().sum())

state_df["Date"] = pd.to_datetime(state_df["Date"], format="%b-%Y")

# There's one missing Unemployment_Rate value (Puerto Rico, March 2020).
# Since we have Jan and Feb for Puerto Rico, we fill the gap using
# that state's own average instead of dropping the row entirely --
# this keeps Puerto Rico in the state comparison below.
missing_before = state_df["Unemployment_Rate"].isnull().sum()
state_df["Unemployment_Rate"] = state_df.groupby("State")["Unemployment_Rate"].transform(
    lambda x: x.fillna(x.mean())
)
print(f"\nFilled {missing_before} missing value(s) using that state's own average.")

print("\nDuplicate rows:", state_df.duplicated().sum())
state_df = state_df.drop_duplicates()


# ---------------------------------------------------------------
# WHICH STATES SAW THE BIGGEST EARLY JUMP (JAN vs MAR 2020)?
# ---------------------------------------------------------------
print("\nSTEP 7: Ranking states by change from January to March 2020...")

pivot = state_df.pivot(index="State", columns="Month", values="Unemployment_Rate")
pivot["Change_Jan_to_Mar"] = pivot["Mar"] - pivot["Jan"]
pivot = pivot.sort_values("Change_Jan_to_Mar", ascending=False)

print("\nTop 10 states with the biggest early increase:")
print(pivot.head(10))

top_states = pivot.head(15)
plt.figure(figsize=(10, 8))
sns.barplot(x=top_states["Change_Jan_to_Mar"], y=top_states.index, color="steelblue")
plt.title("Top 15 States: Unemployment Rate Increase, Jan to Mar 2020")
plt.xlabel("Percentage Point Increase")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_states_early_covid_jump.png")
plt.close()

print("Saved: top_states_early_covid_jump.png")


# ---------------------------------------------------------------
# NATIONAL AVERAGE TREND ACROSS THE SAME 3 MONTHS
# ---------------------------------------------------------------
print("\nSTEP 8: National average trend, Jan-Mar 2020 (all states)...")

monthly_national_avg = state_df.groupby("Month")["Unemployment_Rate"].mean().reindex(["Jan", "Feb", "Mar"])
print(monthly_national_avg)

plt.figure(figsize=(8, 5))
sns.lineplot(x=monthly_national_avg.index, y=monthly_national_avg.values, marker="o", color="steelblue")
plt.title("Average State Unemployment Rate: Jan-Mar 2020")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/state_avg_jan_to_mar.png")
plt.close()


# ---------------------------------------------------------------
# SAVE CLEAN SUMMARY FILES
# ---------------------------------------------------------------
print("\nSTEP 9: Saving cleaned summary data...")
comparison.to_csv(f"{OUTPUT_DIR}/national_demographic_change_summary.csv")
pivot.to_csv(f"{OUTPUT_DIR}/state_jan_to_mar_summary.csv")

print("\nDone! Check the 'outputs' folder for all charts and summary CSVs.")
print("\nKey takeaways to write up in your report:")
print(" - The national file's demographic breakdown stops being reported")
print("   after March 2020, right as Covid-19 hit -- a real data limitation.")
print(" - Even in just Jan-Mar 2020, several demographic groups and states")
print("   already show a measurable increase in unemployment (see the")
print("   'Change' columns in the two summary CSVs).")
print(" - Historically (2010-2019), unemployment shows a mild seasonal")
print("   pattern -- see seasonal_pattern.png.")
