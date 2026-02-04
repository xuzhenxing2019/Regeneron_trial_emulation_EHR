# Databricks notebook source
import re
from functools import reduce
from time import time
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    coalesce, col, collect_list, count, countDistinct, datediff, expr,
    first, least, lower, lit, when)
from pyspark.sql.functions import min as ps_min, max as ps_max
from copy import deepcopy
import os

schema = "gd_prd.optum_onco_202502"

# COMMAND ----------

therapy_version = "combined_therapy_pres_adm_pro_14_90"
cancer_type = "ansclc"

# COMMAND ----------

df_cohort = spark.sql("SELECT * FROM gd_prd.dh_rwdna_ml.optum_202502_{}_cohort_{}".format(cancer_type, therapy_version))
display(df_cohort)
df_cohort = df_cohort.toPandas()
df_cohort = df_cohort.loc[:,~df_cohort.columns.duplicated()].copy()

# COMMAND ----------

df_cohort.columns

# COMMAND ----------

print("Size: {}".format(len(df_cohort)))
df_cohort = df_cohort[(abs((pd.to_datetime(df_cohort['prior_malignancy_date']) - pd.to_datetime(df_cohort['initial_amel_dx_date'])).dt.days) > 365 * 5) | df_cohort['prior_malignancy_date'].isna()]
print("Size: {}".format(len(df_cohort)))

obj_col = ['ptid', 'age_at_dx', 'age_at_amel_dx', 'ecog']
df_cohort = df_cohort[obj_col]

# COMMAND ----------

# MAGIC %md
# MAGIC # Demographics
# MAGIC
# MAGIC Demographics derived from `pt_clinical`

# COMMAND ----------

demotbl = spark.sql(
    """
    SELECT  p.ptid,
            ix.age_at_ansclc_dx_date AS age,
            gender,
            race,
            ethnicity,
            region
            
        FROM gd_prd.optum_onco_202502.pt_clinical p
        INNER JOIN gd_prd.dh_rwdna_ml.optum_202502_ansclc_first_line_cohort ix
        ON p.ptid = ix.ptid
    """
)

display(demotbl.limit(10))
dem_df = demotbl.toPandas()
# remove repeated rows
dem_df = dem_df.drop_duplicates(subset=['ptid'])


# COMMAND ----------

dem_df_chort = dem_df[['ptid', 'gender', 'race', 'ethnicity', 'region']]
dem_df_chort

# COMMAND ----------

region = 'South'
dem_df_chort_tem = dem_df_chort[dem_df_chort['region'] == region]
print("Size: {}".format(len(dem_df_chort_tem)))

# COMMAND ----------

# MAGIC %md
# MAGIC # Smoking status
# MAGIC
# MAGIC Just use observations table; require observations within 180d
# MAGIC
# MAGIC Smoking status ranking (take closest, then apply ranking): Current > Former > Never > Unknown

# COMMAND ----------

smoking_status = spark.sql(
    """
    WITH smoking AS (
        SELECT  ptid,
                (
                    CASE
                        WHEN LOWER(obs_result) RLIKE 'current' THEN 'current'
                        WHEN LOWER(obs_result) RLIKE 'previous' THEN 'former'
                        WHEN LOWER(obs_result) RLIKE 'never|(^not applicable|current)' THEN 'never'
                        ELSE 'unknown'
                    END
                ) AS smoking_status,
                (
                    CASE
                        WHEN LOWER(obs_result) RLIKE 'current' THEN 1
                        WHEN LOWER(obs_result) RLIKE 'former' THEN 2
                        WHEN LOWER(obs_result) RLIKE 'never' THEN 3
                        ELSE 4
                    END
                ) AS ranker,
                obs_date AS test_date
        FROM gd_prd.optum_onco_202502.obs
        WHERE LOWER(obs_type) = 'smoke' AND LOWER(obs_result) NOT RLIKE 'cessation'
    ), sm_interim AS (  -- JOIN WITH INDEX TABLE
        SELECT  smoking.*,
                ix.index_date,
                ABS(CAST(DATE_DIFF(ix.index_date, smoking.test_date) AS INTEGER)) AS diff
        FROM smoking
        INNER JOIN gd_prd.dh_rwdna_ml.optum_202502_ansclc_first_line_cohort ix
        ON smoking.ptid = ix.ptid
            AND smoking.test_date BETWEEN ix.index_date - INTERVAL 180 DAY
                AND ix.index_date - INTERVAL 1 DAY 
    )
    SELECT  ptid,
            index_date,
            FIRST(smoking_status) OVER (PARTITION BY ptid ORDER BY diff, ranker) AS smoking_status,
            FIRST(test_date) OVER (PARTITION BY ptid ORDER BY diff, ranker) AS test_date
    FROM sm_interim
    """
).dropDuplicates().cache()

display(smoking_status.limit(50))

# COMMAND ----------

df_smoking = smoking_status.toPandas()
# remove repeated rows
df_smoking = df_smoking.drop_duplicates(subset=['ptid'])

df_smoking = df_smoking[['ptid', 'smoking_status']]

df_smoking

# COMMAND ----------

# MAGIC %md
# MAGIC # outcome overall survial and SAE

# COMMAND ----------

dod_data = pd.read_csv("./exp_data/{}/{}/df_dod.csv".format(cancer_type, therapy_version))
dod_data['available_date'] = pd.to_datetime(dod_data['available_date']).dt.date

with open('./exp_data/{}/{}/pat_diag_sae_dict.pkl'.format(cancer_type, therapy_version), "rb") as f:
    pat_diag_sae_dict = pkl.load(f) 

# COMMAND ----------

dod_data

# COMMAND ----------

pat_diag_sae_dict

# COMMAND ----------

# MAGIC %md
# MAGIC # combine dem, smoking with df_cohort

# COMMAND ----------

# merge df_cohort, dem_df_chort, df_smoking, based on the ptid in df_cohort
df_cohort_combined = df_cohort.merge(dem_df_chort, how='left', on='ptid')
df_cohort_combined = df_cohort_combined.merge(df_smoking, how='left', on='ptid')

# fill nan gender
df_cohort_combined['gender'] = df_cohort_combined['gender'].fillna('Female')
df_cohort_combined['gender'] = df_cohort_combined['gender'].fillna('Female')

# address race
df_cohort_combined.loc[df_cohort_combined['race'].isin(['Caucasian']), 'race'] = 'White'
df_cohort_combined.loc[~df_cohort_combined['race'].isin(['White']), 'race'] = 'Non White'

# address ethnicity
df_cohort_combined.loc[~df_cohort_combined['ethnicity'].isin(['Not Hispanic','Hispanic']), 'ethnicity'] = 'Not Hispanic'

# address smoking_status
df_cohort_combined.loc[df_cohort_combined['smoking_status'].isin(['current', 'former']), 'smoking_status'] = 'History of smoking'
df_cohort_combined.loc[~df_cohort_combined['smoking_status'].isin(['History of smoking']), 'smoking_status'] = 'No history of smoking'

# address ecog 0, 1, >=2
df_cohort_combined.loc[df_cohort_combined['ecog']>=2, 'ecog'] = 2
df_cohort_combined.loc[~df_cohort_combined['ecog'].isin([0,1,2]), 'ecog'] = 0

# address region
df_cohort_combined.loc[~df_cohort_combined['region'].isin(['West', 'Northeast', 'South', 'Midwest']), 'region'] = 'Other/Unknown'


df_cohort_combined

# COMMAND ----------

# Compute summary statistics and store in a pandas DataFrame
import pandas as pd

results = []

# 1) Median and IQR for age_at_dx
age_dx_median = round(df_cohort_combined['age_at_dx'].median(),2)
age_dx_iqr = (round(df_cohort_combined['age_at_dx'].quantile(0.25),2), round(df_cohort_combined['age_at_dx'].quantile(0.75),2))
results.append(
    {"variable": "age_at_dx", "metric": "median, IQR", "value": str(age_dx_median)+str((age_dx_iqr))}
)


# 2) Median and IQR for age_at_amel_dx
age_amel_median = round(df_cohort_combined['age_at_amel_dx'].median(),2)
age_amel_iqr = (round(df_cohort_combined['age_at_amel_dx'].quantile(0.25),2), round(df_cohort_combined['age_at_amel_dx'].quantile(0.75),2))
results.append(
    {"variable": "age_at_amel_dx", "metric": "median, IQR", "value": str(age_amel_median)+str((age_amel_iqr))}
)

# 3) Categorical counts and percentages
total_patients = len(df_cohort_combined)

cat_vars = ["gender", "race", "ethnicity", "region", "smoking_status", "ecog"]
for var in cat_vars:
    if var in df_cohort_combined.columns:
        counts = df_cohort_combined[var].value_counts(dropna=False)
        for category, cnt in counts.items():
            pct = (cnt / total_patients) * 100
            results.append(
                {
                    "variable": var,
                    "category": category,
                    "metric": "count (percentage)",
                    "value": str(int(cnt)) + " (" + str(round(pct, 2)) + "%)",
                }
            )

# Assemble results into a DataFrame
summary_df = pd.DataFrame(results)

summary_df.to_csv('./exp_data/{}/{}/{}/{}_characteristic_summary.csv'.format(cancer_type, therapy_version, 'tem_result', cancer_type), index=False)


# COMMAND ----------

summary_df

# COMMAND ----------

# MAGIC %md
# MAGIC # add supportive care drugs

# COMMAND ----------

# please read the following file
data_soc = pd.read_excel('/Workspace/Users/zhenxing.xu@regeneron.com/ansclc_0915/exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_HR_p_value_dod_forest_plot.xlsx')

N_total = 3858

data_soc['num_treated_percentage'] = data_soc['num_treated'].apply(
    lambda x: f"{int(x)} ({round(x / N_total * 100, 2)}%)"
)

data_soc.to_csv('/Workspace/Users/zhenxing.xu@regeneron.com/ansclc_0915/exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_HR_p_value_dod_forest_plot_with_percentage.csv')

data_soc

# COMMAND ----------

print('Done.')