# Databricks notebook source
import re
from functools import reduce
from time import time

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

print("Size: {}".format(len(df_cohort)))
df_cohort = df_cohort[(abs((pd.to_datetime(df_cohort['prior_malignancy_date']) - pd.to_datetime(df_cohort['initial_amel_dx_date'])).dt.days) > 365 * 5) | df_cohort['prior_malignancy_date'].isna()]
print("Size: {}".format(len(df_cohort)))

# COMMAND ----------

df_rx_presc_his= spark.sql(
    """
    SELECT *
    FROM gd_prd.optum_onco_202502.rx_presc rx
    INNER JOIN (
        SELECT ptid, initial_amel_dx_date, combined_therapy_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_{}_cohort_{}) cohort
    ON rx.ptid = cohort.ptid AND rx.RXDATE < cohort.combined_therapy_date
    """.format(cancer_type, therapy_version))

df_rx_adm_his = spark.sql(
    """
    SELECT *
    FROM gd_prd.optum_onco_202502.rx_adm rx
    INNER JOIN (
        SELECT ptid, initial_amel_dx_date, combined_therapy_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_{}_cohort_{}) cohort
    ON rx.ptid = cohort.ptid AND rx.ADMIN_DATE < cohort.combined_therapy_date
    """.format(cancer_type, therapy_version))

df_rx_presc_his = df_rx_presc_his.toPandas()
df_rx_presc_his = df_rx_presc_his.loc[:,~df_rx_presc_his.columns.duplicated()].copy()

df_rx_adm_his = df_rx_adm_his.toPandas()
df_rx_adm_his = df_rx_adm_his.loc[:,~df_rx_adm_his.columns.duplicated()].copy()

# COMMAND ----------

df_rx_presc = spark.sql(
    """
    SELECT *
    FROM gd_prd.optum_onco_202502.rx_presc rx
    INNER JOIN (
        SELECT ptid, initial_amel_dx_date, combined_therapy_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_{}_cohort_{}) cohort
    ON rx.ptid = cohort.ptid AND rx.RXDATE >= cohort.combined_therapy_date AND ABS(DATEDIFF(rx.RXDATE, cohort.combined_therapy_date)) <= (90)
    """.format(cancer_type, therapy_version))

df_rx_adm = spark.sql(
    """
    SELECT *
    FROM gd_prd.optum_onco_202502.rx_adm rx
    INNER JOIN (
        SELECT ptid, initial_amel_dx_date, combined_therapy_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_{}_cohort_{}) cohort
    ON rx.ptid = cohort.ptid AND rx.ADMIN_DATE >= cohort.combined_therapy_date AND ABS(DATEDIFF(rx.ADMIN_DATE, cohort.combined_therapy_date)) <= (90)
    """.format(cancer_type, therapy_version))

df_rx_presc = df_rx_presc.toPandas()
df_rx_presc = df_rx_presc.loc[:,~df_rx_presc.columns.duplicated()].copy()

df_rx_adm = df_rx_adm.toPandas()
df_rx_adm = df_rx_adm.loc[:,~df_rx_adm.columns.duplicated()].copy()

# COMMAND ----------

df_neoplastics = spark.sql("SELECT * FROM gd_prd.dh_rwdna_ml.nsclc_neoplastics_ndc_cpt_hcpcs")
display(df_neoplastics)
df_neoplastics = df_neoplastics.toPandas()

# COMMAND ----------

df_rx_adm_clippped = df_rx_adm[['ptid', 'ADMIN_DATE', 'initial_amel_dx_date', 'combined_therapy_date', 'generic_desc']]
df_rx_adm_clippped.rename(columns={'ADMIN_DATE': 'RXDATE'}, inplace=True)
df_rx_adm_clippped['Type'] = "adm"
df_rx_presc_clipped = df_rx_presc[['ptid', 'RXDATE', 'initial_amel_dx_date', 'combined_therapy_date','generic_desc']]
df_rx_presc_clipped['Type'] = "pres"
df_rx_combined = pd.concat([df_rx_adm_clippped, df_rx_presc_clipped])

# # PEGFILGRASTIM/PEGFILGRASTIM-APGF – same drug; using PEGFILGRASTIM to replace PEGFILGRASTIM-APGF
df_rx_combined['generic_desc'] = df_rx_combined['generic_desc'].replace('PEGFILGRASTIM-APGF', 'PEGFILGRASTIM')
df_rx_combined['generic_desc'] = df_rx_combined['generic_desc'].replace('DEXAMETHASONE SODIUM PHOSPHATE', 'DEXAMETHASONE')

# COMMAND ----------

df_rx_adm_his_clippped = df_rx_adm_his[['ptid', 'ADMIN_DATE', 'initial_amel_dx_date', 'combined_therapy_date','generic_desc']]
df_rx_adm_his_clippped.rename(columns={'ADMIN_DATE': 'RXDATE'}, inplace=True)
df_rx_adm_his_clippped['Type'] = "adm"
df_rx_presc_his_clipped = df_rx_presc_his[['ptid', 'RXDATE', 'initial_amel_dx_date', 'combined_therapy_date','generic_desc']]
df_rx_presc_his_clipped['Type'] = "pres"
df_rx_his_combined = pd.concat([df_rx_adm_his_clippped, df_rx_presc_his_clipped])

# # PEGFILGRASTIM/PEGFILGRASTIM-APGF – same drug; using PEGFILGRASTIM to replace PEGFILGRASTIM-APGF
df_rx_his_combined['generic_desc'] = df_rx_his_combined['generic_desc'].replace('PEGFILGRASTIM-APGF', 'PEGFILGRASTIM')
df_rx_his_combined['generic_desc'] = df_rx_his_combined['generic_desc'].replace('DEXAMETHASONE SODIUM PHOSPHATE', 'DEXAMETHASONE')

# COMMAND ----------

generic_cnt = df_rx_combined['generic_desc'].value_counts()
drug_cnt_combined = {}
for i, cnt in generic_cnt.items():
    drug_cnt_combined[i] = cnt
drug_cnt_combined = np.array([(key, int(value)) for key, value in drug_cnt_combined.items()])

# COMMAND ----------

sel_drug_list = []

for idx in range(len(drug_cnt_combined)):
    drug, drug_cnt = drug_cnt_combined[idx][0], int(drug_cnt_combined[idx][1])
    if drug_cnt > 100 and drug.lower() not in df_neoplastics['drug'].str.lower().values:
        print(drug, drug_cnt)
        treat_pid, control_pid = [], []
        treat_index_date, treat_therapy_date, treat_delta_date = [], [], []
        drug_rx_combined = df_rx_combined[df_rx_combined['generic_desc'] == drug]
        drug_rx_his_combined = df_rx_his_combined[df_rx_his_combined['generic_desc'] == drug]
        unique_drug_pat_ids = drug_rx_combined['ptid'].unique()
        for pat_id in df_cohort['ptid'].unique():
            if pat_id in unique_drug_pat_ids:
                pat_drug_rx_combined = drug_rx_combined[drug_rx_combined['ptid'] == pat_id]
                pat_drug_rx_his_combined = drug_rx_his_combined[drug_rx_his_combined['ptid'] == pat_id]
                if len(pat_drug_rx_combined) > 0:
                    sorted_pat_drug_rx_combined = pat_drug_rx_combined.sort_values(by=['RXDATE'])
                    sorted_pat_drug_rx_his_combined = pat_drug_rx_his_combined.sort_values(by=['RXDATE'])
                    if abs((sorted_pat_drug_rx_combined.iloc[0]['RXDATE'] - sorted_pat_drug_rx_combined.iloc[-1]['RXDATE']).days) > 0:
                        if len(sorted_pat_drug_rx_his_combined) > 0:
                            if abs((sorted_pat_drug_rx_combined.iloc[0]['RXDATE'] - sorted_pat_drug_rx_his_combined.iloc[-1]['RXDATE']).days) > 90:
                                # break
                                treat_pid.append(pat_id)
                                treat_index_date.append(sorted_pat_drug_rx_combined.iloc[0]['RXDATE'])
                                treat_therapy_date.append(sorted_pat_drug_rx_combined.iloc[0]['combined_therapy_date'])
                                treat_delta_date.append((sorted_pat_drug_rx_combined.iloc[0]['RXDATE'] - sorted_pat_drug_rx_combined.iloc[0]['combined_therapy_date']).days)
                            else:
                                # break
                                control_pid.append(pat_id)
                        else:
                            treat_pid.append(pat_id)
                            treat_index_date.append(sorted_pat_drug_rx_combined.iloc[0]['RXDATE'])
                            treat_therapy_date.append(sorted_pat_drug_rx_combined.iloc[0]['combined_therapy_date'])
                            treat_delta_date.append((sorted_pat_drug_rx_combined.iloc[0]['RXDATE'] - sorted_pat_drug_rx_combined.iloc[0]['combined_therapy_date']).days)
                    else:
                        # break
                        control_pid.append(pat_id)
                else:
                    control_pid.append(pat_id)
            else:
                control_pid.append(pat_id)

        print(len(treat_pid), len(control_pid), len(treat_pid) + len(control_pid))

        if len(treat_pid) > 100:
            os.makedirs("./exp_data/{}/{}/{}".format(cancer_type, therapy_version, drug.replace("/", "-")), exist_ok=True)
            df_treat = pd.DataFrame({'ptid': treat_pid, 'index_date': treat_index_date, 'combined_therapy_date': treat_therapy_date, 'delta_days': treat_delta_date})
            df_treat.to_csv("./exp_data/{}/{}/{}/treat.csv".format(cancer_type, therapy_version, drug.replace("/", "-")), index=False)
            for b_idx in range(100):
                # bootstrap_control_pid = np.random.choice(control_pid, size=len(treat_pid) * 1, replace=True)
                # bootstrap_control_delta_days = np.random.choice(treat_delta_date, size=len(treat_pid) * 1, replace=True)
                bootstrap_control_pid = np.random.choice(control_pid, size=len(treat_pid) * 5, replace=True)
                bootstrap_control_delta_days = np.random.choice(treat_delta_date, size=len(treat_pid) * 5, replace=True)
                # bootstrap_control_pid = np.random.choice(control_pid, size=len(df_cohort['ptid'].unique()), replace=True)
                # bootstrap_control_delta_days = np.random.choice(treat_delta_date, size=len(df_cohort['ptid'].unique()), replace=True)
                control_therapy_date = []
                for c_idx in range(len(bootstrap_control_pid)):
                    c_pid = bootstrap_control_pid[c_idx]
                    df_cohort_c_idx = df_cohort[df_cohort['ptid'] == c_pid]
                    control_therapy_date.append(df_cohort_c_idx.iloc[0]['combined_therapy_date'])
                df_control = pd.DataFrame({'ptid': bootstrap_control_pid, 'combined_therapy_date': control_therapy_date, 'delta_days': bootstrap_control_delta_days})
                df_control['index_date'] = pd.to_datetime(df_control['combined_therapy_date']) + pd.to_timedelta(df_control['delta_days'], unit='D')
                df_control.to_csv("./exp_data/{}/{}/{}/control_{}.csv".format(cancer_type, therapy_version, drug.replace("/", "-"), b_idx), index=False)
                
            sel_drug_list.append((drug, len(treat_pid)))
        # break

# COMMAND ----------

for item in sel_drug_list:
    print(item[0], item[1])

# COMMAND ----------

df_sel_drug_list = pd.DataFrame(data = np.array(sel_drug_list), columns = ['generic_desc', 'count'])
df_sel_drug_list.to_csv("./exp_data/{}/{}/sel_drug_list.csv".format(cancer_type, therapy_version), index=False)

# COMMAND ----------

df_sel_drug_list

# COMMAND ----------

print('Done.')

# COMMAND ----------

