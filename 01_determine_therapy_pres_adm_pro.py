# Databricks notebook source
# MAGIC %md
# MAGIC # Objective
# MAGIC
# MAGIC To extract aNSCLC cohort with chemo + ici
# MAGIC
# MAGIC Credit: Deep Hathi

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Libraries

# COMMAND ----------

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

schema = "gd_prd.optum_onco_202502"

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Tables

# COMMAND ----------

df_ansclc = spark.sql("SELECT * FROM gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort")
display(df_ansclc)
df_ansclc = df_ansclc.toPandas()
df_ansclc = df_ansclc.loc[:,~df_ansclc.columns.duplicated()].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC Load tables before and after diagnoisis date and combine (to avoid memory issue)

# COMMAND ----------

df_rx_presc = spark.sql(
    """
    SELECT *
    FROM gd_prd.optum_onco_202502.rx_presc rx
    INNER JOIN (
        SELECT ptid, initial_amel_dx_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort) cohort
    ON rx.ptid = cohort.ptid AND rx.RXDATE >= cohort.initial_amel_dx_date AND ABS(DATEDIFF(rx.RXDATE, cohort.initial_amel_dx_date)) <= (90)
    """)

df_rx_adm = spark.sql(
    """
    SELECT *
    FROM gd_prd.optum_onco_202502.rx_adm rx
    INNER JOIN (
        SELECT ptid, initial_amel_dx_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort) cohort
    ON rx.ptid = cohort.ptid AND rx.ADMIN_DATE >= cohort.initial_amel_dx_date AND ABS(DATEDIFF(rx.ADMIN_DATE, cohort.initial_amel_dx_date)) <= (90) 
    """)

pd_df_rx_presc = df_rx_presc.toPandas()
pd_df_rx_presc = pd_df_rx_presc.loc[:,~pd_df_rx_presc.columns.duplicated()].copy()

pd_df_rx_adm = df_rx_adm.toPandas()
pd_df_rx_adm = pd_df_rx_adm.loc[:,~pd_df_rx_adm.columns.duplicated()].copy()

# COMMAND ----------

df_rx_presc_his = spark.sql(
    """
    SELECT *
    FROM gd_prd.optum_onco_202502.rx_presc rx
    INNER JOIN (
        SELECT ptid, initial_amel_dx_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort) cohort
    ON rx.ptid = cohort.ptid AND rx.RXDATE < cohort.initial_amel_dx_date AND ABS(DATEDIFF(rx.RXDATE, cohort.initial_amel_dx_date)) <= (14)
    """)

df_rx_adm_his = spark.sql(
    """
    SELECT *
    FROM gd_prd.optum_onco_202502.rx_adm rx
    INNER JOIN (
        SELECT ptid, initial_amel_dx_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort) cohort
    ON rx.ptid = cohort.ptid AND rx.ADMIN_DATE < cohort.initial_amel_dx_date AND ABS(DATEDIFF(rx.ADMIN_DATE, cohort.initial_amel_dx_date)) <= (14) 
    """)

pd_df_rx_presc_his = df_rx_presc_his.toPandas()
pd_df_rx_presc_his = pd_df_rx_presc_his.loc[:,~pd_df_rx_presc_his.columns.duplicated()].copy()

pd_df_rx_adm_his = df_rx_adm_his.toPandas()
pd_df_rx_adm_his = pd_df_rx_adm_his.loc[:,~pd_df_rx_adm_his.columns.duplicated()].copy()

# COMMAND ----------

pd_df_rx_presc = pd.concat((pd_df_rx_presc, pd_df_rx_presc_his))
pd_df_rx_adm = pd.concat((pd_df_rx_adm, pd_df_rx_adm_his))

# COMMAND ----------

df_neoplastics = spark.sql("SELECT * FROM gd_prd.dh_rwdna_ml.nsclc_neoplastics_ndc_cpt_hcpcs")
display(df_neoplastics)
df_neoplastics = df_neoplastics.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # Map NDC Codes for Chemo and ICI

# COMMAND ----------

# df_neoplastics[df_neoplastics['nsclc_therapy_class'] == 'immunotherapy']['generic_name'].unique()

ici_list = ['cemiplimab','durvalumab','nivolumab','pembrolizumab','ipilimumab','atezolizumab']

chemo_list = ['etoposide', 'methotrexate', 'doxorubicin', 'carboplatin', 'docetaxel',
              'gemcitabine', 'paclitaxel', 'pemetrexed', 'vinorelbine',
              'doxorubicin', 'oxaliplatin', 'gemcitabine', 'etoposide',
              'methotrexate']

def construct_regex_str(l: list[str], wrap: bool = True, end_drug: str | list[str] = None) -> str:
    if isinstance(end_drug, str):
        end_drug = [end_drug]
    elif end_drug is None:
        end_drug = []

    ptn = r"|".join([
        f"{c}"
        if (c not in end_drug) else rf"{c}$"
        for c in l])
    if wrap:
        return "(" + ptn + ")"
    return ptn


ici_drug_str = construct_regex_str(ici_list, end_drug=["nivolumab", "ipilimumab"])
chemo_drug_list = construct_regex_str(chemo_list, end_drug=[])

# COMMAND ----------

ndc_drug_mapping = spark.sql(
    f"""
    SELECT  *,
            regexp_extract(generic_name, r'{ici_drug_str}', 1) AS drug
    FROM gd_prd.dh_rwdna_ml.all_ndc_fields_dictionary
    WHERE generic_name RLIKE '{ici_drug_str}'
    """
)

# CPT codes
cpt_srch = construct_regex_str(ici_list)
cpt_drug_mapping = spark.sql(
    f"""
    SELECT  *,
            regexp_extract(lower(Short_Descriptor), r'{cpt_srch}', 1) AS drug
    FROM gd_prd.aapc.cpt_hcpcs_20241231
    WHERE LOWER(Short_Descriptor) RLIKE '{cpt_srch}'
    """
)

# join on drugs
ici_drug_mapping = ndc_drug_mapping.join(cpt_drug_mapping, on="drug", how="outer").dropDuplicates()
ici_drug_mapping = ici_drug_mapping.toPandas()

# COMMAND ----------

ndc_drug_mapping = spark.sql(
    f"""
    SELECT  *,
            regexp_extract(generic_name, r'{chemo_drug_list}', 1) AS drug
    FROM gd_prd.dh_rwdna_ml.all_ndc_fields_dictionary
    WHERE generic_name RLIKE '{chemo_drug_list}'
    """
)

# CPT codes
cpt_srch = construct_regex_str(chemo_list)
cpt_drug_mapping = spark.sql(
    f"""
    SELECT  *,
            regexp_extract(lower(Short_Descriptor), r'{cpt_srch}', 1) AS drug
    FROM gd_prd.aapc.cpt_hcpcs_20241231
    WHERE LOWER(Short_Descriptor) RLIKE '{cpt_srch}'
    """
)

# join on drugs
chemo_drug_mapping = ndc_drug_mapping.join(cpt_drug_mapping, on="drug", how="outer").dropDuplicates()
chemo_drug_mapping = chemo_drug_mapping.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # Find Records of Chemo and ICI

# COMMAND ----------

# rx_adm_chemo = pd_df_rx_adm[pd_df_rx_adm['generic_desc'].str.contains("|".join(chemo_list), case=False, na=False)]
# rx_adm_ici = pd_df_rx_adm[pd_df_rx_adm['generic_desc'].str.contains("|".join(ici_list), case=False, na=False)]

# rx_presc_chemo = pd_df_rx_presc[pd_df_rx_presc['generic_desc'].str.contains("|".join(chemo_list), case=False, na=False)]
# rx_presc_ici = pd_df_rx_presc[pd_df_rx_presc['generic_desc'].str.contains("|".join(ici_list), case=False, na=False)]

rx_adm_chemo = pd_df_rx_adm[pd_df_rx_adm['ndc'].isin(chemo_drug_mapping['NDC'])]
rx_adm_ici = pd_df_rx_adm[pd_df_rx_adm['ndc'].isin(ici_drug_mapping['NDC'])]

rx_presc_chemo = pd_df_rx_presc[pd_df_rx_presc['ndc'].isin(chemo_drug_mapping['NDC'])]
rx_presc_ici = pd_df_rx_presc[pd_df_rx_presc['ndc'].isin(ici_drug_mapping['NDC'])]

# COMMAND ----------

rx_adm_clippped = rx_adm_chemo[['ptid', 'ADMIN_DATE', 'initial_amel_dx_date', 'generic_desc']]
rx_adm_clippped.rename(columns={'ADMIN_DATE': 'RXDATE'}, inplace=True)
rx_adm_clippped['Type'] = "adm"

rx_presc_clipped = rx_presc_chemo[['ptid', 'RXDATE', 'initial_amel_dx_date','generic_desc']]
rx_presc_clipped['Type'] = "pres"
rx_combined_chemo = pd.concat([rx_adm_clippped, rx_presc_clipped])

rx_adm_clippped = rx_adm_ici[['ptid', 'ADMIN_DATE', 'initial_amel_dx_date', 'generic_desc']]
rx_adm_clippped.rename(columns={'ADMIN_DATE': 'RXDATE'}, inplace=True)
rx_adm_clippped['Type'] = "adm"

rx_presc_clipped = rx_presc_ici[['ptid', 'RXDATE', 'initial_amel_dx_date','generic_desc']]
rx_presc_clipped['Type'] = "pres"
rx_combined_ici = pd.concat([rx_adm_clippped, rx_presc_clipped])

# COMMAND ----------

# MAGIC %md
# MAGIC # Determine Patients with Chemo and ICI

# COMMAND ----------

combined_therapy = []
combined_therapy_date = []
for idx in range(len(df_ansclc)):
    ptid = df_ansclc.iloc[idx]['ptid']
    pt_ici = rx_combined_ici.loc[rx_combined_ici['ptid'] == ptid]
    pt_chemo = rx_combined_chemo.loc[rx_combined_chemo['ptid'] == ptid]

    if len(pt_ici) > 0 and len(pt_chemo) > 0:
        combined_therapy.append(1)
        combined_therapy_date.append(min(pt_chemo['RXDATE'].min(), pt_ici['RXDATE'].min()))
    else:
        combined_therapy.append(0)
        combined_therapy_date.append(np.nan)
    
    if len(combined_therapy) % 1000 == 0:
        print(len(combined_therapy), np.array(combined_therapy).mean())
df_ansclc['combined_therapy'] = combined_therapy
df_ansclc['combined_therapy_date'] = combined_therapy_date
df_ansclc_sel = df_ansclc[df_ansclc['combined_therapy'] == 1]

# COMMAND ----------

df_ansclc_sel

# COMMAND ----------

# (
#     spark.createDataFrame(df_ansclc_sel)
#     .write
#     .format("delta")
#     .mode("overwrite")
#     .option("overwriteSchema", "true")
#     .saveAsTable("gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort_combined_therapy_pres_adm_14_90")
# )

# COMMAND ----------

# MAGIC %md
# MAGIC # integrate the procedure information with pres and adm

# COMMAND ----------

# MAGIC %md
# MAGIC ## step 1: load pres_adm
# MAGIC ## step 2: load pro
# MAGIC ## step 3: combine
# MAGIC ## step 4: save 

# COMMAND ----------

# step 1: load optum_202502_ansclc_cohort_combined_therapy_pres_adm_14_90
df_pres_adm_testing = spark.sql(
    """
    SELECT *
    FROM gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort_combined_therapy_pres_adm_14_90
    """)
df_pres_adm_testing = df_pres_adm_testing.toPandas()
df_pres_adm_testing = df_pres_adm_testing.loc[:,~df_pres_adm_testing.columns.duplicated()].copy()
df_pres_adm_testing = df_pres_adm_testing.drop_duplicates(subset=['ptid'])
display(df_pres_adm_testing)

# COMMAND ----------

# step 2: load optum_202502_ansclc_cohort_combined_therapy_lot_all_14_90

df_pro_testing = spark.sql(
    """
    SELECT *
    FROM gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort_combined_therapy_lot_all_14_90
    """)
df_pro_testing = df_pro_testing.toPandas()
df_pro_testing = df_pro_testing.loc[:,~df_pro_testing.columns.duplicated()].copy()
df_pro_testing = df_pro_testing.drop_duplicates(subset=['ptid'])
display(df_pro_testing)

# COMMAND ----------

# step 3: combine the pres_adm and pro
# choose patients from df_pro_testing if there are no records in the pres_adm and then add them to the pres_adm

df_pres_adm_pro = pd.concat([df_pres_adm_testing, df_pro_testing.loc[~df_pro_testing['ptid'].isin(df_pres_adm_testing['ptid'])]])
df_pres_adm_pro = df_pres_adm_pro.drop_duplicates(subset=['ptid'])  
df_pres_adm_pro = df_pres_adm_pro.reset_index(drop=True)
display(df_pres_adm_pro)

# COMMAND ----------

# step 4:  save the pres_adm_pro
# (
#     spark.createDataFrame(df_pres_adm_pro)
#     .write
#     .format("delta")
#     .mode("overwrite")
#     .option("overwriteSchema", "true")
#     .saveAsTable("gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort_combined_therapy_pres_adm_pro_14_90")
# )

# COMMAND ----------

print('DONE.')