# Databricks notebook source
import re
from functools import reduce
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    coalesce, col, collect_list, count, countDistinct, datediff, expr,
    first, least, lower, lit, when)
from pyspark.sql.functions import min as ps_min, max as ps_max
from datetime import timedelta
schema = "gd_prd.optum_onco_202502"
cancer_type = "ansclc"
therapy_version = "combined_therapy_pres_adm_pro_14_90"

# COMMAND ----------

df_dod = spark.sql(
    """
    SELECT dod.ptid, dod.unvalidated_dod, dod.last_act_dt, cohort.initial_amel_dx_date, cohort.combined_therapy_date
    FROM gd_prd.heor_query.optum_onco_202502_dod dod
    INNER JOIN (
        SELECT ptid, initial_amel_dx_date, combined_therapy_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_{}_cohort_{}) cohort
    ON dod.ptid = cohort.ptid
    """.format(cancer_type, therapy_version))
df_dod = df_dod.toPandas()
df_dod = df_dod.loc[:,~df_dod.columns.duplicated()].copy()

# COMMAND ----------

df_dod

# COMMAND ----------

# this is old version
# (
#     spark.createDataFrame(df_dod)
#     .write
#     .format("delta")
#     .mode("overwrite")
#     .option("overwriteSchema", "true")
#     .saveAsTable("gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort_tmp_dod")
# )

# COMMAND ----------

# this is new version, considering procedure during building cohort
#(
#     spark.createDataFrame(df_dod)
#     .write
#     .format("delta")
#     .mode("overwrite")
#     .option("overwriteSchema", "true")
#     .saveAsTable("gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort_tmp_dod_pro")
#)

# COMMAND ----------

df_rx_presc = spark.sql(
    """
    SELECT *
    FROM gd_prd.optum_onco_202502.rx_presc rx
    INNER JOIN (
        SELECT ptid, unvalidated_dod, initial_amel_dx_date, combined_therapy_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort_tmp_dod_pro) cohort
    ON rx.ptid = cohort.ptid AND rx.RXDATE >= cohort.initial_amel_dx_date AND rx.RXDATE >= cohort.combined_therapy_date
    """)

df_proc = spark.sql(
    """
    SELECT *
    FROM gd_prd.optum_onco_202502.proc pr
    INNER JOIN (
        SELECT ptid, unvalidated_dod, initial_amel_dx_date, combined_therapy_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort_tmp_dod_pro) cohort
    ON pr.ptid = cohort.ptid AND pr.PROC_DATE >= cohort.initial_amel_dx_date AND pr.PROC_DATE >= cohort.combined_therapy_date
    """)

df_rx_adm = spark.sql(
    """
    SELECT *
    FROM gd_prd.optum_onco_202502.rx_adm rx
    INNER JOIN (
        SELECT ptid, unvalidated_dod, initial_amel_dx_date, combined_therapy_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort_tmp_dod_pro) cohort
    ON rx.ptid = cohort.ptid AND rx.ADMIN_DATE >= cohort.initial_amel_dx_date AND rx.ADMIN_DATE >= cohort.combined_therapy_date
    """)

df_lab = spark.sql(
    """
    SELECT *
    FROM gd_prd.optum_onco_202502.lab lab
    INNER JOIN (
        SELECT ptid, unvalidated_dod, initial_amel_dx_date, combined_therapy_date
        FROM gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort_tmp_dod_pro) cohort
    ON lab.ptid = cohort.ptid AND lab.RESULT_DATE >= cohort.initial_amel_dx_date AND lab.RESULT_DATE >= cohort.combined_therapy_date
    """)

# COMMAND ----------

df_dod

# COMMAND ----------

pd_df_rx_presc = df_rx_presc.toPandas()
pd_df_rx_presc = pd_df_rx_presc.loc[:,~pd_df_rx_presc.columns.duplicated()].copy()

pd_df_rx_adm = df_rx_adm.toPandas()
pd_df_rx_adm = pd_df_rx_adm.loc[:,~pd_df_rx_adm.columns.duplicated()].copy()

pd_df_proc = df_proc.toPandas()
pd_df_proc = pd_df_proc.loc[:,~pd_df_proc.columns.duplicated()].copy()

pd_df_lab = df_lab.toPandas()
pd_df_lab = pd_df_lab.loc[:,~pd_df_lab.columns.duplicated()].copy()

# COMMAND ----------

available_date, new_last_act_dt = [], []
cnt = 0
for idx in range(len(df_dod)):
    tmp_ptid = df_dod.iloc[idx]['ptid']
    tmp_last_act_dt = max(df_dod.iloc[idx]['last_act_dt'], df_dod.iloc[idx]['initial_amel_dx_date'],
    df_dod.iloc[idx]['combined_therapy_date'])
    tmp_unvalidated_dod = df_dod.iloc[idx]['unvalidated_dod']

    pt_rx_presc = pd_df_rx_presc[pd_df_rx_presc['ptid'] == tmp_ptid]
    pt_rx_adm = pd_df_rx_adm[pd_df_rx_adm['ptid'] == tmp_ptid]
    pt_proc = pd_df_proc[pd_df_proc['ptid'] == tmp_ptid]
    pt_lab = pd_df_lab[pd_df_lab['ptid'] == tmp_ptid]

    if len(pt_rx_presc) + len(pt_rx_adm) + len(pt_lab) + len(pt_proc) > 0:
        if len(pt_rx_presc) > 0:
            tmp_last_act_dt = max(tmp_last_act_dt, pt_rx_presc['RXDATE'].max())
        if len(pt_rx_adm) > 0:
            tmp_last_act_dt = max(tmp_last_act_dt, pt_rx_adm['ADMIN_DATE'].max())
        if len(pt_lab) > 0:
            tmp_last_act_dt = max(tmp_last_act_dt, pt_lab['RESULT_DATE'].max())
        if len(pt_proc) > 0:
            tmp_last_act_dt = max(tmp_last_act_dt, pt_proc['PROC_DATE'].max())

    new_last_act_dt.append(tmp_last_act_dt)
    if tmp_unvalidated_dod is not None and tmp_last_act_dt >= tmp_unvalidated_dod + timedelta(days=30):
        available_date.append(None)
        print(idx, tmp_ptid)
        cnt += 1
    else:
        available_date.append(tmp_unvalidated_dod)

    #     last_act_date.append(tmp_last_act_dt)
    #     cnt += 1
    # else:
    #     available_date.append(df_dod.iloc[idx]['unvalidated_dod'])
    #     last_act_date.append(None)

# COMMAND ----------

available_date

# COMMAND ----------

event_list = []
for idx in range(len(df_dod)):
    if available_date[idx] is not None:
        event_list.append(1)
    else:
        event_list.append(0)

# COMMAND ----------

df_dod

# COMMAND ----------

df_dod.insert(5, "available_date", available_date)
df_dod.insert(6, "d_flag", event_list)
df_dod['last_act_dt'] = new_last_act_dt

# COMMAND ----------

df_dod.to_csv("./exp_data/{}/{}/df_dod.csv".format(cancer_type, therapy_version))

# COMMAND ----------

df_dod

# COMMAND ----------

print('DONE.')

# COMMAND ----------

