# Databricks notebook source
import re
from functools import reduce
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import pickle
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

def concat_data(drug_name, prefix):
    drug_key = drug_name.replace('/', '-')
    
    pd_data = pd.read_csv('./exp_data/{}/{}/{}/{}_pat_data_basic_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix))
    pd_metastatic = pd.read_csv('./exp_data/{}/{}/{}/{}_pat_data_metastatic_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix))
    pd_vital = pd.read_csv('./exp_data/{}/{}/{}/{}_pat_data_vital_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix))
    pd_lab = pd.read_csv('./exp_data/{}/{}/{}/{}_pat_data_lab_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix))
    pd_med = pd.read_csv('./exp_data/{}/{}/{}/{}_pat_data_med_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix))
    pd_com = pd.read_csv('./exp_data/{}/{}/{}/{}_pat_data_com_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix))

    pd_metastatic = pd_metastatic.drop_duplicates(subset=['ptid', 'index_date'])
    pd_vital = pd_vital.drop_duplicates(subset=['ptid', 'index_date'])
    pd_lab = pd_lab.drop_duplicates(subset=['ptid', 'index_date'])
    pd_med = pd_med.drop_duplicates(subset=['ptid', 'index_date'])
    pd_com = pd_com.drop_duplicates(subset=['ptid', 'index_date'])

    print(pd_data.shape)
    pd_data = pd_data.merge(pd_metastatic, how='left', on=['ptid', 'index_date'])
    print(pd_data.shape)
    pd_data = pd_data.merge(pd_vital, how='left', on=['ptid', 'index_date'])
    print(pd_data.shape)
    pd_data = pd_data.merge(pd_lab, how='left', on=['ptid', 'index_date'])
    print(pd_data.shape)
    pd_data = pd_data.merge(pd_med, how='left', on=['ptid', 'index_date'])
    print(pd_data.shape)
    pd_data = pd_data.merge(pd_com, how='left', on=['ptid', 'index_date'])
    print(pd_data.shape)

    pd_data.to_csv('./exp_data/{}/{}/{}/{}_pat_data_all_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix), index=False)

# COMMAND ----------

df_sel_drug_list = pd.read_csv("./exp_data/{}/{}/sel_drug_list.csv".format(cancer_type, therapy_version))

# COMMAND ----------

for idx in range(len(df_sel_drug_list)):
    drug_name = df_sel_drug_list.iloc[idx]['generic_desc']
    if drug_name in ["PEMETREXED DISODIUM"]:
        continue
    
    print("Work on: {}, {}".format(drug_name, "treat"))
    concat_data(drug_name, "treat")

    for b_idx in range(100):
        print("Work on: {}, {}, {}".format(drug_name, "control", b_idx))
        concat_data(drug_name, "control_{}".format(b_idx))
 