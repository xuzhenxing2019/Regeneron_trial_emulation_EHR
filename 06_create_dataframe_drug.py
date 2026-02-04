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
from datetime import datetime

schema = "gd_prd.optum_onco_202502"
cancer_type = "ansclc"
therapy_version = "combined_therapy_pres_adm_pro_14_90"
import pickle

# COMMAND ----------

def convert_str_datetime(time_str):
    date_format = "%Y-%m-%d"
    return datetime.strptime(time_str, date_format)

# COMMAND ----------

dod_data = pd.read_csv("./exp_data/{}/{}/df_dod.csv".format(cancer_type, therapy_version))
df_sel_drug_list = pd.read_csv("./exp_data/{}/{}/sel_drug_list.csv".format(cancer_type, therapy_version))

# COMMAND ----------

for idx in range(len(df_sel_drug_list)):
    drug_name = df_sel_drug_list.iloc[idx]['generic_desc']
    if drug_name in ["PEMETREXED DISODIUM"]:
        continue
    data_dict = {}
    drug_key = drug_name.replace('/', '-')
    t_data = pd.read_csv('./exp_data/{}/{}/{}/treat_pat_data_all_v0.csv'.format(cancer_type, therapy_version, drug_key))
    print(t_data.shape)
    t_med_date = pd.read_csv('./exp_data/{}/{}/{}/treat.csv'.format(cancer_type, therapy_version, drug_key))
    t_med_date = t_med_date.merge(dod_data, on="ptid", how="left")
    t_med_date['available_date'] = pd.to_datetime(t_med_date['available_date'])
    t_med_date['last_act_dt'] = pd.to_datetime(t_med_date['last_act_dt'])
    t_med_date['index_date'] = pd.to_datetime(t_med_date['index_date'])

    d_delta_days = []
    for idx in range(len(t_med_date)):
        # print(idx)
        if t_med_date.iloc[idx]["d_flag"] == 1:
            d_delta_days.append((t_med_date.iloc[idx]['available_date'] - t_med_date.iloc[idx]['index_date']).days)
        else:
            d_delta_days.append((t_med_date.iloc[idx]['last_act_dt'] - t_med_date.iloc[idx]['index_date']).days)

    t_med_date['d_delta_days'] = d_delta_days
    t_med_date = t_med_date[['ptid', 'index_date', 'd_flag', 'd_delta_days']]
    t_med_date = t_med_date.drop_duplicates(subset=['ptid', 'index_date'])

    t_data['index_date'] = pd.to_datetime(t_data['index_date'])
    t_data = t_data.merge(t_med_date, on=["ptid", "index_date"], how="left")

    t_data['gender'] = t_data['gender'].replace('Male', 1)
    t_data['gender'] = t_data['gender'].replace('Female', 0)

    t_data['race'] = np.where(t_data['race'] == 'Caucasian', 1, t_data['race'])
    t_data['race'] = np.where(t_data['race'] != 1, 0, t_data['race'])
    t_data["treatment"] = np.ones(len(t_data)).astype(int)

    for control_idx in range(100):
        c_data = pd.read_csv('./exp_data/{}/{}/{}/control_{}_pat_data_all_v0.csv'.format(cancer_type, therapy_version, drug_key, control_idx))
        print("--------")
        print(c_data.shape)
        c_data = c_data[~c_data["ptid"].isin(t_data["ptid"])]

        c_med_date = pd.read_csv('./exp_data/{}/{}/{}/control_{}.csv'.format(cancer_type, therapy_version, drug_key, control_idx))
        c_med_date = c_med_date.merge(dod_data, on="ptid", how="left")
        c_med_date['available_date'] = pd.to_datetime(c_med_date['available_date'])
        c_med_date['last_act_dt'] = pd.to_datetime(c_med_date['last_act_dt'])
        c_med_date['index_date'] = pd.to_datetime(c_med_date['index_date'])

        d_delta_days = []
        for idx in range(len(c_med_date)):
            if c_med_date.iloc[idx]["d_flag"] == 1:
                d_delta_days.append((c_med_date.iloc[idx]['available_date'] - c_med_date.iloc[idx]['index_date']).days)
            else:
                d_delta_days.append((c_med_date.iloc[idx]['last_act_dt'] - c_med_date.iloc[idx]['index_date']).days)

        c_med_date['d_delta_days'] = d_delta_days
        c_med_date = c_med_date[['ptid', 'index_date', 'd_flag', 'd_delta_days']]
        c_med_date = c_med_date.drop_duplicates(subset=['ptid', 'index_date'])

        c_data['index_date'] = pd.to_datetime(c_data['index_date'])
        c_data = c_data.merge(c_med_date, on=["ptid", "index_date"], how="left")

        c_data['gender'] = c_data['gender'].replace('Male', 1)
        c_data['gender'] = c_data['gender'].replace('Female', 0)

        c_data['race'] = np.where(c_data['race'] == 'Caucasian', 1, c_data['race'])
        c_data['race'] = np.where(c_data['race'] != 1, 0, c_data['race'])
        c_data["treatment"] = np.zeros(len(c_data)).astype(int)

        all_data = pd.concat((t_data, c_data))
        # break

        all_data.to_csv('./exp_data/{}/{}/{}/all_data_v0_control_{}.csv'.format(cancer_type, therapy_version, drug_key, control_idx), index=False)
        # break

# COMMAND ----------

t_data

# COMMAND ----------

print('DONE.')

# COMMAND ----------

