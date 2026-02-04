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

with open('./exp_data/{}/{}/pat_pt_clinical_dict.pkl'.format(cancer_type, therapy_version), "rb") as f:
    pat_pt_clinical_dict = pickle.load(f) 

with open('./exp_data/{}/{}/pat_vital_dict.pkl'.format(cancer_type, therapy_version), "rb") as f:
    pat_vital_dict = pickle.load(f) 

with open('./exp_data/{}/{}/pat_lab_dict.pkl'.format(cancer_type, therapy_version), "rb") as f:
    pat_lab_dict = pickle.load(f) 

with open('./exp_data/{}/{}/pat_meta_loc_dict.pkl'.format(cancer_type, therapy_version), "rb") as f:
    pat_meta_loc_dict = pickle.load(f) 

with open('./exp_data/{}/{}/pat_diag_dict.pkl'.format(cancer_type, therapy_version), "rb") as f:
    pat_diag_dict = pickle.load(f) 

with open('./exp_data/{}/{}/pat_med_dict.pkl'.format(cancer_type, therapy_version), "rb") as f:
    pat_med_dict = pickle.load(f) 

# COMMAND ----------

with open('./exp_data/{}/{}/med_list.json'.format(cancer_type, therapy_version), 'r') as f:
    sel_med = json.load(f)

with open('./exp_data/{}/{}/com_list.json'.format(cancer_type, therapy_version), 'r') as f:
    sel_com = json.load(f)

# COMMAND ----------

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# COMMAND ----------

df_sel_drug_list = pd.read_csv("./exp_data/{}/{}/sel_drug_list.csv".format(cancer_type, therapy_version))

# COMMAND ----------

df_sel_drug_list

# COMMAND ----------

# for drug_name in ['MORPHINE SULFATE','DEXAMETHASONE SODIUM PHOSPHATE','PALONOSETRON HCL','ONDANSETRON HCL','OXYCODONE HCL','ACETAMINOPHEN','PANTOPRAZOLE SODIUM','FOSAPREPITANT DIMEGLUMINE','HEPARIN SODIUM;PORCINE/PF','DEXAMETHASONE','POTASSIUM CHLORIDE','FAMOTIDINE','ENOXAPARIN SODIUM','DIPHENHYDRAMINE HCL','LORAZEPAM','CYANOCOBALAMIN (VITAMIN B-12)','FUROSEMIDE','MAGNESIUM SULFATE IN STERILE WATER','PREDNISONE','PEGFILGRASTIM','PACLITAXEL PROTEIN-BOUND']:
#     print(drug_name)
#     for b_idx in range(100):
#         print(b_idx)
#         drug_key = drug_name.replace('/', '-')
#         pd_pat = pd.read_csv('./exp_data/aNSCLC/{}/control_{}.csv'.format(drug_key, b_idx))
#         if 'index_date' not in pd_pat.columns:
#             pd_pat['index_date'] = pd.to_datetime(pd_pat['combined_therapy_date']) + pd.to_timedelta(pd_pat['delta_days'], unit='D')
#             pd_pat.to_csv('./exp_data/aNSCLC/{}/control_{}.csv'.format(drug_key, b_idx), index=False)

# COMMAND ----------

def preprocess_data(drug_name, prefix):
    drug_key = drug_name.replace('/', '-')
    pd_pat = pd.read_csv('./exp_data/{}/{}/{}/{}.csv'.format(cancer_type, therapy_version, drug_key, prefix))
    pd_pat['index_date'] = pd.to_datetime(pd_pat['index_date'])
    pd_pat['index_date'] = pd_pat['index_date'].dt.date

    print("Basic")

    demog_list = []
    for idx in range(len(pd_pat)):
        ptid = pd_pat.iloc[idx]['ptid']
        index_date = pd_pat.iloc[idx]['index_date']
        pat_pt_clinical = pat_pt_clinical_dict.get(ptid, {})
        pat_pt_clinical_list = [ptid, index_date]
        # Make sure 'birth_yr' exist and always be the last one
        for pt_clinical_name in ['gender', 'race', 'ecog', 'birth_yr']:
            pat_pt_clinical_list.append(pat_pt_clinical[pt_clinical_name])

        if pat_pt_clinical_list[-1] == "1935 and Earlier":
            pat_pt_clinical_list.append(index_date.year - 1935)
        else:
            pat_pt_clinical_list.append(index_date.year - int(pat_pt_clinical_list[-1]))
        demog_list.append(pat_pt_clinical_list)

    demog_list = np.array(demog_list)
    pd_data = pd.DataFrame(demog_list, columns=['ptid', 'index_date', 'gender', 'race', 'ecog', 'birth_year', 'age_at_index'])
    pd_data = pd_data[['ptid', 'index_date', 'gender', 'race', 'ecog', 'age_at_index']]

    print("Meta")

    metastastic_list = []
    for idx in range(len(pd_pat)):
        ptid = pd_pat.iloc[idx]['ptid']
        index_date = pd_pat.iloc[idx]['index_date']
        obs_date = index_date - timedelta(days=365)
        pat_onc_metastatic_location = pat_meta_loc_dict.get(ptid, [])

        if len(pat_onc_metastatic_location) == 0:
            pat_metastastic_list = [ptid, index_date, 0, 0, 0, 0, 0, 0]
        else:
            unique_metastatic_loc = set()
            for item in pat_onc_metastatic_location:
                if item[1] > index_date:
                    break
                else:
                    unique_metastatic_loc.add(item[2])
            pat_metastastic_list = [ptid, index_date, len(unique_metastatic_loc)]
            for loc in ['brain', 'bone', 'liver', 'lung', 'adrenal gland']:
                if loc in unique_metastatic_loc:
                    pat_metastastic_list.append(1)
                else:
                    pat_metastastic_list.append(0)
        metastastic_list.append(pat_metastastic_list)
    # pd_df_onc_metastatic_location

    metastastic_list = np.array(metastastic_list)
    pd_metastatic = pd.DataFrame(metastastic_list, columns=['ptid', 'index_date'] + ['meta_total', 'meta_brain', 'meta_bone', 'meta_liver', 'meta_lung', 'meta_adrenal_gland'])

    print("Vital")

    vital_name_list = ['SBP', 'DBP', 'WT', 'HT', 'TEMP', 'PULSE', 'RESP', 'PAIN', 'HR', 'UROUT', 'PACK_YEARS']
    vital_list = []
    for idx in range(len(pd_pat)):
        ptid = pd_pat.iloc[idx]['ptid']
        index_date = pd_pat.iloc[idx]['index_date']
        obs_date = index_date - timedelta(days=365)
        pat_vital = pat_vital_dict[ptid]
        pat_vital_list = [ptid, index_date]
        for vital_name in vital_name_list:
            tmp_results = []
            pat_vital_tmp = pat_vital.get(vital_name, [])
            for item in pat_vital_tmp:
                if item[0] < obs_date:
                    continue
                elif item[0] > index_date:
                    break
                else:
                    tmp_results.append(float(item[1]))
            if len(tmp_results) > 0:
                ave_result = np.array(tmp_results).mean()
            else:
                ave_result = None
            pat_vital_list.append(ave_result)
        vital_list.append(pat_vital_list)

    vital_list = np.array(vital_list)
    pd_vital = pd.DataFrame(vital_list, columns=['ptid', 'index_date'] + ["vital_" + item for item in vital_name_list])

    print("Lab")

    lab_name_list = ['hemoglobin','platelets','hematocrit','serum_creatinine','calcium','albumin','blood_glucose','blood_urea_nitrogen','blood_bilirubin_total','blood_erythrocytes','blood_wbc','neutrophils_percent','lymphocytes_percent','monocytes_percent','eosinophils_percent','basophils_percent','ast','alt','potassium','chloride','sodium','alkaline_phosphatase','carbon_dioxide','protein']
    lab_list = []

    for idx in range(len(pd_pat)):
        ptid = pd_pat.iloc[idx]['ptid']
        index_date = pd_pat.iloc[idx]['index_date']
        obs_date = index_date - timedelta(days=365)
        pat_lab = pat_lab_dict.get(ptid, {})
        pat_lab_list = [ptid, index_date]
        for lab_name in lab_name_list:
            tmp_results = []
            pat_lab_tmp = pat_lab.get(lab_name, [])
            for item in pat_lab_tmp:
                if item[0] < obs_date:
                    continue
                elif item[0] > index_date:
                    break
                else:
                    if is_number(item[1]):
                        tmp_results.append(float(item[1]))

            if len(tmp_results) > 0:
                last_result = tmp_results[-1]
            else:
                last_result = None
            pat_lab_list.append(last_result)
            # break
        # break
        lab_list.append(pat_lab_list)

    lab_list = np.array(lab_list)
    pd_lab = pd.DataFrame(lab_list, columns=['ptid', 'index_date'] + ["lab_" + item for item in lab_name_list])


    print("Med")

    med_name_list = sel_med
    med_list = []
    for idx in range(len(pd_pat)):
        ptid = pd_pat.iloc[idx]['ptid']
        index_date = pd_pat.iloc[idx]['index_date']
        obs_date = index_date - timedelta(days=365)
        pat_med = pat_med_dict.get(ptid, {})
        pat_med_list = [ptid, index_date]
        for med_name in med_name_list:
            pat_med_tmp = pat_med.get(med_name, [])
            med_flag = 0
            for item in pat_med_tmp:
                if item < obs_date:
                    continue
                elif item > index_date:
                    break
                else:
                    med_flag = 1
                    break

            pat_med_list.append(med_flag)
        med_list.append(pat_med_list)

    med_list = np.array(med_list)
    pd_med = pd.DataFrame(med_list, columns=['ptid', 'index_date'] + ["med_" + item for item in med_name_list])

    print("Com")
    com_name_list = sel_com
    com_list = []
    for idx in range(len(pd_pat)):
        ptid = pd_pat.iloc[idx]['ptid']
        index_date = pd_pat.iloc[idx]['index_date']
        obs_date = index_date - timedelta(days=365)
        pat_diag = pat_diag_dict.get(ptid, {})
        pat_com_list = [ptid, index_date]
        for com_name in com_name_list:
            pat_diag_tmp = pat_diag.get(com_name, [])
            com_flag = 0
            for item in pat_diag_tmp:
                if item[0] < obs_date:
                    continue
                elif item[0] > index_date:
                    break
                else:
                    com_flag = 1
                    break

            pat_com_list.append(com_flag)
        com_list.append(pat_com_list)

    com_list = np.array(com_list)
    pd_com = pd.DataFrame(com_list, columns=['ptid', 'index_date'] + ["com_" + item for item in com_name_list])

    pd_data.to_csv('./exp_data/{}/{}/{}/{}_pat_data_basic_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix), index=False)
    pd_metastatic.to_csv('./exp_data/{}/{}/{}/{}_pat_data_metastatic_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix), index=False)
    pd_vital.to_csv('./exp_data/{}/{}/{}/{}_pat_data_vital_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix), index=False)
    pd_lab.to_csv('./exp_data/{}/{}/{}/{}_pat_data_lab_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix), index=False)
    pd_med.to_csv('./exp_data/{}/{}/{}/{}_pat_data_med_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix), index=False)
    pd_com.to_csv('./exp_data/{}/{}/{}/{}_pat_data_com_v0.csv'.format(cancer_type, therapy_version, drug_key, prefix), index=False)

# COMMAND ----------

# for drug_name in ['MORPHINE SULFATE','DEXAMETHASONE SODIUM PHOSPHATE','PALONOSETRON HCL','ONDANSETRON HCL','OXYCODONE HCL','ACETAMINOPHEN','PANTOPRAZOLE SODIUM','FOSAPREPITANT DIMEGLUMINE','HEPARIN SODIUM;PORCINE/PF','DEXAMETHASONE','POTASSIUM CHLORIDE','FAMOTIDINE','ENOXAPARIN SODIUM','DIPHENHYDRAMINE HCL','LORAZEPAM','CYANOCOBALAMIN (VITAMIN B-12)','FUROSEMIDE','MAGNESIUM SULFATE IN STERILE WATER','PREDNISONE','PEGFILGRASTIM','PACLITAXEL PROTEIN-BOUND']:
# for drug_name in ['MORPHINE SULFATE', 'DEXAMETHASONE SODIUM PHOSPHATE', 'PALONOSETRON HCL']:
for idx in range(len(df_sel_drug_list)):
    drug_name = df_sel_drug_list.iloc[idx]['generic_desc']
    if drug_name in ["PEMETREXED DISODIUM"]:
        continue
    
    print("Work on: {}, {}".format(drug_name, "treat"))
    preprocess_data(drug_name, "treat")

    for b_idx in range(100):
        print("Work on: {}, {}, {}".format(drug_name, "control", b_idx))
        preprocess_data(drug_name, "control_{}".format(b_idx))
 

# COMMAND ----------

# drug_name = 'PALONOSETRON HCL'
# prefix = 'control_0'

# drug_key = drug_name.replace('/', '-')
# pd_pat = pd.read_csv('./exp_data/aNSCLC/{}/{}.csv'.format(drug_key, prefix))
# pd_pat['index_date'] = pd.to_datetime(pd_pat['index_date'])
# pd_pat['index_date'] = pd_pat['index_date'].dt.date

# print("Basic")

# pd_df_pt_clinical = df_pt_clinical.toPandas()
# pd_df_pt_clinical = pd_df_pt_clinical.loc[:,~pd_df_pt_clinical.columns.duplicated()].copy()

# demog = pd_df_pt_clinical[pd_df_pt_clinical['ptid'].isin(pd_pat['ptid'])].copy()
# demog = pd.merge(pd_pat, demog, on='ptid')

# age_list = []
# for idx in range(len(demog)):
#     if demog.iloc[idx]['birth_yr'] == "1935 and Earlier":
#         age_list.append(demog.iloc[idx]['index_date'].year - 1935)
#     else:
#         age_list.append(demog.iloc[idx]['index_date'].year - int(demog.iloc[idx]['birth_yr']))
# demog['age_at_index'] = age_list

# df_cohort = spark.sql("SELECT * FROM gd_prd.dh_rwdna_ml.optum_202502_ansclc_cohort_{}".format(therapy_version))
# pd_df_cohort = df_cohort.toPandas()
# ecog = pd_df_cohort[pd_df_cohort['ptid'].isin(pd_pat['ptid'])].copy()

# pd_data_2 = demog[['ptid', 'index_date', 'gender', 'race', 'age_at_index']]
# pd_ecog = ecog[['ptid', 'ecog']]
# pd_data_2 = pd_data_2.merge(pd_ecog, how='left', on='ptid')

# COMMAND ----------

# drug_name = 'PALONOSETRON HCL'
# prefix = 'control_0'

# drug_key = drug_name.replace('/', '-')
# pd_pat = pd.read_csv('./exp_data/aNSCLC/{}/{}.csv'.format(drug_key, prefix))
# pd_pat['index_date'] = pd.to_datetime(pd_pat['index_date'])
# pd_pat['index_date'] = pd_pat['index_date'].dt.date

# print("Basic")

# demog_list = []
# for idx in range(len(pd_pat)):
#     ptid = pd_pat.iloc[idx]['ptid']
#     index_date = pd_pat.iloc[idx]['index_date']
#     pat_pt_clinical = pat_pt_clinical_dict.get(ptid, {})
#     pat_pt_clinical_list = [ptid, index_date]
#     # Make sure 'birth_yr' exist and always be the last one
#     for pt_clinical_name in ['gender', 'race', 'ecog', 'birth_yr']:
#         pat_pt_clinical_list.append(pat_pt_clinical[pt_clinical_name])

#     if pat_pt_clinical_list[-1] == "1935 and Earlier":
#         pat_pt_clinical_list.append(index_date.year - 1935)
#     else:
#         pat_pt_clinical_list.append(index_date.year - int(pat_pt_clinical_list[-1]))
#     demog_list.append(pat_pt_clinical_list)

# demog_list = np.array(demog_list)
# pd_data = pd.DataFrame(demog_list, columns=['ptid', 'index_date', 'gender', 'race', 'ecog', 'birth_year', 'age_at_index'])
# pd_data = pd_data[['ptid', 'index_date', 'gender', 'race', 'ecog', 'age_at_index']]

# COMMAND ----------

print('DONE.')

# COMMAND ----------

