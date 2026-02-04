# Databricks notebook source
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os.path as osp
import argparse
import json
import os
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

cancer_type = "ansclc"
therapy_version = "combined_therapy_pres_adm_pro_14_90"

# COMMAND ----------

# compute the HR and p-value
# the follwoing code can be found from Chengxi's paper:
# Zang, C., Zhang, H., Xu, J. et al. High-throughput target trial emulation for Alzheimer’s disease drug repurposing with real-world data. Nat Commun 14, 8180 (2023). https://doi.org/10.1038/s41467-023-43929-1

# For each drug, we reported their sample means of different outcome estimators with 1000-time bootstrapped 95% confidence intervals over all the balanced trials. The bootstrapping hypothesis testing is used to test if the sample means of the adjusted aHRs is <1 and we reported the aHR’s bootstrapped P value. The significance level of aHR was corrected by the Bonferroni method for multiple testing.

# see Chengxi's code:
# https://github.com/calvin-zcx/pasc_phenotype/blob/c6be3b3a32b6371118c165873306a46ba107f054/misc/utils.py

# https://github.com/calvin-zcx/pasc_phenotype/blob/c6be3b3a32b6371118c165873306a46ba107f054/iptw/test_multiple_comparision.py 


import os
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
import numpy as np
import statsmodels.stats.multitest as smsmlt
from statsmodels.stats.multitest import fdrcorrection
# import multipy.fdr as fdr
np.random.seed(0)

#from lifelines import AalenJohansenFitter
#import matplotlib.pyplot as plt
#from lifelines.plotting import add_at_risk_counts

# #################################################   for all drugs  # compute HR,--CI, --p-value. ############

def boot_matrix(z, B):
    """Bootstrap sample
    Returns all bootstrap samples in a matrix"""
    z = np.array(z).flatten()
    n = len(z)  # sample size
    idz = np.random.randint(0, n, size=(B, n))  # indices to pick for all boostrap samples
    return z[idz]

def bootstrap_mean_ci(x, B=1000, alpha=0.05):
    n = len(x)
    # Generate boostrap distribution of sample mean
    xboot = boot_matrix(x, B=B)
    sampling_distribution = xboot.mean(axis=1)
    quantile_confidence_interval = np.percentile(sampling_distribution, q=(100 * alpha / 2, 100 * (1 - alpha / 2)))
    std = sampling_distribution.std()
    # if plot:
    #     plt.hist(sampling_distribution, bins="fd")
    return quantile_confidence_interval, std

def bootstrap_mean_pvalue(x, expected_mean=0., B=1000):
    """
    Ref:
    1. https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#cite_note-:0-1
    2. https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    3. https://github.com/mayer79/Bootstrap-p-values/blob/master/Bootstrap%20p%20values.ipynb
    4. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html?highlight=one%20sample%20ttest
    Bootstrap p values for one-sample t test
    Returns boostrap p value, test statistics and parametric p value"""
    n = len(x)
    orig = stats.ttest_1samp(x, expected_mean)
    # Generate boostrap distribution of sample mean
    x_boots = boot_matrix(x - x.mean() + expected_mean, B=B)
    x_boots_mean = x_boots.mean(axis=1)
    t_boots = (x_boots_mean - expected_mean) / (x_boots.std(axis=1, ddof=1) / np.sqrt(n))
    p = np.mean(t_boots >= orig[0])
    p_final = 2 * min(p, 1 - p)
    # Plot bootstrap distribution
    # if plot:
    #     plt.figure()
    #     plt.hist(x_boots_mean, bins="fd")
    return p_final, orig


def multiple_test_correct(pv_list, fdr_threshold=0.05):
    vbool_bonf, vp_bonf, _, threshold_bon = smsmlt.multipletests(pv_list, alpha=fdr_threshold, method='bonferroni')
    vbool_bh, vp_bh = smsmlt.fdrcorrection(pv_list, alpha=fdr_threshold)
    vbool_by, vp_by = smsmlt.fdrcorrection(pv_list, method="negcorr", alpha=fdr_threshold)
    # vbool_storey, vq_storey = fdr.qvalue(pv_list, threshold=fdr_threshold)

    df = pd.DataFrame({'p-value': pv_list,
                        'bool_bonf': vbool_bonf.astype(int), 'p-bonf': vp_bonf,
                        'bool_by': vbool_by.astype(int), 'p-by': vp_by,
                        'bool_bh': vbool_bh.astype(int), 'p-bh': vp_bh,
                        # 'bool_storey': vbool_storey.astype(int), 'q-storey': vq_storey,
                        })

    return df


# COMMAND ----------

# # testing a specifal case 
# data_t = pd.read_csv('/Workspace/Users/zhenxing.xu@regeneron.com/ansclc_0915/exp_data/95%_CI_Case.csv')
# data_t_hr = list(data_t['HR'])

# # produce a list with sampe distribution with data_t_hr

# # produce a list with sample distribution from data_t_hr (e.g., 100 bootstrap samples)
# np.random.seed(0)  # for reproducibility
# sampled_hr_list = np.random.choice(data_t_hr, size=100, replace=True).tolist()

# # compute the HR and p-value
# CI_t,_ = bootstrap_mean_ci(data_t_hr)
# CI_NM = np.percentile(data_t_hr, [2.5, 97.5])
# print('CI_t',CI_t)
# print('CI_NM',CI_NM)

# p_value_test, _ = bootstrap_mean_pvalue(np.array(data_t_hr), 1)

# print(p_value_test)

# COMMAND ----------

import numpy as np
import pandas as pd
from scipy.stats import norm

def pool_hr(hr_list, lower_ci_list, upper_ci_list, alpha=0.05):
    """
    Pool hazard ratios (HRs) using inverse-variance meta-analysis.
    
    Parameters:
    - hr_list: list of hazard ratios
    - lower_ci_list: list of lower 95% CI bounds
    - upper_ci_list: list of upper 95% CI bounds
    - alpha: significance level (default 0.05 for 95% CI)

    Returns:
    - pooled HR
    - pooled CI (lower, upper)
    - p-value
    """

    # Convert HR and CI to log scale
    log_hr = np.log(hr_list)
    log_lower = np.log(lower_ci_list)
    log_upper = np.log(upper_ci_list)

    # Standard errors from CIs
    se = (log_upper - log_lower) / (2 * norm.ppf(1 - alpha/2))

    # Weights = inverse variance
    weights = 1 / se**2

    # Pooled log(HR)
    pooled_log_hr = np.sum(weights * log_hr) / np.sum(weights)

    # SE of pooled log(HR)
    pooled_se = np.sqrt(1 / np.sum(weights))

    # 95% CI
    z = norm.ppf(1 - alpha/2)
    ci_lower = pooled_log_hr - z * pooled_se
    ci_upper = pooled_log_hr + z * pooled_se

    # Back-transform
    pooled_hr = np.exp(pooled_log_hr)
    pooled_ci = (np.exp(ci_lower), np.exp(ci_upper))

    # Z-test for p-value
    z_score = pooled_log_hr / pooled_se
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    return pooled_hr, pooled_ci, p_value


# # ===== Example usage =====
# hr_list = [0.85, 0.90, 1.10, 0.95, 0.88]   # example HRs from 5 experiments
# lower_ci_list = [0.70, 0.75, 0.85, 0.80, 0.72]
# upper_ci_list = [1.05, 1.08, 1.42, 1.15, 1.10]

# pooled_hr, pooled_ci, p_value = pool_hr(hr_list, lower_ci_list, upper_ci_list)

# print(f"Pooled HR: {pooled_hr:.3f}")
# print(f"95% CI: {pooled_ci[0]:.3f} – {pooled_ci[1]:.3f}")
# print(f"P-value: {p_value:.4f}")


# COMMAND ----------

# An example for HR, CI, and p-value ----NC and NM methods

# import numpy as np
# HR_sample = np.random.uniform(0.6, 1.19, 100) # # please genereate a list with 100 numbers between [0.1--2]
# print(HR_sample)

# HR_value = HR_sample

# HR_value_mean = float('%.2f' % np.mean(np.array(HR_value)))
# HR_value_CI_NC = bootstrap_mean_ci(np.array(HR_value))[0]
# HR_value_CI_low_NC = float('%.2f' % HR_value_CI_NC[0])
# HR_value_CI_high_NC = float('%.2f' % HR_value_CI_NC[1])
# HR_value_CI_NC_paper = "["+ str(HR_value_CI_low_NC) + ", "+ str(HR_value_CI_high_NC) +"]"
# print('HR_value_CI_NC',HR_value_CI_NC_paper)

# #Nature Medicine paper: 
# HR_CI_NM = np.percentile(HR_value, [2.5, 97.5]) 
# HR_CI_NM_low = float('%.2f' % HR_CI_NM[0])
# HR_CI_NM_high = float('%.2f' % HR_CI_NM[1])
# HR_CI_NM_paper = "["+ str(HR_CI_NM_low) + ", "+ str(HR_CI_NM_high) +"]"
# print('HR_CI_NM_paper',HR_CI_NM_paper)

# HR_value_p_value, _ = bootstrap_mean_pvalue(np.array(HR_value), expected_mean=1)

# print('HR_value_p_value', HR_value_p_value)


# COMMAND ----------

# MAGIC %md
# MAGIC # loading drug list

# COMMAND ----------

df_sel_drug_list = pd.read_csv("./exp_data/{}/{}/sel_drug_list.csv".format(cancer_type, therapy_version))

vomiting = ['ONDANSETRON HCL', 'DEXAMETHASONE', 'PALONOSETRON HCL', 'FOSAPREPITANT DIMEGLUMINE', 'LORAZEPAM']
pain = ['OXYCODONE HCL', 'ACETAMINOPHEN'] ## exclude 'MORPHINE SULFATE'
stomach_acid = ['FAMOTIDINE', 'PANTOPRAZOLE SODIUM']
# infection = ['PEGFILGRASTIM'] # VANCOMYCIN HCL
blood_clot = ['HEPARIN SODIUM;PORCINE/PF', 'ENOXAPARIN SODIUM']
other = ['DENOSUMAB', 'PEGFILGRASTIM', 'CYANOCOBALAMIN (VITAMIN B-12)', 'POTASSIUM CHLORIDE', 'MAGNESIUM SULFATE IN STERILE WATER', 'DIPHENHYDRAMINE HCL', 'PREDNISONE', 'FOLIC ACID']

df_sel_drug_list = df_sel_drug_list.loc[df_sel_drug_list['generic_desc'].isin(vomiting + pain + stomach_acid + blood_clot + other)]

# Rank drugs according to the predefined order in vomiting + other
drug_order = vomiting + pain + stomach_acid + blood_clot + other
drug_order = sorted(drug_order)

# Create an ordered categorical column based on the drug order
df_sel_drug_list = df_sel_drug_list.assign(
    rank=pd.Categorical(df_sel_drug_list['generic_desc'],
                        categories=drug_order,
                        ordered=True).codes + 1
)

# Sort the dataframe by the rank (optional, if you want the dataframe ordered)
df_sel_drug_list = df_sel_drug_list.sort_values('rank').reset_index(drop=True)

df_sel_drug_list

# COMMAND ----------

# MAGIC %md
# MAGIC # outcome---overall survival 

# COMMAND ----------

outcome = 'dod'# dod

# COMMAND ----------

threshold_balanced_covariate = 20 # 10 , 15, 20, we will use 10 as final setting
threshold_balanced_trial = 10 # 10%, we have 100 trials

obj_columns = ['drug_name', 'num_treated' , 'HR', '95% CI (NC_paper)', '95% CI (NM_paper)', 'p-value', 'pooled_hr', '95% CI (pooled)', 'pooled_p_value']
# obj_columns = ['drug_name', 'num_treated' , 'HR', pooled_ci[1]]
final_result_df = pd.DataFrame(columns = obj_columns)

# loading raw Hazard ratio
for idx in range(len(df_sel_drug_list)):# len(df_sel_drug_list)
    drug_name = df_sel_drug_list.iloc[idx]['generic_desc']
    if drug_name in ["PEMETREXED DISODIUM", "0.9 % SODIUM CHLORIDE", "PEGFILGRASTIM-APGF", "DEXTROSE 5 % IN WATER"]:
        continue
    drug_key = drug_name.replace('/', '-')

    if outcome=='dod':
        data = pd.read_csv('./exp_data/{}/{}/{}/results_v0.csv'.format(cancer_type, therapy_version, drug_key))
        case_control_data = pd.read_csv('./exp_data/{}/{}/{}/all_data_v0_control_{}.csv'.format(cancer_type, therapy_version, drug_key,"0"))
        num_treated = len(case_control_data.loc[case_control_data['treatment'] == 1])
    else:
        data = pd.read_csv('./exp_data/{}/{}/{}/results_v0_sae.csv'.format(cancer_type, therapy_version, drug_key))
        case_control_data = pd.read_csv('./exp_data/{}/{}/{}/all_data_v0_control_{}_sae.csv'.format(cancer_type, therapy_version, drug_key,"0"))
        num_treated = len(case_control_data.loc[case_control_data['treatment'] == 1])

    n_balanced_trial = len(data.loc[data['unbalanced_covariates'] < threshold_balanced_covariate])

    if n_balanced_trial>=threshold_balanced_trial:
        HR_value = data.loc[data['unbalanced_covariates'] < threshold_balanced_covariate]['HR'].values
        HR_value_CI_data = data.loc[data['unbalanced_covariates'] < threshold_balanced_covariate]['CI'].values
        HR_value_CI_data_l_list =  []
        HR_value_CI_data_h_list =  []
        for ci_s in HR_value_CI_data:
            # Remove surrounding brackets, split on whitespace, convert to float
            ci_nums = ci_s.strip('[]').split()
            HR_value_CI_data_l_list.append(float(ci_nums[0]))
            HR_value_CI_data_h_list.append(float(ci_nums[1]))

        # HR_value_median = float('%.2f' % np.median(np.array(HR_value)))
        # HR_value_bootstrap_ci = bootstrap((HR_value,),np.median,confidence_level=0.95,random_state=42,method='percentile')

        HR_value_mean = float('%.3f' % np.mean(np.array(HR_value)))
        # HR_value_bootstrap_ci = bootstrap((HR_value,), np.mean, confidence_level=0.95, random_state=42, method='percentile')
        # HR_value_CI_low = float('%.2f' % HR_value_bootstrap_ci.confidence_interval[0])
        # HR_value_CI_high = float('%.2f' % HR_value_bootstrap_ci.confidence_interval[1])

        HR_value_CI = bootstrap_mean_ci(np.array(HR_value))[0]
        HR_value_CI_low = float('%.3f' % HR_value_CI[0])
        HR_value_CI_high = float('%.3f' % HR_value_CI[1])

        #Nature Medicine paper: 
        HR_CI_NM = np.percentile(HR_value, [2.5, 97.5]) 
        HR_CI_NM_low = float('%.3f' % HR_CI_NM[0])
        HR_CI_NM_high = float('%.3f' % HR_CI_NM[1])
        HR_CI_NM_paper = "["+ str(HR_CI_NM_low) + ", "+ str(HR_CI_NM_high) +"]"
        
        # compute p-value
        HR_value_p_value, _ = bootstrap_mean_pvalue(np.array(HR_value), expected_mean=1)

        CI_HR = "["+ str(HR_value_CI_low) + ", "+ str(HR_value_CI_high) +"]"

        # compute HR CI, p-value based on pool_hr
        pooled_hr, pooled_ci_s, pooled_p_value = pool_hr(HR_value, HR_value_CI_data_l_list, HR_value_CI_data_h_list)
        pooled_hr = float('%.3f' % pooled_hr)
        pooled_p_value = float('%.3f' % pooled_p_value)
        pooled_ci = "["+ str(float('%.3f' % pooled_ci_s[0])) + ", "+ str(float('%.3f' % pooled_ci_s[1])) +"]"

        # final_result_df.loc[len(final_result_df)] = [drug_name, num_treated, HR_value_median, CI_HR, HR_value_p_value]
        final_result_df.loc[len(final_result_df)] = [drug_name, num_treated, HR_value_mean, CI_HR,  HR_CI_NM_paper, HR_value_p_value, pooled_hr, pooled_ci, pooled_p_value]

    else:
        print(f'this drug is not balanced: {drug_name}')

# pvalue correction
pv_list = final_result_df['p-value'].values
pv_list_df = multiple_test_correct(pv_list)

# merge the multiple‑testing correction results with the HR summary
# drop the original p-value column
final_result_df = final_result_df.drop(columns=['p-value'])
final_result_df = final_result_df.merge(pv_list_df, left_index=True, right_index=True)

# Generate a set of raw p-values (example data)
# In a real scenario, these would be the p-values from your hypothesis tests.
# p_values = np.array([0.001, 0.005, 0.012, 0.030, 0.045, 0.075, 0.200, 0.500])
# Perform FDR correction using the Benjamini-Hochberg procedure
# rejected, p_adjusted = fdrcorrection(p_values, alpha=0.05)
#print(final_result_df)

# if the value in bonf is less than 0.001, assigns as "<0.001" 
final_result_df['p-bonf'] = final_result_df['p-bonf'].apply(lambda x: "<0.0001" if x<0.0001 else x)
# rename column p-bonf as p-value_corrected_with_bonferroni
final_result_df = final_result_df.rename(columns={'p-bonf': 'p-value_corrected_with_bonferroni'})
# final_result_df = final_result_df[['drug_name', 'num_treated', 'HR', '95% CI (NC_paper)', '95% CI (NM_paper)', 'p-value_corrected_with_bonferroni']]
print(final_result_df)
saved_result_df = final_result_df[['drug_name', 'num_treated', 'HR', '95% CI (NC_paper)', 'p-value_corrected_with_bonferroni']]
#saved_result_df.to_csv('./exp_data/{}/{}/{}/{}_HR_p_value_{}.csv'.format(cancer_type, therapy_version, 'tem_result', cancer_type, outcome),index=False)

print('*******************************************************************************************')
print('Done!')

# COMMAND ----------

final_result_df

# COMMAND ----------

# MAGIC %md
# MAGIC # outcome--sub SAE

# COMMAND ----------

# MAGIC %md
# MAGIC ## compute ths Odds ratio and difference of incidence rate regarding SAE

# COMMAND ----------

# MAGIC %md
# MAGIC ## step 1: loading sub-sae list

# COMMAND ----------

# sub-sae
# loading all sae subgroup and select sae with enough samples
selection_sae_threshold = 1000 #1000

df_ae = pd.read_excel("../TTE_data/CTCAE_v5.0.xlsx", sheet_name='CTCAE v5.0 Clean Copy')
all_sub_sae_list = df_ae['MedDRA SOC'].unique()

sel_sub_sae_list = []
for i in all_sub_sae_list:
    print(i)
    num_pt = 0
    sub_sae = i
    with open('./exp_data/{}/{}/pat_diag_sae_dict_{}.pkl'.format(cancer_type, therapy_version, sub_sae), 'rb') as f:
        sub_sae_dic = pkl.load(f) 
    print(len(sub_sae_dic))
    for key, value in sub_sae_dic.items():
        if value:
            #print(f"Key: {key}, Value: {value}")
            num_pt = num_pt + 1
    if num_pt > selection_sae_threshold:
        sel_sub_sae_list.append(sub_sae)
    else:
        print("no pt for this sub_sae")

sel_sub_sae_list

# COMMAND ----------

# MAGIC %md
# MAGIC ## select sae for figure (order them)

# COMMAND ----------

# Second selecting SAE
# sel_sub_sae_list = sel_sub_sae_list

# good results for this sel_sub_sae_list
# sel_sub_sae_list = ['Gastrointestinal disorders', 'Metabolism and nutrition disorders', 'Respiratory, thoracic and mediastinal disorders',
#                       'Blood and lymphatic system disorders', 
#                      'Nervous system disorders', 'Vascular disorders', 'Cardiac disorders',
#                      'General disorders and administration site conditions']


sel_sub_sae_list = sorted(sel_sub_sae_list) # by alphabetical order


# COMMAND ----------

# MAGIC %md
# MAGIC ## step 2: compute the odds ratio, incidence rate difference

# COMMAND ----------

threshold_balanced_covariate = 20
threshold_balanced_trial = 10
unbalanced_covariates = 'unbalanced_covariates_02' # smd>0.2; unbalanced_covariates_03  smd>0.3

#sel_sub_sae_list = ['Skin and subcutaneous tissue disorders'] #'Vascular disorders'
#df_sel_drug_list = df_sel_drug_list[:1] #['DEXAMETHASONE']

# Exclude FAMOTIDINE from the selected drug list
df_sel_drug_list = df_sel_drug_list[~df_sel_drug_list['generic_desc'].isin(['FAMOTIDINE'])].reset_index(drop=True) # it is not balanced

results_df_OR_allday = pd.DataFrame(columns= sel_sub_sae_list, index=list(df_sel_drug_list['generic_desc']))
results_df_OR_180day = pd.DataFrame(columns= sel_sub_sae_list, index=list(df_sel_drug_list['generic_desc']))
results_df_IRD_allday = pd.DataFrame(columns= sel_sub_sae_list, index=list(df_sel_drug_list['generic_desc']))
results_df_IRD_180day = pd.DataFrame(columns= sel_sub_sae_list, index=list(df_sel_drug_list['generic_desc']))

for sub_sae in sel_sub_sae_list:
    print('sub_sae:',sub_sae)
    for idx in range(len(df_sel_drug_list)):
        drug_name = df_sel_drug_list.iloc[idx]['generic_desc']
        drug_key = drug_name.replace('/', '-')
        # data = pd.read_csv('./exp_data/{}/{}/{}/{}/results_v0_sae_odds.csv'.format(cancer_type, therapy_version, drug_key, sub_sae))
        data = pd.read_csv('./exp_data/{}/{}/{}/{}/results_v0_sae_hr_odds_ird.csv'.format(cancer_type, therapy_version, drug_key, sub_sae))

        n_balanced_trial = len(data.loc[data[unbalanced_covariates] < threshold_balanced_covariate])
        if n_balanced_trial>=threshold_balanced_trial:
            # OR---all day
            OR_allday_value_list = data.loc[data[unbalanced_covariates] < threshold_balanced_covariate]['OR_all_day'].values
            OR_allday_value_list = OR_allday_value_list[np.isfinite(OR_allday_value_list)] ## remove inf from OR_allday_value_list
            OR_allday_value_mean = float('%.3f' % np.mean(np.array(OR_allday_value_list)))
            OR_allday_p_value, _ = bootstrap_mean_pvalue(np.array(OR_allday_value_list), expected_mean=1)

            # OR---180 day
            OR_180day_value_list = data.loc[data[unbalanced_covariates] < threshold_balanced_covariate]['OR_180day'].values
            OR_180day_value_list = OR_180day_value_list[np.isfinite(OR_180day_value_list)]
            OR_180day_value_mean = float('%.3f' % np.mean(np.array(OR_180day_value_list)))
            OR_180day_p_value, _ = bootstrap_mean_pvalue(np.array(OR_180day_value_list), expected_mean=1)

            # difference of incidence rate---all day
            IRD_allday_value_list = data.loc[data[unbalanced_covariates] < threshold_balanced_covariate]['diff_ir_all_day'].values
            IRD_allday_value_list = IRD_allday_value_list[np.isfinite(IRD_allday_value_list)]
            IRD_allday_value_mean = float('%.3f' % np.mean(np.array(IRD_allday_value_list)))
            IRD_allday_p_value, _ = bootstrap_mean_pvalue(np.array(IRD_allday_value_list), expected_mean=0)

            # difference of incidence rate---180 day
            IRD_180day_value_list = data.loc[data[unbalanced_covariates] < threshold_balanced_covariate]['diff_ir_180day'].values
            IRD_180day_value_list = IRD_180day_value_list[np.isfinite(IRD_180day_value_list)]
            IRD_180day_value_mean = float('%.3f' % np.mean(np.array(IRD_180day_value_list)))
            IRD_180day_p_value, _ = bootstrap_mean_pvalue(np.array(IRD_180day_value_list), expected_mean=0)


            if OR_allday_p_value<0.05:
                results_df_OR_allday.loc[drug_name, sub_sae] = OR_allday_value_mean
            if OR_180day_p_value<0.05:
                results_df_OR_180day.loc[drug_name, sub_sae] = OR_180day_value_mean
            if IRD_allday_p_value<0.05:
                results_df_IRD_allday.loc[drug_name, sub_sae] = IRD_allday_value_mean
            if IRD_180day_p_value<0.05:
                results_df_IRD_180day.loc[drug_name, sub_sae] = IRD_180day_value_mean

#results_df_OR.to_csv('./exp_data/{}/{}/{}/{}_OR_{}.csv'.format(cancer_type, therapy_version, 'tem_result', cancer_type, 'sub_sae'))
#results_df_IRD.to_csv('./exp_data/{}/{}/{}/{}_IRD_{}.csv'.format(cancer_type, therapy_version, 'tem_result', cancer_type, 'sub_sae'))


# COMMAND ----------

# an example for plotting heatmap

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# Sample data with NaN
data = np.array([[1, 2, np.nan], [4, 0.8, 6], [7, np.nan, 9]])
df = pd.DataFrame(data, columns=['ColA', 'ColB', 'ColC'],
                  index=['index_1', 'index_2', 'index_3'])


# Define colormap: green (<1), white (=1), red (>1)
cmap = sns.diverging_palette(150, 10, as_cmap=True)
cmap.set_bad(color="lightgray")  # NaN values → gray


# Normalize around 1
norm = TwoSlopeNorm(vmin=np.nanmin(df.values), vcenter=1, vmax=np.nanmax(df.values))


# Plot heatmap
plt.figure(figsize=(6, 4))
ax = sns.heatmap(df, annot=True, cmap=cmap, norm=norm, cbar=True)


# Put column labels on top and rotate 45°
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xticks(rotation=45)
# Adjust cell dimensions – make each heatmap cell square (size controlled by figure size)
ax.set_aspect("equal")          # ensures equal width/height for cells

# Move the colorbar (legend) to the left side of the heatmap
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
# Optional: adjust its position relative to the plot
# cbar.ax.set_position([ -0.15, 0.1, 0.03, 0.8 ])
# Adjust the colorbar position to be closer to the heatmap
# (reduce the horizontal offset from the plot)
cbar.ax.set_position([0.17, 0.18, 0.03, 0.7])  # [left, bottom, width, height]

# put the index at the right side of the heatmap
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
plt.yticks(rotation=0)

plt.title("Conditional Heatmap (Green < 1, Red > 1, NaN = Gray)", pad=40)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # plotting odds ratio--all day

# COMMAND ----------

figure_size_length = 8
figure_size_width = 6

# COMMAND ----------

# # odds ratio for allday

df = results_df_OR_allday.copy()
df = df.fillna(np.nan)

#Define colormap: green (<1), white (=1), red (>1)
cmap = sns.diverging_palette(150, 10, as_cmap=True)
cmap.set_bad(color="gray")  # NaN values → gray

# Normalize around 1
norm = TwoSlopeNorm(vmin=np.nanmin(df.values), vcenter=1, vmax=np.nanmax(df.values))

# Plot heatmap
plt.figure(figsize=(figure_size_length, figure_size_width))
ax = sns.heatmap(df, annot=True, cmap=cmap, norm=norm, cbar=True, fmt='.3g')

# Put column labels on top and rotate 45°
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xticks(rotation=85)

plt.title("Cohort ANSCLC: Odds ratio on outcome Sub-SAE (all days), (Green < 1, White =1, Red > 1, Gray (p-value>0.05))", pad=40)
#plt.savefig("./exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_OR_sub_sae_allday.pdf", dpi=500, format="pdf",bbox_inches='tight')

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # plotting odds ratio--180 day

# COMMAND ----------

results_df_OR_180day_tem = results_df_OR_180day.copy()

# COMMAND ----------

# odds ratio for 180day
df = results_df_OR_180day_tem.copy()
df = df.fillna(np.nan)

#Define colormap: green (<1), white (=1), red (>1)
cmap = sns.diverging_palette(150, 10, as_cmap=True)
cmap.set_bad(color="gray")  # NaN values → gray

# Normalize around 1
norm = TwoSlopeNorm(vmin=np.nanmin(df.values), vcenter=1, vmax=np.nanmax(df.values))

# Plot heatmap
plt.figure(figsize=(figure_size_length, figure_size_width))
ax = sns.heatmap(df, annot=True, cmap=cmap, norm=norm, cbar=True, fmt='.3g')# decimal

# Put column labels on top and rotate 45°
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xticks(rotation=85)

plt.title("Cohort ANSCLC: Odds ratio on outcome Sub-SAE (180 day), (Green < 1, White =1, Red > 1, Gray (p-value>0.05))", pad=40)
#plt.savefig("./exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_OR_sub_sae_180day.pdf", dpi=500, format="pdf",bbox_inches='tight')

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # plotting odds ratio ---180 day, without num in the cell

# COMMAND ----------

### rename the SAE outcome name and rename the row name 

# Rename stratify columns
col_rename = {
'Blood and lymphatic system disorders': 'SAE_A',
'Cardiac disorders':'SAE_B',
'Gastrointestinal disorders':'SAE_C',
'General disorders and administration site conditions':'SAE_D',
'Metabolism and nutrition disorders':'SAE_E',
'Nervous system disorders':'SAE_F',
'Respiratory, thoracic and mediastinal disorders':'SAE_G',
'Vascular disorders':'SAE_H'
}
results_df_OR_180day_tem = results_df_OR_180day_tem.rename(columns=col_rename)

# Drug name mapping (drug_name → Category, Drug)
drug_info = pd.DataFrame(
    [
        ["PANTOPRAZOLE SODIUM", "Acid suppression", "Pantoprazole"],
        ["OXYCODONE HCL", "Analgesic", "Oxycodone"],
        ["ACETAMINOPHEN", "Analgesic", "Acetaminophen"],
        ["ENOXAPARIN SODIUM", "Anticoagulant", "Enoxaparin"],
        ["HEPARIN SODIUM;PORCINE/PF", "Anticoagulant", "Heparin"],
        ["FOSAPREPITANT DIMEGLUMINE", "Antiemetic", "Fosaprepitant"],
        ["ONDANSETRON HCL", "Antiemetic", "Ondansetron"],
        ["PALONOSETRON HCL", "Antiemetic", "Palonosetron"],
        ["DIPHENHYDRAMINE HCL", "Antihistamine", "Diphenhydramine"],
        ["LORAZEPAM", "Anxiolytic", "Lorazepam"],
        ["DENOSUMAB", "Bone support", "Denosumab"],
        ["DEXAMETHASONE", "Corticosteroid", "Dexamethasone"],
        ["PREDNISONE", "Corticosteroid", "Prednisone"],
        ["MAGNESIUM SULFATE IN STERILE WATER", "Electrolyte replacement", "Magnesium sulfate"],
        ["POTASSIUM CHLORIDE", "Electrolyte replacement", "Potassium chloride"],
        ["PEGFILGRASTIM", "Hematologic support", "Pegfilgrastim"],
        ["CYANOCOBALAMIN (VITAMIN B-12)", "Vitamin", "Cyanocobalamin"],
        ["FOLIC ACID", "Vitamin", "Folic acid"],
    ],
    columns=["drug_name", "Category", "Drug"],
)

# reverse the order of rows in drug_info
drug_info = drug_info.iloc[::-1]

# Map drug_name → Drug (new index) and add Category column
drug_to_name = drug_info.set_index("drug_name")["Drug"].to_dict()
cat_map = drug_info.set_index("drug_name")["Category"].to_dict()

results_df_OR_180day_tem = results_df_OR_180day_tem.rename(index=drug_to_name)
results_df_OR_180day_tem["Category"] = results_df_OR_180day_tem.index.map(lambda x: cat_map.get(x, None))

# Reorder columns: Category first, then the stratify columns
ordered_cols = ["Category"] + [c for c in results_df_OR_180day_tem.columns if c != "Category"]
results_df_OR_180day_tem = results_df_OR_180day_tem[ordered_cols]
results_df_OR_180day_tem = results_df_OR_180day_tem.drop(columns=["Category"])

# Order rows to follow the drug_info order
desired_order = [
    drug_to_name[drug] for drug in drug_info["drug_name"]
    if drug_to_name.get(drug) in results_df_OR_180day_tem.index
]
results_df_OR_180day_tem = results_df_OR_180day_tem.reindex(desired_order)


results_df_OR_180day_tem

# COMMAND ----------

# odds ratio for 180day
df = results_df_OR_180day_tem.copy()
df = df.fillna(np.nan)

figure_size_length_tem = 20
figure_size_width_tem = 8

#Define colormap: green (<1), white (=1), red (>1)
cmap = sns.diverging_palette(150, 10, as_cmap=True)
cmap.set_bad(color="gray")  # NaN values → gray

# Normalize around 1
norm = TwoSlopeNorm(vmin=np.nanmin(df.values), vcenter=1, vmax=np.nanmax(df.values))

# Plot heatmap
plt.figure(figsize=(figure_size_length, figure_size_width))
ax = sns.heatmap(df, annot=False, cmap=cmap, norm=norm, cbar=True, fmt='.3g')# decimal

# Put column labels on top and rotate 45°
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xticks(rotation=90)

ax.set_aspect("equal")          # ensures equal width/height for cells

# Move the colorbar (legend) to the left side of the heatmap
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_ticks_position('right')
cbar.ax.yaxis.set_label_position('right')
# Optional: adjust its position relative to the plot
# cbar.ax.set_position([ -0.15, 0.1, 0.03, 0.8 ])
# Adjust the colorbar position to be closer to the heatmap
# (reduce the horizontal offset from the plot)
cbar.ax.set_position([0.78, 0.12, 0.03, 0.5])  # [left, bottom, width, height]
# put the drug name at the right side of the heatmap
ax.yaxis.tick_left()
ax.yaxis.set_label_position('left')
plt.yticks(rotation=0)

plt.title("Cohort ANSCLC: Odds ratio on outcome Sub-SAE (180 day), (Green < 1, White =1, Red > 1, Gray (p-value>0.05))", pad=40)
plt.savefig("./exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_OR_sub_sae_180day_without_num.pdf", dpi=500, format="pdf",bbox_inches='tight')

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # plottiing incidence rate difference (IRD) all day

# COMMAND ----------

results_df_IRD_allday_tem = results_df_IRD_allday.copy()

# COMMAND ----------

# for incidence rate difference ---all day

df = results_df_IRD_allday_tem.copy()
df = df.fillna(np.nan)

#Define colormap: green (<0), white (=0), red (>0)
cmap = sns.diverging_palette(150, 10, as_cmap=True)
cmap.set_bad(color="gray")  # NaN values → gray

# Normalize around 0
norm = TwoSlopeNorm(vmin=np.nanmin(df.values), vcenter=0, vmax=np.nanmax(df.values))

# Plot heatmap
plt.figure(figsize=(figure_size_length, figure_size_width))
ax = sns.heatmap(df, annot=True, cmap=cmap, norm=norm, cbar=True,fmt='.3g')

# Put column labels on top and rotate 45°
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xticks(rotation=85)

plt.title("Cohort ANSCLC: Incidence rate difference on outcome Sub-SAE (all days), (Green < 0, White =0, Red > 0, Gray (p-value>0.05))", pad=40)
#plt.savefig("./exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_IRD_sub_sae_allday.pdf", dpi=500, format="pdf",bbox_inches='tight')


plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # plotting incidence rate difference ---180 day

# COMMAND ----------

results_df_IRD_180day_tem = results_df_IRD_180day.copy()

# COMMAND ----------

# for incidence rate difference ---180 day

df = results_df_IRD_180day_tem.copy()
df = df.fillna(np.nan)

#Define colormap: green (<0), white (=0), red (>0)
cmap = sns.diverging_palette(150, 10, as_cmap=True)
cmap.set_bad(color="gray")  # NaN values → gray

# Normalize around 0
norm = TwoSlopeNorm(vmin=np.nanmin(df.values), vcenter=0, vmax=np.nanmax(df.values))

# Plot heatmap
plt.figure(figsize=(figure_size_length, figure_size_width))
ax = sns.heatmap(df, annot=True, cmap=cmap, norm=norm, cbar=True,fmt='.3g')

# Put column labels on top and rotate 45°
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xticks(rotation=85)

plt.title("Cohort ANSCLC: Incidence rate difference on outcome Sub-SAE (180 day), (Green < 0, White =0, Red > 0, Gray (p-value>0.05))", pad=40)
plt.savefig("./exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_IRD_sub_sae_180day.pdf", dpi=500, format="pdf",bbox_inches='tight')


plt.show()

# COMMAND ----------

results_df_IRD_180day_tem

# COMMAND ----------

# MAGIC %md
# MAGIC # plotting incidence rate difference ---180 day, without num in the cell

# COMMAND ----------

### rename the SAE outcome name and rename the row name 

# Rename stratify columns
col_rename = {
'Blood and lymphatic system disorders': 'SAE_A',
'Cardiac disorders':'SAE_B',
'Gastrointestinal disorders':'SAE_C',
'General disorders and administration site conditions':'SAE_D',
'Metabolism and nutrition disorders':'SAE_E',
'Nervous system disorders':'SAE_F',
'Respiratory, thoracic and mediastinal disorders':'SAE_G',
'Vascular disorders':'SAE_H'
}
results_df_IRD_180day_tem = results_df_IRD_180day_tem.rename(columns=col_rename)

# Drug name mapping (drug_name → Category, Drug)
drug_info = pd.DataFrame(
    [
        ["PANTOPRAZOLE SODIUM", "Acid suppression", "Pantoprazole"],
        ["OXYCODONE HCL", "Analgesic", "Oxycodone"],
        ["ACETAMINOPHEN", "Analgesic", "Acetaminophen"],
        ["ENOXAPARIN SODIUM", "Anticoagulant", "Enoxaparin"],
        ["HEPARIN SODIUM;PORCINE/PF", "Anticoagulant", "Heparin"],
        ["FOSAPREPITANT DIMEGLUMINE", "Antiemetic", "Fosaprepitant"],
        ["ONDANSETRON HCL", "Antiemetic", "Ondansetron"],
        ["PALONOSETRON HCL", "Antiemetic", "Palonosetron"],
        ["DIPHENHYDRAMINE HCL", "Antihistamine", "Diphenhydramine"],
        ["LORAZEPAM", "Anxiolytic", "Lorazepam"],
        ["DENOSUMAB", "Bone support", "Denosumab"],
        ["DEXAMETHASONE", "Corticosteroid", "Dexamethasone"],
        ["PREDNISONE", "Corticosteroid", "Prednisone"],
        ["MAGNESIUM SULFATE IN STERILE WATER", "Electrolyte replacement", "Magnesium sulfate"],
        ["POTASSIUM CHLORIDE", "Electrolyte replacement", "Potassium chloride"],
        ["PEGFILGRASTIM", "Hematologic support", "Pegfilgrastim"],
        ["CYANOCOBALAMIN (VITAMIN B-12)", "Vitamin", "Cyanocobalamin"],
        ["FOLIC ACID", "Vitamin", "Folic acid"],
    ],
    columns=["drug_name", "Category", "Drug"],
)

# reverse the order of rows in drug_info
drug_info = drug_info.iloc[::-1]

# Map drug_name → Drug (new index) and add Category column
drug_to_name = drug_info.set_index("drug_name")["Drug"].to_dict()
cat_map = drug_info.set_index("drug_name")["Category"].to_dict()

results_df_IRD_180day_tem = results_df_IRD_180day_tem.rename(index=drug_to_name)
results_df_IRD_180day_tem["Category"] = results_df_IRD_180day_tem.index.map(lambda x: cat_map.get(x, None))

# Reorder columns: Category first, then the stratify columns
ordered_cols = ["Category"] + [c for c in results_df_IRD_180day_tem.columns if c != "Category"]
results_df_IRD_180day_tem = results_df_IRD_180day_tem[ordered_cols]
results_df_IRD_180day_tem = results_df_IRD_180day_tem.drop(columns=["Category"])

# Order rows to follow the drug_info order
desired_order = [
    drug_to_name[drug] for drug in drug_info["drug_name"]
    if drug_to_name.get(drug) in results_df_IRD_180day_tem.index
]
results_df_IRD_180day_tem = results_df_IRD_180day_tem.reindex(desired_order)


results_df_IRD_180day_tem

# COMMAND ----------

# for incidence rate difference ---180 day

df = results_df_IRD_180day_tem.copy()
df = df.fillna(np.nan)

figure_size_length_tem = 20
figure_size_width_tem = 8

#Define colormap: green (<0), white (=0), red (>0)
cmap = sns.diverging_palette(150, 10, as_cmap=True)
cmap.set_bad(color="gray")  # NaN values → gray

# Normalize around 0
norm = TwoSlopeNorm(vmin=np.nanmin(df.values), vcenter=0, vmax=np.nanmax(df.values))

# Plot heatmap
plt.figure(figsize=(figure_size_length, figure_size_width))
ax = sns.heatmap(df, annot=False, cmap=cmap, norm=norm, cbar=True,fmt='.3g')

# Put column labels on top and rotate 45°
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xticks(rotation=90)
ax.set_aspect("equal")          # ensures equal width/height for cells

# Move the colorbar (legend) to the left side of the heatmap
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
# Optional: adjust its position relative to the plot
# cbar.ax.set_position([ -0.15, 0.1, 0.03, 0.8 ])
# Adjust the colorbar position to be closer to the heatmap
# (reduce the horizontal offset from the plot)
cbar.ax.set_position([0.43, 0.12, 0.03, 0.5])  # [left, bottom, width, height]

# put the drug name at the right side of the heatmap
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
plt.yticks(rotation=0)

plt.title("Cohort ANSCLC: Incidence rate difference on outcome Sub-SAE (180 day), (Green < 0, White =0, Red > 0, Gray (p-value>0.05))", pad=40)
plt.savefig("./exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_IRD_sub_sae_180day_without_num.pdf", dpi=500, format="pdf",bbox_inches='tight')


plt.show()

# COMMAND ----------

results_df_IRD_180day_tem

# COMMAND ----------

# MAGIC %md
# MAGIC # stratify analysis ---Overall survival--Hazard Ratio

# COMMAND ----------

threshold_balanced_covariate = 20
threshold_balanced_trial = 10
unbalanced_covariates = 'unbalanced_covariates_02' # smd>0.2; unbalanced_covariates_03  smd>0.3

sel_sub_stratify_list = ['gender_0', 'gender_1', 'age_less_65', 'age_large_65', 'meta_total_0', 'meta_total_1', 'meta_total_2', 'meta_lung_1', 'meta_brain_1', 'meta_bone_1',  'meta_liver_1', 'meta_adrenal_gland_1']
#df_sel_drug_list = df_sel_drug_list[:1] #['DEXAMETHASONE']

# Exclude FAMOTIDINE from the selected drug list,  it is not balanced
df_sel_drug_list = df_sel_drug_list[~df_sel_drug_list['generic_desc'].isin(['FAMOTIDINE'])].reset_index(drop=True) # it is not balanced

results_df_stratify = pd.DataFrame(columns= sel_sub_stratify_list, index=list(df_sel_drug_list['generic_desc']))

for sub_stratify in sel_sub_stratify_list:
    print('sub_stratify:',sub_stratify)
    for idx in range(len(df_sel_drug_list)):
        drug_name = df_sel_drug_list.iloc[idx]['generic_desc']
        drug_key = drug_name.replace('/', '-')
        # data = pd.read_csv('./exp_data/{}/{}/{}/{}/results_v0_sae_odds.csv'.format(cancer_type, therapy_version, drug_key, sub_sae))
        #data = pd.read_csv('./exp_data/{}/{}/{}/{}/results_v0_sae_hr_odds_ird.csv'.format(cancer_type, therapy_version, drug_key, sub_sae))
        data = pd.read_csv('./exp_data/{}/{}/{}/results_v0_dod_hr_stratify_{}.csv'.format(cancer_type, therapy_version, drug_key, sub_stratify))

        n_balanced_trial = len(data.loc[data[unbalanced_covariates] < threshold_balanced_covariate])
        if n_balanced_trial>=threshold_balanced_trial:
            HR_value_list = data.loc[data[unbalanced_covariates] < threshold_balanced_covariate]['HR'].values
            #OR_allday_value_list = OR_allday_value_list[np.isfinite(OR_allday_value_list)] ## remove inf from OR_allday_value_list
            HR_value_mean = float('%.3f' % np.mean(np.array(HR_value_list)))
            HR_p_value, _ = bootstrap_mean_pvalue(np.array(HR_value_list), expected_mean=1)

            if HR_p_value<0.05:
                results_df_stratify.loc[drug_name, sub_stratify] = HR_value_mean

results_df_stratify.to_csv('./exp_data/{}/{}/{}/{}_HR_dod_{}.csv'.format(cancer_type, therapy_version, 'tem_result', cancer_type, 'stratify'))


# COMMAND ----------

results_df_stratify

# COMMAND ----------

results_df_stratify_tem = results_df_stratify.copy()

# COMMAND ----------

# Rename stratify columns
col_rename = {
    "gender_0": "Female",
    "gender_1": "Male",
    "age_less_65": "age<65",
    "age_large_65": "age>=65",
    "meta_total_0": "Total Metastasis number=0",
    "meta_total_1": "Total Metastasis number=1",
    "meta_total_2": "Total Metastasis number>=2",
    "meta_lung_1": "Metastasis site at the lung",
    "meta_brain_1": "Metastasis site in the brain",
    "meta_bone_1": "Metastasis site at the bone",
    "meta_liver_1": "Metastasis site at the liver",
    "meta_adrenal_gland_1": "Metastasis site at the adrenal glands",
}
results_df_stratify_tem = results_df_stratify_tem.rename(columns=col_rename)

# Drug name mapping (drug_name → Category, Drug)
drug_info = pd.DataFrame(
    [
        ["PANTOPRAZOLE SODIUM", "Acid suppression", "Pantoprazole"],
        ["OXYCODONE HCL", "Analgesic", "Oxycodone"],
        ["ACETAMINOPHEN", "Analgesic", "Acetaminophen"],
        ["ENOXAPARIN SODIUM", "Anticoagulant", "Enoxaparin"],
        ["HEPARIN SODIUM;PORCINE/PF", "Anticoagulant", "Heparin"],
        ["FOSAPREPITANT DIMEGLUMINE", "Antiemetic", "Fosaprepitant"],
        ["ONDANSETRON HCL", "Antiemetic", "Ondansetron"],
        ["PALONOSETRON HCL", "Antiemetic", "Palonosetron"],
        ["DIPHENHYDRAMINE HCL", "Antihistamine", "Diphenhydramine"],
        ["LORAZEPAM", "Anxiolytic", "Lorazepam"],
        ["DENOSUMAB", "Bone support", "Denosumab"],
        ["DEXAMETHASONE", "Corticosteroid", "Dexamethasone"],
        ["PREDNISONE", "Corticosteroid", "Prednisone"],
        ["MAGNESIUM SULFATE IN STERILE WATER", "Electrolyte replacement", "Magnesium sulfate"],
        ["POTASSIUM CHLORIDE", "Electrolyte replacement", "Potassium chloride"],
        ["PEGFILGRASTIM", "Hematologic support", "Pegfilgrastim"],
        ["CYANOCOBALAMIN (VITAMIN B-12)", "Vitamin", "Cyanocobalamin"],
        ["FOLIC ACID", "Vitamin", "Folic acid"],
    ],
    columns=["drug_name", "Category", "Drug"],
)

# reverse the order of rows in drug_info
drug_info = drug_info.iloc[::-1]

# Map drug_name → Drug (new index) and add Category column
drug_to_name = drug_info.set_index("drug_name")["Drug"].to_dict()
cat_map = drug_info.set_index("drug_name")["Category"].to_dict()

results_df_stratify_tem = results_df_stratify_tem.rename(index=drug_to_name)
results_df_stratify_tem["Category"] = results_df_stratify_tem.index.map(lambda x: cat_map.get(x, None))

# Reorder columns: Category first, then the stratify columns
ordered_cols = ["Category"] + [c for c in results_df_stratify_tem.columns if c != "Category"]
results_df_stratify_tem = results_df_stratify_tem[ordered_cols]

results_df_stratify_tem = results_df_stratify_tem.drop(columns=["Category"])

# Order rows to follow the drug_info order
desired_order = [
    drug_to_name[drug] for drug in drug_info["drug_name"]
    if drug_to_name.get(drug) in results_df_stratify_tem.index
]
results_df_stratify_tem = results_df_stratify_tem.reindex(desired_order)

results_df_stratify_tem

# COMMAND ----------

# MAGIC %md
# MAGIC # stratify analysis ---Overall survival--Hazard Ratio
# MAGIC
# MAGIC ## heatmap

# COMMAND ----------

# for HR--stratify

df = results_df_stratify_tem.copy()
df = df.fillna(np.nan)

#Define colormap: green (<0), white (=0), red (>0)
cmap = sns.diverging_palette(150, 10, as_cmap=True)
cmap.set_bad(color="gray")  # NaN values → gray

# Normalize around 0
norm = TwoSlopeNorm(vmin=np.nanmin(df.values), vcenter=1, vmax=np.nanmax(df.values))

# Plot heatmap
plt.figure(figsize=(figure_size_length, figure_size_width))
ax = sns.heatmap(df, annot=True, cmap=cmap, norm=norm, cbar=True,fmt='.3g')

# Put column labels on top and rotate 45°
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xticks(rotation=85)

plt.title("Cohort ANSCLC: Hazard ratio (HR) on outcome overall survival stratified by covariates, (Green < 1, White =1, Red > 1, Gray (p-value>0.05))", pad=40)
plt.savefig("./exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_HR_dod_stratify.pdf", dpi=500, format="pdf",bbox_inches='tight')


plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # stratify analysis ---Overall survival--Hazard Ratio
# MAGIC
# MAGIC ## bubble plot

# COMMAND ----------

# MAGIC %md
# MAGIC ## an example for bubble plot

# COMMAND ----------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Example data similar to your figure
# data = {
#     'Gene': ['Cxcl12', 'Mdk', 'Kit', 'Notch2', 'Pdgfra', 'Runx2', 'Col1a1'],
#     'CellType': ['CP-like cell', 'CP-like cell', 'Pre-Od', 'Pre-Od', 'Mature Od', 'Mature Od', 'Mature Od'],
#     'AverageExpression': [1.0, 0.8, 0.3, 0.2, -0.1, 0.5, 0.9],
#     'PercentExpressed': [95, 80, 70, 60, 40, 50, 85]
# }

# df = pd.DataFrame(data)

# # Set up the figure
# plt.figure(figsize=(6, 8))

# # Create bubble plot
# scatter = sns.scatterplot(
#     data=df,
#     x="CellType",
#     y="Gene",
#     size="PercentExpressed",
#     hue="AverageExpression",
#     palette="coolwarm",
#     sizes=(20, 400),
#     edgecolor="black",
#     linewidth=0.5
# )

# # Customize appearance
# plt.title("Gene Expression Across Cell Types", fontsize=14)
# plt.xlabel("")
# plt.ylabel("")
# plt.xticks(rotation=30, ha='right')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# # Adjust layout
# plt.tight_layout()
# plt.show()


# COMMAND ----------

results_df_stratify_tem

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = results_df_stratify_tem.copy()
# reverse the order of rows in drug_info
df = df.iloc[::-1]

# add drug name for the first row
df['drug_name'] = df.index
df = df.reset_index(drop=True)

# Melt the DataFrame into long format
df_melted = df.melt(id_vars=["drug_name"], var_name="Covariate", value_name="HR")

# Define color mapping
def get_color(val):
    if pd.isna(val):
        return "gray"
    elif val > 1:
        red_intensity = min(1, (val - 1) / 1.5)
        return (1, 1 - red_intensity, 1 - red_intensity)
    else:
        green_intensity = min(1, (1 - val) / 1)
        return (1 - green_intensity, 1, 1 - green_intensity)
        # chang the color to deep green color , not green
        # return (0, 0.5, 0)

# Apply color and size mapping
df_melted["color"] = df_melted["HR"].apply(get_color)
df_melted["size"] = df_melted["HR"].apply(lambda x: 50 if pd.isna(x) else abs(x - 1) * 400 + 50)

# Plot bubble chart
fig, ax = plt.subplots(figsize=(10, 8))
for _, row in df_melted.iterrows():
    ax.scatter(row["Covariate"], row["drug_name"], s=row["size"], color=row["color"], edgecolor="black", alpha=0.8)

ax.set_xlabel("Stratified Covariates", fontsize=12)
ax.set_ylabel("Drug Name", fontsize=12)
ax.set_title("Stratified Hazard Ratios (HRs) for aNSCLC Cohort", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("./exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_HR_dod_stratify_bubble_plot.pdf", dpi=500, format="pdf",bbox_inches='tight')

plt.show()


# COMMAND ----------

df

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # stratify analysis ---Overall survival--Hazard Ratio
# MAGIC
# MAGIC ## Circular Summary Plot

# COMMAND ----------

# prepare the data to standard format

import pandas as pd

# Reset index to turn the drug names (index) into a column
df_long = (
    results_df_stratify_tem
    .reset_index()
    .melt(id_vars=results_df_stratify_tem.index.name or "index",
          var_name="Category",
          value_name="HR")
)

# Rename the index column to "Outcome"
df_long = df_long.rename(columns={results_df_stratify_tem.index.name or "index": "Outcome"})

# Create p‑value column:
#   - if the HR (or original p‑value) is not NaN → set pval = 0.001
#   - if it is NaN → set pval = 0.8
df_long["pval"] = df_long["HR"].apply(lambda x: 0.001 if pd.notnull(x) else 0.8)

# Reorder columns as requested
result_df = df_long[["Outcome", "Category", "HR", "pval"]]

# fill missing HR wiht 1
result_df = result_df.fillna(1)

# Display the resulting dataframe
result_df
# result_df.to_csv('./exp_data/{}/{}/{}/{}_testing_testing.csv'.format(cancer_type, therapy_version, 'tem_result', cancer_type))

# COMMAND ----------

result_df

# COMMAND ----------

##  Circular Summary Plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# --- Synthetic example dataset ---
np.random.seed(42)
# n_outcomes = 216
# categories = ["Circulatory", "Digestive", "Metabolic", "Infectious", "Respiratory", "Neoplasm"]

# df = pd.DataFrame({
#     "Outcome": [f"Outcome {i+1}" for i in range(n_outcomes)],
#     "Category": np.random.choice(categories, n_outcomes),
#     "HR": np.random.uniform(0.6, 1.6, n_outcomes),
#     "pval": np.random.uniform(0.0001, 0.2, n_outcomes)
# })

n_outcomes = len(result_df.index)
categories = ['Female', 'Male', 'age<65', 'age>=65', 'Total Metastasis number=0',
       'Total Metastasis number=1', 'Total Metastasis number>=2',
       'Metastasis site at the lung', 'Metastasis site in the brain',
       'Metastasis site at the bone', 'Metastasis site at the liver',
       'Metastasis site at the adrenal glands']
df = result_df.copy()

# df_raw = df.copy()


# derive color
def get_color(hr, p):
    if p < 0.05:
        return "#FF6666" if hr > 1 else "green"
    return "lightgray"

df["color"] = [get_color(hr, p) for hr, p in zip(df["HR"], df["pval"])]

# order by category
df = df.sort_values("Category").reset_index(drop=True)

# Adjust angles to include a wrap‑around gap between categories, including between last and first
gap = 0.03  # radians gap between categories
n_categories = df["Category"].nunique()
total_gap = gap * n_categories
available_angle = 2 * np.pi - total_gap

start = 0.0
theta_vals = np.empty(len(df))

# df is already sorted by Category, so groupby preserves order
for cat, group in df.groupby("Category", sort=False):
    n = len(group)
    span = (n / len(df)) * available_angle
    angles = np.linspace(start, start + span, n, endpoint=False)
    theta_vals[group.index] = angles
    start += span + gap  # add gap before next category (including after last)

df["theta"] = theta_vals


# --- Start plot ---
fig = plt.figure(figsize=(12,12))
ax = plt.subplot(111, polar=True)
ax.set_theta_direction(-1)
ax.set_theta_zero_location("N")

# ring 1: HR heatmap ring (radius ~ 1)
for i, row in df.iterrows():
    bar = ax.bar(
        row["theta"],
        0.5,              # radial width of the bar
        width=(2*np.pi)/n_outcomes*0.9,
        bottom=4.0,       # radius position of this ring
        color=row["color"],
        edgecolor='none'
    )

# ring 2: HR magnitude bars (for HR>1)
for i, row in df.iterrows():
    mag = row["HR"] - 1 if row["HR"] > 1 else 1 - row["HR"]
    height = np.clip(mag, 0, 0.7)
    color = "#FF6666" if (row["HR"] > 1 and row["pval"]<0.05) else ("lightgray" if (row["HR"]<1 and row["pval"]<0.05) else "lightgray") # "#FF6666" ---deep red
    ax.bar(row["theta"], height, width=(2*np.pi)/n_outcomes*0.7, bottom=3.0, color=color, edgecolor='none')

# ring 3: HR magnitude bars (for HR<1)
for i, row in df.iterrows():
    mag = row["HR"] - 1 if row["HR"] > 1 else 1 - row["HR"]
    height = np.clip(mag, 0, 0.7)
    color = "lightgray" if (row["HR"] > 1 and row["pval"]<0.05) else ("green" if (row["HR"]<1 and row["pval"]<0.05) else "lightgray")
    ax.bar(row["theta"], height, width=(2*np.pi)/n_outcomes*0.7, bottom=2.0, color=color, edgecolor='none')

# inner ring: -log10(p-value)
# for i, row in df.iterrows():
#     val = -np.log10(row["pval"])
#     h = np.clip(val/5, 0, 1.0)
#     color = "gold" if row["pval"]<0.05 else "gray"
#     ax.bar(row["theta"], h, width=(2*np.pi)/n_outcomes*0.7, bottom=2.0, color=color, edgecolor='none')


# --- outermost ring: visual separation by category ---
# assign a distinct color to each category
cat_list = df["Category"].unique()
# palette = plt.cm.tab20.colors  # enough distinct colors
# cat_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(cat_list)}
# set all category colors to black
cat_colors = {cat: "black" for cat in cat_list}

# draw a thin colored arc for each category just outside the label radius
for cat, group in df.groupby("Category", sort=False):
    start_angle = group["theta"].min()
    # add a small extra width to cover the last bar of the group
    extra = (2 * np.pi) / n_outcomes * 0.9
    end_angle = group["theta"].max() + extra
    span = end_angle - start_angle
    mid_angle = start_angle + span / 2

    ax.bar(
        mid_angle,
        0.02,                     # thickness of the outer ring
        width=span,
        bottom=5.8,              # just outside the category label radius
        color=cat_colors[cat],
        edgecolor="none",
    )

# hide the central cross line (angular grid line at 0°)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.spines["polar"].set_visible(False)

ax.set_ylim(0, 6)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title("Systematic evaluation of risks and benefits (simplified Fig. 4 style)", pad=40)

# save_path = save_path
save_path = "./exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_HR_dod_stratify_circular_summary_plot.pdf"
plt.savefig(save_path, dpi=500, format="pdf", bbox_inches='tight')

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # stratify analysis ---SAE--odds ratio, incidence rate difference

# COMMAND ----------

# plot sub-sae--stratify
def fun_plotting_sub_sae(data_raw, fig_title, center_val, save_path):
    df = data_raw.copy()
    df = df.fillna(np.nan)

    #Define colormap: green (<0), white (=0), red (>0)
    cmap = sns.diverging_palette(150, 10, as_cmap=True)
    cmap.set_bad(color="gray")  # NaN values → gray

    # Normalize around 0
    norm = TwoSlopeNorm(vmin=np.nanmin(df.values), vcenter=center_val, vmax=np.nanmax(df.values))

    # Plot heatmap
    plt.figure(figsize=(figure_size_length, figure_size_width))
    ax = sns.heatmap(df, annot=True, cmap=cmap, norm=norm, cbar=True, fmt='.3g')

    # Put column labels on top and rotate 45°
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=85)

    plt.title(fig_title, pad=40)
    plt.savefig(save_path, dpi=500, format="pdf", bbox_inches='tight')

    # plt.show()
    plt.close()


# COMMAND ----------

threshold_balanced_covariate = 20
threshold_balanced_trial = 10
unbalanced_covariates = 'unbalanced_covariates_02' # smd>0.2; unbalanced_covariates_03  smd>0.3

# Exclude FAMOTIDINE from the selected drug list
df_sel_drug_list = df_sel_drug_list[~df_sel_drug_list['generic_desc'].isin(['FAMOTIDINE'])].reset_index(drop=True) # it is not balanced

sel_sub_stratify_list = ['gender_0', 'gender_1', 'age_less_65', 'age_large_65', 'meta_total_0', 'meta_total_1', 'meta_total_2', 
                        'meta_lung_1', 'meta_brain_1', 'meta_bone_1',  'meta_liver_1', 'meta_adrenal_gland_1']

# testing
#sel_sub_sae_list = ['Vascular disorders'] #'Vascular disorders'
# df_sel_drug_list = df_sel_drug_list[:1] #['DEXAMETHASONE']
#sel_sub_stratify_list = ['gender_0']

results_df_OR_allday_stratify = pd.DataFrame(columns= sel_sub_stratify_list, index=list(df_sel_drug_list['generic_desc']))
results_df_OR_180day_stratify = pd.DataFrame(columns= sel_sub_stratify_list, index=list(df_sel_drug_list['generic_desc']))
results_df_IRD_allday_stratify = pd.DataFrame(columns= sel_sub_stratify_list, index=list(df_sel_drug_list['generic_desc']))
results_df_IRD_180day_stratify = pd.DataFrame(columns= sel_sub_stratify_list, index=list(df_sel_drug_list['generic_desc']))

for sub_sae in sel_sub_sae_list:
    print('sub_sae:--------------------------------')
    print('sub_sae:',sub_sae)

    for sub_stratify in sel_sub_stratify_list:
        print('sub_stratify:',sub_stratify)

        for idx in range(len(df_sel_drug_list)):
            drug_name = df_sel_drug_list.iloc[idx]['generic_desc']
            drug_key = drug_name.replace('/', '-')
            data = pd.read_csv('./exp_data/{}/{}/{}/{}/results_v0_sae_hr_odds_ird_stratify_{}.csv'.format(cancer_type, therapy_version, drug_key, sub_sae, sub_stratify))

            n_balanced_trial = len(data.loc[data[unbalanced_covariates] < threshold_balanced_covariate])
            if n_balanced_trial>=threshold_balanced_trial:
                # OR---all day
                OR_allday_value_list = data.loc[data[unbalanced_covariates] < threshold_balanced_covariate]['OR_all_day'].values
                OR_allday_value_list = OR_allday_value_list[np.isfinite(OR_allday_value_list)] ## remove inf from OR_allday_value_list
                OR_allday_value_mean = float('%.3f' % np.mean(np.array(OR_allday_value_list)))
                OR_allday_p_value, _ = bootstrap_mean_pvalue(np.array(OR_allday_value_list), expected_mean=1)

                # OR---180 day
                OR_180day_value_list = data.loc[data[unbalanced_covariates] < threshold_balanced_covariate]['OR_180day'].values
                OR_180day_value_list = OR_180day_value_list[np.isfinite(OR_180day_value_list)]
                OR_180day_value_mean = float('%.3f' % np.mean(np.array(OR_180day_value_list)))
                OR_180day_p_value, _ = bootstrap_mean_pvalue(np.array(OR_180day_value_list), expected_mean=1)

                # difference of incidence rate---all day
                IRD_allday_value_list = data.loc[data[unbalanced_covariates] < threshold_balanced_covariate]['diff_ir_all_day'].values
                IRD_allday_value_list = IRD_allday_value_list[np.isfinite(IRD_allday_value_list)]
                IRD_allday_value_mean = float('%.3f' % np.mean(np.array(IRD_allday_value_list)))
                IRD_allday_p_value, _ = bootstrap_mean_pvalue(np.array(IRD_allday_value_list), expected_mean=0)

                # difference of incidence rate---180 day
                IRD_180day_value_list = data.loc[data[unbalanced_covariates] < threshold_balanced_covariate]['diff_ir_180day'].values
                IRD_180day_value_list = IRD_180day_value_list[np.isfinite(IRD_180day_value_list)]
                IRD_180day_value_mean = float('%.3f' % np.mean(np.array(IRD_180day_value_list)))
                IRD_180day_p_value, _ = bootstrap_mean_pvalue(np.array(IRD_180day_value_list), expected_mean=0)

                if OR_allday_p_value<0.05:
                    results_df_OR_allday_stratify.loc[drug_name, sub_stratify] = OR_allday_value_mean
                if OR_180day_p_value<0.05:
                    results_df_OR_180day_stratify.loc[drug_name, sub_stratify] = OR_180day_value_mean
                if IRD_allday_p_value<0.05:
                    results_df_IRD_allday_stratify.loc[drug_name, sub_stratify] = IRD_allday_value_mean
                if IRD_180day_p_value<0.05:
                    results_df_IRD_180day_stratify.loc[drug_name, sub_stratify] = IRD_180day_value_mean

    results_df_OR_180day_stratify.to_csv('./exp_data/{}/{}/{}/{}_OR_sae_{}_{}_{}.csv'.format(cancer_type, therapy_version, 'tem_result', cancer_type, 'stratify', sub_sae, '180day'))
    results_df_IRD_180day_stratify.to_csv('./exp_data/{}/{}/{}/{}_IRD_sae_{}_{}_{}.csv'.format(cancer_type, therapy_version, 'tem_result', cancer_type, 'stratify', sub_sae, '180day'))

    # plot figure
    OR_figure_title = "Cohort ANSCLC: OR on outcome " + sub_sae + " (Green < 1, White =1, Red > 1, Gray (p-value>0.05))"
    OR_save_path = './exp_data/{}/{}/{}/{}_OR_sae_{}_{}_{}.pdf'.format(cancer_type, therapy_version, 'tem_result', cancer_type, 'stratify', sub_sae, '180day')
    OR_center_value_plot = 1
    IRD_figure_title = "Cohort ANSCLC: IRD on outcome " + sub_sae + " (Green < 0, White =0, Red > 0, Gray (p-value>0.05))"
    IRD_save_path = './exp_data/{}/{}/{}/{}_IRD_sae_{}_{}_{}.pdf'.format(cancer_type, therapy_version, 'tem_result', cancer_type, 'stratify', sub_sae, '180day')
    IRD_center_value_plot = 0
    fun_plotting_sub_sae(results_df_OR_180day_stratify, OR_figure_title, OR_center_value_plot, OR_save_path)
    fun_plotting_sub_sae(results_df_IRD_180day_stratify, IRD_figure_title, IRD_center_value_plot, IRD_save_path)
    

# COMMAND ----------

results_df_OR_180day_stratify

# COMMAND ----------


# sel_sub_sae_list = ['Blood and lymphatic system disorders',
#  'Cardiac disorders',
#  'Ear and labyrinth disorders',
#  'Endocrine disorders',
#  'Gastrointestinal disorders',
#  'General disorders and administration site conditions',
#  'Infections and infestations',
#  'Injury, poisoning and procedural complications',
#  'Metabolism and nutrition disorders',
#  'Musculoskeletal and connective tissue disorders',
#  'Nervous system disorders',
#  'Psychiatric disorders',
#  'Renal and urinary disorders',
#  'Respiratory, thoracic and mediastinal disorders',
#  'Skin and subcutaneous tissue disorders',
#  'Vascular disorders'] 

# COMMAND ----------

# MAGIC %md
# MAGIC # Forest plot

# COMMAND ----------

# MAGIC %md
# MAGIC ## An example for Forest plot

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your real results)
data = {
    "Outcome": [
        "All-cause mortality",
        "Cardiovascular events",
        "Renal failure",
        "Hypoglycemia",
        "Pancreatitis",
        "Fractures",
    ],
    "HR": [0.82, 0.91, 0.74, 1.15, 1.32, 1.05],
    "CI_lower": [0.76, 0.84, 0.61, 1.02, 1.10, 0.91],
    "CI_upper": [0.89, 0.98, 0.90, 1.29, 1.58, 1.20],
    "pvalue": [0.0006, 0.002, 0.01, 0.08, 0.03, 0.40],
}
df = pd.DataFrame(data)

# Sort outcomes (optional)
df = df.sort_values("HR")

# Define colors based on HR and p-value
def get_color(row):
    if row["pvalue"] < 0.05:
        if row["HR"] > 1:
            return "pink"   # significantly harmful
        elif row["HR"] < 1:
            return "green" # significantly protective
    return "gray"          # non-significant

df["color"] = df.apply(get_color, axis=1)

# Create figure
plt.figure(figsize=(8, 4.5))

# Plot horizontal CIs and HR points
for i, row in df.iterrows():
    plt.errorbar(
        row["HR"],
        row["Outcome"],
        xerr=[[row["HR"] - row["CI_lower"]], [row["CI_upper"] - row["HR"]]],
        fmt="o",
        color=row["color"],
        ecolor=row["color"],
        elinewidth=1,
        capsize=3,
    )

# Add vertical reference line at HR=1
plt.axvline(x=1, color="black", linestyle="--", linewidth=1)

# Log scale (recommended for HR)
# plt.xscale("log")

# Labeling
plt.xlabel("Hazard Ratio (log scale)")
plt.title("Forest Plot of Hazard Ratios by Significance and Direction")
plt.tight_layout()

plt.show()
# save_path = ''
# plt.savefig(save_path, dpi=500, format="pdf", bbox_inches='tight')

# plt.show()
plt.close()



# COMMAND ----------

df

# COMMAND ----------

# please read a file, /Workspace/Users/zhenxing.xu@regeneron.com/ansclc_0915/exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_HR_p_value_dod_forest_plot.xlsx
df_ansclc = pd.read_excel("/Workspace/Users/zhenxing.xu@regeneron.com/ansclc_0915/exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_HR_p_value_dod_forest_plot.xlsx")

# based on column 95% CI to generate two columns, CI_lower, CI_upper
df_ansclc['CI_lower'] = df_ansclc['95% CI'].apply(lambda x: float(x.split(',')[0].split('[')[1]))
df_ansclc['CI_upper'] = df_ansclc['95% CI'].apply(lambda x: float(x.split(',')[1].split(']')[0]))
df_ansclc

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Forest Plot for Overall Survival--Hazard Ratio

# COMMAND ----------

df = df_ansclc.copy()

# df = df.sort_values("HR")

# Define colors based on HR and p-value
def get_color(row):
    if row["p-value"] == '<0.0001':
        if row["HR"] > 1:
            return "#FF6666"  # shallow red "#8B0000"  # deep red  # significantly harmful
        elif row["HR"] < 1:
            return "green" # significantly protective
    return "gray"          # non-significant


df["color"] = df.apply(get_color, axis=1)

# Create figure
plt.figure(figsize=(12, 7)) # (8, 4.5)

# Plot horizontal CIs and HR points
for i, row in df.iterrows():
    plt.errorbar(
        row["HR"],
        row["Drug"],
        xerr=[[row["HR"] - row["CI_lower"]], [row["CI_upper"] - row["HR"]]],
        fmt="o",
        color=row["color"],
        ecolor=row["color"],
        elinewidth=1,
        capsize=2,
    )

# Add vertical reference line at HR=1
plt.axvline(x=1, color="black", linestyle="--", linewidth=1)

# Log scale (recommended for HR)
# plt.xscale("log")

# Set x-axis limits and format (e.g., start at 0.600)
xmin = 0.6
xmax = df["CI_upper"].max() * 1.002  # add a small margin
plt.xlim(xmin, xmax)
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))


# Labeling
plt.xlabel("Hazard Ratio (HR)")
plt.title("Forest Plot for the Effects of Supportive Care Drugs on OS in aNSCLC")
plt.tight_layout()

# save_path = save_path
save_path = "/Workspace/Users/zhenxing.xu@regeneron.com/ansclc_0915/exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_HR_p_value_dod_forest_plot.pdf"

# plt.savefig(save_path, dpi=500, format="pdf", bbox_inches='tight')

plt.show()
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Circular Summary Plot for Overall Survival--Hazard Ratio

# COMMAND ----------

df_ansclc_tem = df_ansclc.copy()
# reverse the order of rows in drug_info
df_ansclc_tem = df_ansclc_tem.iloc[::-1]
rename = {'p-value': 'pval'}
df_ansclc_tem.rename(columns=rename, inplace=True)

df_ansclc_tem

# COMMAND ----------

##  Circular Summary Plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# --- Synthetic example dataset ---
np.random.seed(42)
# n_outcomes = 216
# categories = ["Circulatory", "Digestive", "Metabolic", "Infectious", "Respiratory", "Neoplasm"]

# df = pd.DataFrame({
#     "Outcome": [f"Outcome {i+1}" for i in range(n_outcomes)],
#     "Category": np.random.choice(categories, n_outcomes),
#     "HR": np.random.uniform(0.6, 1.6, n_outcomes),
#     "pval": np.random.uniform(0.0001, 0.2, n_outcomes)
# })

n_outcomes = len(df_ansclc_tem.index)

# change the category of Oxycodone from Analgesic to Analgesic / Antipyretic
df_ansclc_tem = df_ansclc_tem.replace({'Category': {'Analgesic': 'Analgesic / Antipyretic'}})

categories = ['Vitamin', 'Hematologic support', 'Electrolyte replacement',
       'Corticosteroid', 'Bone support', 'Anxiolytic', 'Antihistamine',
       'Antiemetic', 'Anticoagulant', 'Analgesic / Antipyretic', 'Acid suppression']
df = df_ansclc_tem.copy()

set_color_1 = "lightcoral" # HR >1
set_color_2 = 'seagreen' # HR <1

# derive color
def get_color(hr, p):
    if p == '<0.0001':
        return set_color_1 if hr > 1 else set_color_2
    return "lightgray"

df["color"] = [get_color(hr, p) for hr, p in zip(df["HR"], df["pval"])]

# order by category
df = df.sort_values("Category").reset_index(drop=True)

# Adjust angles to include a wrap‑around gap between categories, including between last and first
gap = 0.03  # radians gap between categories
n_categories = df["Category"].nunique()
total_gap = gap * n_categories
available_angle = 2 * np.pi - total_gap

start = 0.0
theta_vals = np.empty(len(df))

# df is already sorted by Category, so groupby preserves order
for cat, group in df.groupby("Category", sort=False):
    n = len(group)
    span = (n / len(df)) * available_angle
    angles = np.linspace(start, start + span, n, endpoint=False)
    theta_vals[group.index] = angles
    start += span + gap  # add gap before next category (including after last)

df["theta"] = theta_vals

# --- Start plot ---
fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
ax.set_theta_direction(-1)
ax.set_theta_zero_location("N")

# ring 1: HR heatmap ring (radius ~ 1)
for i, row in df.iterrows():
    bar = ax.bar(
        row["theta"],
        0.5,              # radial width of the bar
        width=(2*np.pi)/n_outcomes*0.9,
        bottom=4.0,       # radius position of this ring
        color=row["color"],
        edgecolor='none'
    )

# ring 2: HR magnitude bars (for HR>1)
for i, row in df.iterrows():
    # mag = row["HR"] - 1 if row["HR"] > 1 else 1 - row["HR"]
    mag = row["HR"] if row["HR"] > 1 else 1/row["HR"]
    # height = np.clip(mag, 0, 0.7)
    # please keep height wiht mag orignial values
    height = mag
    color = set_color_1 if (row["HR"] > 1 and row["pval"]=='<0.0001') else ("lightgray" if (row["HR"]<1 and row["pval"]=='<0.0001') else "lightgray") # "#FF6666" ---deep red
    ax.bar(row["theta"], height, width=(2*np.pi)/n_outcomes*0.8, bottom=2.0, color=color, edgecolor='none')

# ring 3: HR magnitude bars (for HR<1)
for i, row in df.iterrows():
    mag = row["HR"] if row["HR"] > 1 else 1/row["HR"]
    # height = np.clip(mag, 0, 0.7)
    height = mag
    color = "lightgray" if (row["HR"] > 1 and row["pval"]=='<0.0001') else (set_color_2 if (row["HR"]<1 and row["pval"]=='<0.0001') else "lightgray")
    ax.bar(row["theta"], height, width=(2*np.pi)/n_outcomes*0.7, bottom=0.2, color=color, edgecolor='none')

# --- outermost ring: visual separation by category ---
cat_list = df["Category"].unique()
cat_colors = {cat: "black" for cat in cat_list}

for cat, group in df.groupby("Category", sort=False):
    start_angle = group["theta"].min()
    # extra width to ensure the arc covers the last bar of the group
    extra = (2 * np.pi) / n_outcomes * 0.9
    end_angle = group["theta"].max() + extra
    span = end_angle - start_angle
    mid_angle = start_angle + span / 2

    ax.bar(
        mid_angle-0.15,
        0.03,                     # thickness of the outer ring
        width=span,
        bottom=5.8,               # positioned just outside the outermost data rings
        color=cat_colors[cat],
        edgecolor="none",
    )

# hide the central cross line (angular grid line at 0°)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.spines["polar"].set_visible(False)

ax.set_ylim(0, 6)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title("The Effects of Supportive Care Drugs on OS in aNSCLC", pad=40)

# save_path = save_path
save_path = "./exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_HR_dod_circular_summary_plot.pdf"

plt.savefig(save_path, dpi=500, format="pdf", bbox_inches='tight')

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Circular Summary Plot

# COMMAND ----------

start_angle

# COMMAND ----------

# MAGIC %md
# MAGIC ## an example 1

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# --- Synthetic example dataset ---
np.random.seed(42)
n_outcomes = 36
categories = ["Circulatory", "Digestive", "Metabolic", "Infectious", "Respiratory", "Neoplasm"]
df = pd.DataFrame({
    "Outcome": [f"Outcome {i+1}" for i in range(n_outcomes)],
    "Category": np.random.choice(categories, n_outcomes),
    "HR": np.random.uniform(0.6, 1.6, n_outcomes),
    "pval": np.random.uniform(0.0001, 0.2, n_outcomes)
})

df_raw = df.copy()
# derive color
def get_color(hr, p):
    if p < 0.05:
        return "red" if hr > 1 else "blue"
    return "lightgray"

df["color"] = [get_color(hr, p) for hr, p in zip(df["HR"], df["pval"])]

# order by category
df = df.sort_values("Category").reset_index(drop=True)

# assign angles (around circle)
theta = np.linspace(0, 2*np.pi, n_outcomes, endpoint=False)
df["theta"] = theta

# --- Start plot ---
fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
ax.set_theta_direction(-1)
ax.set_theta_zero_location("N")

# ring 1: HR heatmap ring (radius ~ 1)
for i, row in df.iterrows():
    bar = ax.bar(
        row["theta"],
        0.5,              # radial width of the bar
        width=(2*np.pi)/n_outcomes*0.9,
        bottom=4.0,       # radius position of this ring
        color=row["color"],
        edgecolor='none'
    )

# ring 2: HR magnitude bars (for HR>1 and HR<1)
for i, row in df.iterrows():
    mag = row["HR"] - 1 if row["HR"] > 1 else 1 - row["HR"]
    height = np.clip(mag, 0, 0.7)
    color = "red" if (row["HR"] > 1 and row["pval"]<0.05) else ("blue" if (row["HR"]<1 and row["pval"]<0.05) else "gray")
    ax.bar(row["theta"], height, width=(2*np.pi)/n_outcomes*0.7, bottom=3.0, color=color, edgecolor='none')

# inner ring: -log10(p-value)
for i, row in df.iterrows():
    val = -np.log10(row["pval"])
    h = np.clip(val/5, 0, 1.0)
    color = "gold" if row["pval"]<0.05 else "gray"
    ax.bar(row["theta"], h, width=(2*np.pi)/n_outcomes*0.7, bottom=2.0, color=color, edgecolor='none')

# category labels (outermost ring)
cat_angles = df.groupby("Category")["theta"].mean()
for cat, ang in cat_angles.items():
    ax.text(ang, 5.5, cat, ha='center', va='center', fontsize=10, fontweight='bold', rotation=np.degrees(ang)-90)

ax.set_ylim(0, 6)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title("Systematic evaluation of risks and benefits (simplified Fig. 4 style)", pad=40)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## an example 2

# COMMAND ----------

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# import matplotlib.cm as cm

# # --- Synthetic example dataset ---
# np.random.seed(42)
# n_outcomes = 216
# categories = ["Circulatory", "Digestive", "Metabolic", "Infectious", "Respiratory", "Neoplasm"]

# df = pd.DataFrame({
#     "Outcome": [f"Outcome {i+1}" for i in range(n_outcomes)],
#     "Category": np.random.choice(categories, n_outcomes),
#     "HR": np.random.uniform(0.6, 1.6, n_outcomes),
#     "pval": np.random.uniform(0.0001, 0.2, n_outcomes)
# })

# df_raw = df.copy()
# # derive color
# def get_color(hr, p):
#     if p < 0.05:
#         return "#FF6666" if hr > 1 else "green"
#     return "lightgray"

# df["color"] = [get_color(hr, p) for hr, p in zip(df["HR"], df["pval"])]

# # order by category
# df = df.sort_values("Category").reset_index(drop=True)

# # Adjust angles to include a wrap‑around gap between categories, including between last and first
# gap = 0.03  # radians gap between categories
# n_categories = df["Category"].nunique()
# total_gap = gap * n_categories
# available_angle = 2 * np.pi - total_gap

# start = 0.0
# theta_vals = np.empty(len(df))

# # df is already sorted by Category, so groupby preserves order
# for cat, group in df.groupby("Category", sort=False):
#     n = len(group)
#     span = (n / len(df)) * available_angle
#     angles = np.linspace(start, start + span, n, endpoint=False)
#     theta_vals[group.index] = angles
#     start += span + gap  # add gap before next category (including after last)

# df["theta"] = theta_vals


# # --- Start plot ---
# fig = plt.figure(figsize=(12,12))
# ax = plt.subplot(111, polar=True)
# ax.set_theta_direction(-1)
# ax.set_theta_zero_location("N")

# # ring 1: HR heatmap ring (radius ~ 1)
# for i, row in df.iterrows():
#     bar = ax.bar(
#         row["theta"],
#         0.5,              # radial width of the bar
#         width=(2*np.pi)/n_outcomes*0.9,
#         bottom=4.0,       # radius position of this ring
#         color=row["color"],
#         edgecolor='none'
#     )

# # ring 2: HR magnitude bars (for HR>1)
# for i, row in df.iterrows():
#     mag = row["HR"] - 1 if row["HR"] > 1 else 1 - row["HR"]
#     height = np.clip(mag, 0, 0.7)
#     color = "#FF6666" if (row["HR"] > 1 and row["pval"]<0.05) else ("lightgray" if (row["HR"]<1 and row["pval"]<0.05) else "lightgray") # "#FF6666" ---deep red
#     ax.bar(row["theta"], height, width=(2*np.pi)/n_outcomes*0.7, bottom=3.0, color=color, edgecolor='none')

# # ring 3: HR magnitude bars (for HR<1)
# for i, row in df.iterrows():
#     mag = row["HR"] - 1 if row["HR"] > 1 else 1 - row["HR"]
#     height = np.clip(mag, 0, 0.7)
#     color = "lightgray" if (row["HR"] > 1 and row["pval"]<0.05) else ("green" if (row["HR"]<1 and row["pval"]<0.05) else "lightgray")
#     ax.bar(row["theta"], height, width=(2*np.pi)/n_outcomes*0.7, bottom=2.0, color=color, edgecolor='none')

# # inner ring: -log10(p-value)
# # for i, row in df.iterrows():
# #     val = -np.log10(row["pval"])
# #     h = np.clip(val/5, 0, 1.0)
# #     color = "gold" if row["pval"]<0.05 else "gray"
# #     ax.bar(row["theta"], h, width=(2*np.pi)/n_outcomes*0.7, bottom=2.0, color=color, edgecolor='none')


# # --- outermost ring: visual separation by category ---
# # assign a distinct color to each category
# cat_list = df["Category"].unique()
# # palette = plt.cm.tab20.colors  # enough distinct colors
# # cat_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(cat_list)}
# # set all category colors to black
# cat_colors = {cat: "black" for cat in cat_list}

# # draw a thin colored arc for each category just outside the label radius
# for cat, group in df.groupby("Category", sort=False):
#     start_angle = group["theta"].min()
#     # add a small extra width to cover the last bar of the group
#     extra = (2 * np.pi) / n_outcomes * 0.9
#     end_angle = group["theta"].max() + extra
#     span = end_angle - start_angle
#     mid_angle = start_angle + span / 2

#     ax.bar(
#         mid_angle,
#         0.02,                     # thickness of the outer ring
#         width=span,
#         bottom=5.8,              # just outside the category label radius
#         color=cat_colors[cat],
#         edgecolor="none",
#     )

# # hide the central cross line (angular grid line at 0°)
# ax.xaxis.grid(False)
# ax.yaxis.grid(False)
# ax.spines["polar"].set_visible(False)

# ax.set_ylim(0, 6)
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_title("Systematic evaluation of risks and benefits (simplified Fig. 4 style)", pad=40)

# # save_path = save_path
# save_path = "./exp_data/ansclc/combined_therapy_pres_adm_pro_14_90/tem_result/ansclc_HR_dod_stratify_circular_summary_plot.pdf"
# plt.savefig(save_path, dpi=500, format="pdf", bbox_inches='tight')

# plt.show()


# COMMAND ----------

df_raw

# COMMAND ----------

results_df_stratify_tem

# COMMAND ----------

import pandas as pd

# Reset index to turn the drug names (index) into a column
df_long = (
    results_df_stratify_tem
    .reset_index()
    .melt(id_vars=results_df_stratify_tem.index.name or "index",
          var_name="Category",
          value_name="HR")
)

# Rename the index column to "Outcome"
df_long = df_long.rename(columns={results_df_stratify_tem.index.name or "index": "Outcome"})

# Create p‑value column:
#   - if the HR (or original p‑value) is not NaN → set pval = 0.001
#   - if it is NaN → set pval = 0.8
df_long["pval"] = df_long["HR"].apply(lambda x: 0.001 if pd.notnull(x) else 0.8)

# Reorder columns as requested
result_df = df_long[["Outcome", "Category", "HR", "pval"]]

# fill missing HR wiht 1
results_df_stratify_tem = results_df_stratify_tem.fillna(1)

# Display the resulting dataframe
result_df


# COMMAND ----------

print('DONE.')

# COMMAND ----------

# df_sel_drug_list = pd.read_csv("../../zhenxing.xu@regeneron.com/crc_0805/pre_data/cohort/sel_drug_list.csv".format(cancer_type, therapy_version))

# COMMAND ----------

# with open("crc_hr", "w") as fout:
#     for idx in range(2, len(df_sel_drug_list)):
#         drug_name = df_sel_drug_list.iloc[idx]['generic_desc']
#         cnt = df_sel_drug_list.iloc[idx]['count']
#         if drug_name in ["PEMETREXED DISODIUM"]:
#             continue
#         drug_key = drug_name.replace('/', '-')
#         results = pd.read_csv('../../zhenxing.xu@regeneron.com/crc_0805/pre_data/cohort/{}/results_v0.csv'.format(drug_key))
#         success_result = []
#         for idx in range(len(results)):
#             if results.iloc[idx]['unbalanced_covariates'] < 3:
#                 success_result.append(results.iloc[idx]['HR'])
#         if len(success_result) == 0:
#             print(drug_name, cnt, results['ori_unbalanced_covariates'].values.mean(), results['unbalanced_covariates'].values.mean(), len(success_result) / 100)
#             fout.write("{}#{}#{}#{}#{}\n".format(drug_name, cnt, results['ori_unbalanced_covariates'].values.mean(), results['unbalanced_covariates'].values.mean(), len(success_result) / 100))
#         else:
#             print(drug_name, cnt, results['ori_unbalanced_covariates'].values.mean(), results['unbalanced_covariates'].values.mean(), len(success_result) / 100, np.percentile(success_result, 50), np.percentile(success_result, 5), np.percentile(success_result, 95))
#             fout.write("{}#{}#{}#{}#{}#{:.2f}({:.2f},{:.2f})\n".format(drug_name, cnt, results['ori_unbalanced_covariates'].values.mean(), results['unbalanced_covariates'].values.mean(), len(success_result) / 100, np.percentile(success_result, 50), np.percentile(success_result, 5), np.percentile(success_result, 95)))


# COMMAND ----------

# with open("crc_sae", "w") as fout:
#     for idx in range(2, len(df_sel_drug_list)):
#         drug_name = df_sel_drug_list.iloc[idx]['generic_desc']
#         cnt = df_sel_drug_list.iloc[idx]['count']
#         if drug_name in ["PEMETREXED DISODIUM"]:
#             continue
#         drug_key = drug_name.replace('/', '-')
#         results = pd.read_csv('../../zhenxing.xu@regeneron.com/crc_0805/pre_data/cohort/{}/results_v0_sae.csv'.format(drug_key))
#         success_result = []
#         for idx in range(len(results)):
#             if results.iloc[idx]['unbalanced_covariates'] < 3:
#                 success_result.append(results.iloc[idx]['HR'])
#         if len(success_result) == 0:
#             print(drug_name, cnt, results['ori_unbalanced_covariates'].values.mean(), results['unbalanced_covariates'].values.mean(), len(success_result) / 100)
#             fout.write("{}#{}#{}#{}#{}\n".format(drug_name, cnt, results['ori_unbalanced_covariates'].values.mean(), results['unbalanced_covariates'].values.mean(), len(success_result) / 100))
#         else:
#             print(drug_name, cnt, results['ori_unbalanced_covariates'].values.mean(), results['unbalanced_covariates'].values.mean(), len(success_result) / 100, np.percentile(success_result, 50), np.percentile(success_result, 5), np.percentile(success_result, 95))
#             fout.write("{}#{}#{}#{}#{}#{:.2f}({:.2f},{:.2f})\n".format(drug_name, cnt, results['ori_unbalanced_covariates'].values.mean(), results['unbalanced_covariates'].values.mean(), len(success_result) / 100, np.percentile(success_result, 50), np.percentile(success_result, 5), np.percentile(success_result, 95)))
