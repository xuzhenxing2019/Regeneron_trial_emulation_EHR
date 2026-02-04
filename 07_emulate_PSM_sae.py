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
from ps import PropensityEstimator, final_eval_ml_CV_revise, final_eval_ml_CV_revise_binary
import pickle
# from emulation_utils import perform_psm, perform_balancing_method, check_balance_after_matching
from ps import model_eval_common_simple
from lifelines import CoxPHFitter

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

cancer_type = "ansclc"
therapy_version = "combined_therapy_pres_adm_pro_14_90"

# COMMAND ----------

import copy
import sklearn
from sklearn import linear_model
import lifelines
import scipy
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns
import causallib
from causallib.estimation import IPW, Standardization, StratifiedStandardization
from causallib.estimation import AIPW, PropensityFeatureStandardization, WeightedStandardization
from causallib.evaluation import evaluate
from sklearn.model_selection import train_test_split
# from sksurv.ensemble import RandomSurvivalForest


def check_balance_after_matching(X, all_vars, treatment_col):
    feature_treatment = X[X[treatment_col] == 1][all_vars]
    feature_control = X[X[treatment_col] == 0][all_vars]
    treatment_mean = feature_treatment.mean(0)
    treatment_std = feature_treatment.std(0)
    control_mean = feature_control.mean(0)
    control_std = feature_control.std(0)
    SMD = np.abs(treatment_mean-control_mean) / np.sqrt((treatment_std**2 + control_std**2)/2)
    return SMD

def mad(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def calculate_propensity_scores(X, treatment_col, covariate_list):
    model = linear_model.LogisticRegression()
    model.fit(X[covariate_list], X[treatment_col])
    propensity_scores = model.predict_proba(X[covariate_list])[:, 1]
    X['ps'] = propensity_scores
    return X

def match_pairs(distances, treated_idx, control_idx, N=1, caliper=None):
    matched_pairs = []
    control_pool = copy.deepcopy(control_idx)
    distance_pool = copy.deepcopy(distances)
    matches_per_treated = dict.fromkeys(treated_idx, 0)
    unmatchable = set()
    while len(control_pool) > 0:
        for i, treated in enumerate(treated_idx):
            if treated in unmatchable or matches_per_treated[treated] >= N:
                continue
            if control_pool.empty:
                break
            min_index = np.argmin(distance_pool[i])
            min_distance = distance_pool[i, min_index]
            if caliper is not None and min_distance > caliper:
                unmatchable.add(treated)
                continue
            matched_control = control_pool[min_index]
            matched_pairs.append((treated, matched_control))
            matches_per_treated[treated] += 1
            control_pool = control_pool.drop(matched_control)
            distance_pool = np.delete(distance_pool, min_index, axis=1)
        if min(val for key, val in matches_per_treated.items() if key not in unmatchable) >= N:
            break
    return matched_pairs


def calculate_distance_propensity(X, treated_idx, control_idx, covariates, treatment_col):
    X_ps = calculate_propensity_scores(X, treatment_col, covariates)
    treated_data = X_ps.loc[treated_idx, 'ps'].values.reshape(-1, 1)
    control_data = X_ps.loc[control_idx, 'ps'].values.reshape(-1, 1)
    distances = scipy.spatial.distance.cdist(treated_data, control_data, metric='mahalanobis')
    return distances, X_ps


def matching_propensity(X, unbalanced_covariates, treatment_col, N=1, allow_repetition=False, use_caliper=False):
    treated_idx = X[X[treatment_col] == 1].index
    control_idx = X[X[treatment_col] == 0].index
    running_data = X.copy(deep=True)
    matched_pairs = []
    multiplier = 1
    distances, X_ps = calculate_distance_propensity(running_data, treated_idx, control_idx, unbalanced_covariates, treatment_col)
    flat_distances = distances.flatten()
    caliper = multiplier * mad(flat_distances)
    pairs = match_pairs(distances, treated_idx, control_idx, N=N, caliper=caliper)
    matched_pairs = pairs
    return matched_pairs, X_ps


def perform_balancing_method(df, continuous_variables, categorical_variables, treatment_col, use_caliper=True, smd_threshold=0.2, matching_ratio='1:3'):
    # Scaling continuous variables
    scaler = sklearn.preprocessing.MinMaxScaler()
    X = df.copy(deep=True)
    X[continuous_variables] = scaler.fit_transform(X[continuous_variables])
    # Matching Based on Propensity Scores
    all_vars = continuous_variables + categorical_variables
    balances = check_balance_after_matching(X, all_vars, treatment_col)
    if matching_ratio == '1:1':
        Ns = [1]
    elif matching_ratio == '1:2':
        Ns = [2]
    elif matching_ratio == '1:3':
        Ns = [3]
    elif matching_ratio == '1:4':
        Ns = [4]
    for N in Ns:
        threshs = [0]
        thresh_dict = dict.fromkeys(threshs)
        for thresh in threshs:
            thresh_dict[thresh] = {}
            balances_tuples = balances[balances > thresh]
            unbalanced_covariates = list(balances_tuples.sort_values(ascending=False).index)
            matched_pairs, X_ps = matching_propensity(X, unbalanced_covariates, treatment_col, N=N, use_caliper=use_caliper)
            matched_indices = [idx for pair in matched_pairs for idx in pair]
            X_matched = X_ps.loc[matched_indices].reset_index(drop=True)
            new_balances = check_balance_after_matching(X_matched, all_vars, treatment_col)
            new_balances_tuples = new_balances[new_balances > smd_threshold]
            new_unbalanced_covariates = list(new_balances_tuples.sort_values(ascending=False).index)
            thresh_dict[thresh]['data'] = X_matched
            thresh_dict[thresh]['unbalanced_covars'] = new_unbalanced_covariates
            if len(new_unbalanced_covariates) == 0:
                return X_matched, scaler
    min_key, items = min(thresh_dict.items(), key=lambda x: len(x[1]['unbalanced_covars']))
    return items['data'], scaler



def clean_outliers(data):
    cols = list(data)
    for col in cols:
        min_value = data[col].quantile(0.01)
        max_value = data[col].quantile(0.99)
        data[col][data[col] < min_value] = None
        data[col][data[col] > max_value] = None
    return data



def clean_data(list_of_covariates, list_of_treatment, list_of_outcome, list_of_duration, data):
    # check if data is a filename or a pandas dataframe
    if isinstance(data, str):
        data = pd.read_csv(data)
    else:
        data = data

    data = data[list_of_covariates + list_of_treatment + list_of_outcome + list_of_duration]

    # figure out which covariates are categorical and which are continuous
    categorical_covariates = []
    continuous_covariates = []
    for covariate in list_of_covariates:
        if data[covariate].dtype == 'object':
            # First check if there are any decimal points in the column
            if data[covariate].dropna().astype(str).str.contains('\.').any():
                data[covariate] = data[covariate].astype('float')
                continuous_covariates.append(covariate)
            else:
                categorical_covariates.append(covariate)
        else:
            continuous_covariates.append(covariate)

    data[continuous_covariates] = clean_outliers(data[continuous_covariates])

    # Impute missing values for continuous covariates
    for covariate in continuous_covariates:
        data[covariate] = data[covariate].fillna(data[covariate].mean())
    # Impute missing values for categorical covariates
    for covariate in categorical_covariates:
        data[covariate] = data[covariate].fillna(data[covariate].mode()[0])

    # One hot encode the categorical covariates and add columns to the categorical_covariates list
    new_categorical_covariates = []
    for covariate in categorical_covariates:
        data = pd.concat([data, pd.get_dummies(data[covariate], prefix=covariate)], axis=1)
        # Get the new column names
        new_columns = list(data.columns[-len(data[covariate].unique()):])
        list_of_covariates.remove(covariate)
        list_of_covariates.extend(new_columns)
        new_categorical_covariates.append(new_columns)
        data = data.drop(covariate, axis=1)
    # Update the categorical covariates list
    new_categorical_covariates = [item for sublist in new_categorical_covariates for item in sublist]
    categorical_covariates = new_categorical_covariates

    data[list_of_treatment[0]] = data[list_of_treatment[0]].astype('float')
    data[list_of_outcome[0]] = data[list_of_outcome[0]].astype('float')

    # if any boolean columns, convert to 0 and 1
    for column in data.columns:
        if data[column].dtype == 'bool':
            data[column] = data[column].astype('int')

    return data, categorical_covariates, continuous_covariates, list_of_treatment, list_of_outcome, list_of_duration



def calculate_hazard_ratios(X, list_of_treatment, list_of_outcome, list_of_duration, categorical_covariates, continuous_covariates):
    cph = lifelines.fitters.coxph_fitter.CoxPHFitter(penalizer=0.1)
    features = categorical_covariates + continuous_covariates + list_of_treatment + list_of_outcome + list_of_duration
    if "iptw_weight" in list(X):
        features += ["iptw_weight"]
    data_cox = X[features]
    for col in list(data_cox):
        if (data_cox[col] == data_cox[col].iloc[0]).all():
            data_cox.drop(columns=[col], inplace=True)

    columns_with_nan = X.columns[X.isna().all()].tolist()
    # Drop these columns from data_cox
    data_cox.drop(columns=columns_with_nan, inplace=True)

    if "iptw_weight" in list(data_cox):
        cph.fit(data_cox, duration_col=list_of_duration[0], event_col=list_of_outcome[0], robust=True, weights_col="iptw_weight")
    else:
        cph.fit(data_cox, duration_col=list_of_duration[0], event_col=list_of_outcome[0], robust=True)
    return cph, data_cox


def perform_psm(df, categorical_covariates, continuous_covariates, ignore_covariates, list_of_treatment, list_of_outcome, break_flag, results):
    """
    Performs Propensity Score Matching (PSM) on the DataFrame.
    Returns the matched DataFrame and information about the matching process.
    """
    ps_info = "Propensity Score Matching Summary:"
    matching_ratios = ['1:1', '1:2', '1:3', '1:4']
    number_of_unbalanced_covariates = len(categorical_covariates + continuous_covariates)
    all_vars = continuous_covariates + categorical_covariates
    index_matching_ratio = 0
    while (number_of_unbalanced_covariates > np.ceil(len(categorical_covariates + continuous_covariates) * 0.02)) and (index_matching_ratio < len(matching_ratios)):
        X, scaler = perform_balancing_method(df, continuous_covariates, categorical_covariates, list_of_treatment[0], smd_threshold=0.1, matching_ratio=matching_ratios[index_matching_ratio])
        balances = check_balance_after_matching(X, all_vars, list_of_treatment[0])
        unbalanced_covariates = list(balances[balances > 0.1].index)
        # Remove covariates in ignore_covariates from unbalanced_covariates
        unbalanced_covariates = [covariate for covariate in unbalanced_covariates if covariate not in ignore_covariates]
        balance_str = 'Number of unbalanced covariates: ' + str(len(unbalanced_covariates))
        covariates_str = 'Unbalanced covariates: ' + ', '.join(unbalanced_covariates)
        number_of_unbalanced_covariates = len(unbalanced_covariates)
        if number_of_unbalanced_covariates == 0:
            covariates_str = 'All covariates are balanced.'
        index_matching_ratio += 1
    results['Balances after performing balancing'] = balances

    currently_used_covariates = categorical_covariates + continuous_covariates
    ps_info = ps_info + "\n" + f"\nCurrently used covariates: {', '.join(currently_used_covariates)}"
    ps_info = ps_info + "\n" + balance_str + "\n" + covariates_str

    if number_of_unbalanced_covariates > np.ceil(len(categorical_covariates + continuous_covariates) * 0.02):
        break_flag = True

    if break_flag:
        return X, ps_info, results, break_flag, scaler

    incidence_treated = df[df[list_of_treatment[0]] == 1][list_of_outcome[0]].mean()
    incidence_control = df[df[list_of_treatment[0]] == 0][list_of_outcome[0]].mean()
    incedence_overall = df[list_of_outcome[0]].mean()
    ATE = incidence_treated - incidence_control
    treated_group_incidence_str = f"Incidence in the treated group: {incidence_treated:.4f}"
    control_group_incidence_str = f"Incidence in the control group: {incidence_control:.4f}"
    overall_incidence_str = f"Overall incidence: {incedence_overall:.4f}"
    ATE_str = f"Average Treatment Effect: {ATE:.4f}"
    ps_info = ps_info + "\n" + treated_group_incidence_str + "\n" + control_group_incidence_str + "\n" + overall_incidence_str + "\n" + ATE_str

    return X, ps_info, results, break_flag, scaler



def perform_iptw(df, categorical_covariates, continuous_covariates, ignore_covariates, list_of_treatment, list_of_outcome, break_flag, results):
    """
    Performs Inverse Probability of Treatment Weighting (IPTW) on the DataFrame.
    Returns the weighted DataFrame and information about the weighting process.
    """
    iptw_info = "Inverse Probability of Treatment Weighting Summary:"
    learner = LogisticRegression(solver="liblinear")
    ipw = IPW(learner)
    ipw.fit(df[categorical_covariates+continuous_covariates], df[list_of_treatment[0]])
    df['iptw_weight'] = ipw.compute_weights(df[categorical_covariates+continuous_covariates], df[list_of_treatment[0]])
    results_ipw = evaluate(ipw, df[categorical_covariates+continuous_covariates], df[list_of_treatment[0]], df[list_of_outcome[0]])
    balances = results_ipw.evaluated_metrics.covariate_balance['weighted']
    results["All IPTW results"] = results_ipw
    results["Balances after performing balancing"] = balances

    unbalanced_covariates = list(balances[balances > 0.1].index)
    # Remove covariates in ignore_covariates from unbalanced_covariates
    unbalanced_covariates = [covariate for covariate in unbalanced_covariates if covariate not in ignore_covariates]
    balance_str = 'Number of unbalanced covariates: ' + str(len(unbalanced_covariates))
    covariates_str = 'Unbalanced covariates: ' + ', '.join(unbalanced_covariates)
    number_of_unbalanced_covariates = len(unbalanced_covariates)
    if number_of_unbalanced_covariates == 0:
        covariates_str = 'All covariates are balanced.'

    currently_used_covariates = categorical_covariates + continuous_covariates
    iptw_info = iptw_info + "\n" + f"\nCurrently used covariates: {', '.join(currently_used_covariates)}"
    iptw_info = iptw_info + "\n" + balance_str + "\n" + covariates_str

    if number_of_unbalanced_covariates > np.ceil(len(categorical_covariates + continuous_covariates) * 0.02):
        break_flag = True

    if break_flag:
        return df, iptw_info, results, break_flag

    incidence_control = ipw.estimate_population_outcome(df[categorical_covariates+continuous_covariates], df[list_of_treatment[0]], df[list_of_outcome[0]]).iloc[0]
    incidence_treated = ipw.estimate_population_outcome(df[categorical_covariates+continuous_covariates], df[list_of_treatment[0]], df[list_of_outcome[0]]).iloc[1]
    ATE = ipw.estimate_effect(incidence_treated, incidence_control).loc['diff']
    treated_group_incidence_str = f"Incidence in the treated group: {incidence_treated:.4f}"
    control_group_incidence_str = f"Incidence in the control group: {incidence_control:.4f}"
    ATE_str = f"Average Treatment Effect: {ATE:.4f}"
    iptw_info = iptw_info + "\n" + treated_group_incidence_str + "\n" + control_group_incidence_str + "\n" + ATE_str
    return df, iptw_info, results, break_flag


def run_cox_model(X, list_of_treatment, list_of_outcome, list_of_duration, categorical_covariates, continuous_covariates):
    """
    Runs Cox Proportional Hazards model.
    Returns the model summary and the proportional hazards assumption plot.
    """
    cph_info = "Cox Proportional Hazards Model Summary:"

    cph, data_cox = calculate_hazard_ratios(X, list_of_treatment, list_of_outcome, list_of_duration, categorical_covariates, continuous_covariates)
    cox_summary = cph.summary.to_dict()

    # Plot Schoenfeld residuals
    fig = cph.check_assumptions(data_cox, p_value_threshold=0.05, show_plots=False)

    hr = cph.summary.loc[list_of_treatment[0]]['exp(coef)']
    upper_ci = cph.summary.loc[list_of_treatment[0]]['exp(coef) upper 95%']
    lower_ci = cph.summary.loc[list_of_treatment[0]]['exp(coef) lower 95%']
    p_value = cph.summary.loc[list_of_treatment[0]]['p']
    HR_str = f"Hazard Ratio: {hr:.4f}"
    CI_str = f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})"
    p_value_str = f"p-value: {p_value:.4f}"
    cph_info = cph_info + f"\n{HR_str}\n{CI_str}\n{p_value_str}\n"
    return cox_summary, fig, cph_info


def run_kaplan_meier(df, list_of_treatment, list_of_outcome, list_of_duration):
    """
    Runs Kaplan-Meier Estimator.
    Returns the survival function estimates and the plot.
    """
    treatment_col = list_of_treatment[0]
    outcome_col = list_of_outcome[0]
    duration_col = list_of_duration[0]
    kmf = KaplanMeierFitter()
    groups = df[treatment_col].unique()
    survival_functions = {}
    fig, ax = plt.subplots()
    for group in groups:
        mask = df[treatment_col] == group
        if "iptw_weight" in list(df):
            kmf.fit(df.loc[mask, duration_col], event_observed=df.loc[mask, outcome_col], weights=df.loc[mask, 'iptw_weight'], label=str(group))
        else:
            kmf.fit(df.loc[mask, duration_col], event_observed=df.loc[mask, outcome_col], label=str(group))
        kmf.plot_survival_function(ax=ax)
        survival_functions[str(group)] = kmf.survival_function_
    plt.title('Kaplan-Meier Survival Curves')
    return survival_functions, fig


def run_parametric_model(balanced_df, list_of_treatment, list_of_outcome, list_of_duration):
    # Instantiate each fitter
    wb = lifelines.WeibullFitter()
    log = lifelines.LogNormalFitter()
    loglogis = lifelines.LogLogisticFitter()
    # Fit to data
    parametric_info = "Parametric Survival Models Summary:"
    aics = {}
    for i, model in enumerate([wb, log, loglogis]):
        model.fit(durations = balanced_df[list_of_duration[0]], event_observed = balanced_df[list_of_outcome[0]])
        if i == 0: aics['Weibull Acceleration Factor/ Time Ratio'] = model.AIC_
        elif i == 1: aics['LogNormal Acceleration Factor/ Time Ratio'] = model.AIC_
        else: aics['LogLogistic Acceleration Factor/ Time Ratio'] = model.AIC_
    # Get the best model
    best_model = min(aics, key=aics.get)
    best_model = 'LogNormal Acceleration Factor/ Time Ratio'
    parametric_info = parametric_info + "\n" + f"Using Model: {best_model}"
    if best_model == 'Weibull Acceleration Factor/ Time Ratio':
        model = lifelines.fitters.weibull_aft_fitter.WeibullAFTFitter()
        if 'iptw_weight' in list(balanced_df):
            model.fit(balanced_df, duration_col=list_of_duration[0], event_col=list_of_outcome[0], weights_col='iptw_weight')
        else:
            model.fit(balanced_df, duration_col=list_of_duration[0], event_col=list_of_outcome[0])
        effect = model.summary.loc['lambda_'].loc[list_of_treatment[0]]['exp(coef)']
        upper_ci = model.summary.loc['lambda_'].loc[list_of_treatment[0]]['exp(coef) upper 95%']
        lower_ci = model.summary.loc['lambda_'].loc[list_of_treatment[0]]['exp(coef) lower 95%']
        p_value = model.summary.loc['lambda_'].loc[list_of_treatment[0]]['p']
        HR_str = f"Effect from {best_model}: {effect:.4f}"
        CI_str = f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})"
        p_value_str = f"p-value: {p_value:.4f}"
        parametric_info = parametric_info + f"\n{HR_str}\n{CI_str}\n{p_value_str}\n"
    if best_model == 'LogNormal Acceleration Factor/ Time Ratio':
        model = lifelines.fitters.log_normal_aft_fitter.LogNormalAFTFitter()
        if 'iptw_weight' in list(balanced_df):
            model.fit(balanced_df, duration_col=list_of_duration[0], event_col=list_of_outcome[0], weights_col='iptw_weight')
        else:
            model.fit(balanced_df, duration_col=list_of_duration[0], event_col=list_of_outcome[0])
        effect = model.summary.loc['mu_'].loc[list_of_treatment[0]]['exp(coef)']
        upper_ci = model.summary.loc['mu_'].loc[list_of_treatment[0]]['exp(coef) upper 95%']
        lower_ci = model.summary.loc['mu_'].loc[list_of_treatment[0]]['exp(coef) lower 95%']
        p_value = model.summary.loc['mu_'].loc[list_of_treatment[0]]['p']
        HR_str = f"Effect from {best_model}: {effect:.4f}"
        CI_str = f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})"
        p_value_str = f"p-value: {p_value:.4f}"
        parametric_info = parametric_info + f"\n{HR_str}\n{CI_str}\n{p_value_str}\n"
    if best_model == 'LogLogistic Acceleration Factor/ Time Ratio':
        model = lifelines.fitters.log_logistic_aft_fitter.LogLogisticAFTFitter()
        if 'iptw_weight' in list(balanced_df):
            model.fit(balanced_df, duration_col=list_of_duration[0], event_col=list_of_outcome[0], weights_col='iptw_weight')
        else:
            model.fit(balanced_df, duration_col=list_of_duration[0], event_col=list_of_outcome[0])
        effect = model.summary.loc['alpha_'].loc[list_of_treatment[0]]['exp(coef)']
        upper_ci = model.summary.loc['alpha_'].loc[list_of_treatment[0]]['exp(coef) upper 95%']
        lower_ci = model.summary.loc['alpha_'].loc[list_of_treatment[0]]['exp(coef) lower 95%']
        p_value = model.summary.loc['alpha_'].loc[list_of_treatment[0]]['p']
        HR_str = f"Effect from {best_model}: {effect:.4f}"
        CI_str = f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})"
        p_value_str = f"p-value: {p_value:.4f}"
        parametric_info = parametric_info + f"\n{HR_str}\n{CI_str}\n{p_value_str}\n"
    return model.summary, parametric_info


def run_survival_forest(balanced_df, list_of_treatment, list_of_outcome, list_of_duration, categorical_covariates, continuous_covariates):
    
    # Function to compute bootstrap confidence intervals
    def bootstrap_confidence_intervals(survival_curves, plt_times, n_bootstrap=1000, ci_level=0.95):
        """
        Computes confidence intervals for survival curves using bootstrap resampling.
        """
        bootstrap_survivals = []
        n_samples = len(survival_curves)

        # Bootstrap resampling
        for _ in range(n_bootstrap):
            # Sample with replacement
            sampled_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sampled_curves = [survival_curves[i](plt_times) for i in sampled_indices]
            # Compute mean survival curve for this bootstrap sample
            bootstrap_survivals.append(np.mean(sampled_curves, axis=0))
        
        # Convert to numpy array for easier percentile computation
        bootstrap_survivals = np.array(bootstrap_survivals)

        # Compute percentiles for confidence intervals
        lower_percentile = (1 - ci_level) / 2 * 100
        upper_percentile = (1 + ci_level) / 2 * 100
        lower_ci = np.percentile(bootstrap_survivals, lower_percentile, axis=0)
        upper_ci = np.percentile(bootstrap_survivals, upper_percentile, axis=0)
        
        return lower_ci, upper_ci

    def bootstrap_hazard_ratio(rsf, X, y, treatment_col, time_point, n_bootstrap=100, ci_level=0.95):
        """
        Computes hazard ratio and confidence intervals using bootstrap resampling.
        """
        # Compute observed hazard ratio
        group_0 = X[X[treatment_col] == 0]
        group_1 = X[X[treatment_col] == 1]

        chf_group_0 = rsf.predict_cumulative_hazard_function(group_0)
        chf_group_1 = rsf.predict_cumulative_hazard_function(group_1)

        # Ensure time_point is within valid range
        max_time_0 = min([fn.x[-1] for fn in chf_group_0])
        max_time_1 = min([fn.x[-1] for fn in chf_group_1])
        max_valid_time = min(max_time_0, max_time_1)
        
        time_point = min(time_point, max_valid_time)

        chf_group_0_values = [fn(time_point) for fn in chf_group_0]
        chf_group_1_values = [fn(time_point) for fn in chf_group_1]

        hr_observed = np.mean(chf_group_1_values) / np.mean(chf_group_0_values)

        # Bootstrap resampling
        bootstrap_hrs = []
        n_samples = len(X)

        for _ in range(n_bootstrap):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X.iloc[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            rsf.fit(X_bootstrap, y_bootstrap)

            group_0_bootstrap = X_bootstrap[X_bootstrap[treatment_col] == 0]
            group_1_bootstrap = X_bootstrap[X_bootstrap[treatment_col] == 1]

            chf_group_0_bootstrap = rsf.predict_cumulative_hazard_function(group_0_bootstrap)
            chf_group_1_bootstrap = rsf.predict_cumulative_hazard_function(group_1_bootstrap)

            # Ensure time_point is within valid range for bootstrap samples
            max_time_0_bootstrap = min([fn.x[-1] for fn in chf_group_0_bootstrap])
            max_time_1_bootstrap = min([fn.x[-1] for fn in chf_group_1_bootstrap])
            max_valid_time_bootstrap = min(max_time_0_bootstrap, max_time_1_bootstrap)

            time_point_bootstrap = min(time_point, max_valid_time_bootstrap)

            chf_group_0_bootstrap_values = [fn(time_point_bootstrap) for fn in chf_group_0_bootstrap]
            chf_group_1_bootstrap_values = [fn(time_point_bootstrap) for fn in chf_group_1_bootstrap]

            hr_bootstrap = np.mean(chf_group_1_bootstrap_values) / np.mean(chf_group_0_bootstrap_values)
            bootstrap_hrs.append(hr_bootstrap)

        # Compute confidence intervals
        lower_percentile = (1 - ci_level) / 2 * 100
        upper_percentile = (1 + ci_level) / 2 * 100
        ci_lower = np.percentile(bootstrap_hrs, lower_percentile)
        ci_upper = np.percentile(bootstrap_hrs, upper_percentile)

        return hr_observed, (ci_lower, ci_upper)


    info_str = "Random Survival Forest Results:"
    try:
        Xt = balanced_df[categorical_covariates + continuous_covariates + list_of_treatment]
        y_df = balanced_df[list_of_outcome + list_of_duration]
        y_df[list_of_outcome[0]] = y_df[list_of_outcome[0]].astype(bool)
        y = [tuple(x) for x in y_df[list_of_outcome + list_of_duration].to_numpy()]
        y= np.array(y, dtype=[("censor", bool), ("time", float)])

        random_state = 20
        X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25, random_state=random_state)
        rsf = RandomSurvivalForest(
            n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=random_state
        )
        rsf.fit(X_train, y_train)
        cci = rsf.score(X_test, y_test)
        info_str = info_str + "\n" + f"C-Index: {cci:.4f}"

        # Define time points for evaluation
        plt_times = np.linspace(0, max(y_df[list_of_duration[0]]), num=100)

        # Plot survival curves with confidence intervals
        for group in [0, 1]:  # Assuming binary treatment
            group_indices = X_test[list_of_treatment[0]] == group
            survival_curves = rsf.predict_survival_function(X_test[group_indices])

            # Average survival curve for the group
            avg_survival = np.mean([fn(plt_times) for fn in survival_curves], axis=0)

            # Compute confidence intervals using bootstrap
            lower_ci, upper_ci = bootstrap_confidence_intervals(survival_curves, plt_times)

            # Plot average survival curve
            plt.step(plt_times, avg_survival, label=f"Treatment Group {group}")

            # Plot confidence intervals
            plt.fill_between(
                plt_times, lower_ci, upper_ci, alpha=0.2, label=f"CI Treatment Group {group}"
            )

        # Add plot labels and legend
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.legend()
        plt.title("Survival Curves with Confidence Intervals by Treatment Group")

        # Don't show figure just save it to a variable
        survival_prob_fig = plt.gcf()

        max_val_y_train = max(y_train['time'])
        time_point = min(plt_times[-1], max(y_df[list_of_duration[0]]), max_val_y_train)
        hr, ci = bootstrap_hazard_ratio(rsf, X_test, y_test, treatment_col=list_of_treatment[0], time_point=time_point)

        info_str = info_str + "\n" + f"Random Survival Forest Hazard Ratio at {time_point} days: {hr:.3f}"
        info_str = info_str + "\n" + f"95% Confidence Interval: ({ci[0]:.3f}, {ci[1]:.3f})"
    except Exception as e:
        info_str = info_str + "\n" + "Random Survival Forest could not be run."
        survival_prob_fig = None

    return survival_prob_fig, info_str


def run_doubly_robust(df, list_of_treatment, list_of_outcome, categorical_covariates, continuous_covariates):
    info_str = "Doubly Robust Estimation Summary:\n"
    ipw = IPW(LogisticRegression(solver="liblinear"))
    std = Standardization(LinearRegression())
    dr = PropensityFeatureStandardization(std, ipw)
    dr.fit(df[continuous_covariates + categorical_covariates], df[list_of_treatment[0]], df[list_of_outcome[0]])
    results = evaluate(dr.weight_model, df[continuous_covariates + categorical_covariates], df[list_of_treatment[0]], df[list_of_outcome[0]])

    # Bootstrap for confidence intervals
    def bootstrap_ate(X, a, y, estimator, n_bootstrap=1000, random_state=None):
        rng = np.random.default_rng(random_state)
        estimates = []
        for _ in range(n_bootstrap):
            indices = rng.choice(len(X), size=len(X), replace=True)
            X_b, a_b, y_b = X.iloc[indices], a.iloc[indices], y.iloc[indices]
            estimator.fit(X_b, a_b, y_b)
            outcome = estimator.estimate_population_outcome(X_b, a_b)
            estimates.append(outcome[1] - outcome[0])
        return np.percentile(estimates, [2.5, 97.5]), np.mean(estimates)

    # Get CI and ATE
    ci, ate_mean = bootstrap_ate(df[continuous_covariates + categorical_covariates], df[list_of_treatment[0]], df[list_of_outcome[0]], dr)
    info_str += f"Average Treatment Effect: {ate_mean:.3f}\n95% Confidence Interval: {ci}\n"
    return results, info_str

# COMMAND ----------

# performing psm
from psmpy import PsmPy

def perform_psm_auto(df_raw, psm_inc_vars_list, psm_exc_vars_list, matching_ratio='1:1'):
    if matching_ratio =='1:1':
        how_many = 1

    df_psm = df_raw.copy()
    # add a column as unique id
    df_psm['deal_id'] = range(len(df_raw))
    # exclude variables that not for matching
    exclude_var = psm_exc_vars_list
    exclude_var_df = df_psm[['deal_id']+exclude_var]
    # exclude_var_df = exclude_var_df.set_index('deal_id')

    # perform psm
    psm = PsmPy(df_psm, indx='deal_id', treatment='treatment', exclude=exclude_var)
    psm.logistic_ps(balance=False)
    # print(psm.predicted_data)
    # psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None) # caliper = None
    psm.knn_matched_12n(matcher='propensity_logit', how_many=how_many) # how_many = 1
    df_matched = psm.df_matched

    # contact a completed df
    df_matched = pd.merge(df_matched,exclude_var_df,on='deal_id')

    return df_matched.copy()

# COMMAND ----------

# seed = 0
# df_sel_drug_list = pd.read_csv("./exp_data/{}/{}/sel_drug_list.csv".format(cancer_type, therapy_version))

seed = 0
#df_sel_drug_list = pd.read_csv("./exp_data/{}/{}/sel_drug_list.csv".format(cancer_type, therapy_version))

# checking the common drugs 
df_sel_drug_list_old = pd.read_csv("/Workspace/Users/zhenxing.xu@regeneron.com/ansclc_0815/exp_data/ansclc/combined_therapy_pres_adm_14_90/sel_drug_list.csv")

df_sel_drug_list_new = pd.read_csv("./exp_data/{}/{}/sel_drug_list.csv".format(cancer_type, therapy_version))

df_sel_drug_list = df_sel_drug_list_new.loc[df_sel_drug_list_new['generic_desc'].isin(list(df_sel_drug_list_old['generic_desc']))]
df_sel_drug_list = df_sel_drug_list.reset_index(drop=True)

# df_sel_drug_list_merged = pd.merge(df_sel_drug_list_old, df_sel_drug_list, on='generic_desc')
# df_sel_drug_list_merged

df_sel_drug_list

# COMMAND ----------

for idx in range(len(df_sel_drug_list)):
    drug_name = df_sel_drug_list.iloc[idx]['generic_desc']
    print(idx, drug_name)

# COMMAND ----------

# for idx in range(2):
for idx in range(len(df_sel_drug_list)):
    drug_name = df_sel_drug_list.iloc[idx]['generic_desc']
    if drug_name in ["PEMETREXED DISODIUM", "0.9 % SODIUM CHLORIDE", "PACLITAXEL PROTEIN-BOUND"]:
        continue

    drug_key = drug_name.replace('/', '-')
    all_data = []
    for control_idx in range(100):
        data = pd.read_csv('./exp_data/{}/{}/{}/all_data_v0_control_{}_sae.csv'.format(cancer_type, therapy_version, drug_key, control_idx))
        print("The {} run for drug {}".format(control_idx, drug_name))
        t_data = data['sae_delta_days'].values
        y_data = data['sae_flag'].values
        z_data = data['sae_flag'].values
        a_data = data['treatment'].values
        X = data.drop(columns = ['ptid', 'index_date', 'sae_delta_days', 'sae_flag', 'treatment'])

        X_data = X.values.astype(float)

        for col in X.columns:
            if np.isnan(X[col].values).mean() == 1:
                X = X.drop(col, axis = 1)
                print("{}: all missing and dropped".format(col))

        train_X, test_X, train_y, test_y, train_a, test_a, train_t, test_t, train_z, test_z = train_test_split(X_data, y_data, a_data, t_data, z_data, test_size=0.3, shuffle=True, random_state=seed)

        imp = SimpleImputer(strategy="median")
        imp.fit(train_X)
        train_X = imp.transform(train_X)
        test_X = imp.transform(test_X)

        # std = StandardScaler()
        # std.fit(train_X)
        # train_X = std.transform(train_X)
        # test_X = std.transform(test_X)
        # print(train_X.shape, test_X.shape)

        all_X = np.concatenate((train_X, test_X))
        all_y = np.concatenate((train_y, test_y))
        all_a = np.concatenate((train_a, test_a))
        all_t = np.concatenate((train_t, test_t))

        all_input = np.concatenate((all_X, all_a[:, None], all_y[:, None], all_t[:, None]), axis = 1)
        print(all_X.shape, all_input.shape, len(X.columns))
        df_input = pd.DataFrame(data = all_input, columns = list(X.columns) + ["treatment", "flag", "time"])

        ori_balances = check_balance_after_matching(df_input, list(X.columns), "treatment")
        ori_unbalanced_covariates = list(ori_balances[ori_balances > 0.2].index)
        print("Orignal unbalanced covariates:",len(ori_unbalanced_covariates))

        cph_ori = CoxPHFitter()
        cox_data_ori = pd.DataFrame({'T': df_input['time'], 'event': df_input['flag'], 'treatment': df_input["treatment"]})
        cph_ori.fit(cox_data_ori, 'T', 'event')
        HR_ori = cph_ori.hazard_ratios_['treatment']
        CI_ori = np.exp(cph_ori.confidence_intervals_.values.reshape(-1))
        print("Orignal HR", HR_ori, CI_ori)

        ### old psm code
        # new_all_input, scaler = perform_balancing_method(df_input, list(X.columns), [], "treatment")
        # print(new_all_input.shape)
        # balances = check_balance_after_matching(new_all_input, list(X.columns), "treatment")
        # unbalanced_covariates = list(balances[balances > 0.2].index)
        # print("Unbalanced covariates after PSM:",len(unbalanced_covariates))

        ### new psm code
        psm_inc_vars = list(X.columns)
        psm_exc_vars = ['flag', 'time']
        new_all_input = perform_psm_auto(df_input.copy(), psm_inc_vars, psm_exc_vars, matching_ratio='1:1')
        new_all_input = new_all_input.rename(columns={'propensity_score': 'ps'})
        new_all_input = new_all_input.drop(columns = ['propensity_logit'])
        # print('new_all_input.shape',new_all_input.shape)
        num_balanced_samples_psm = new_all_input.shape[0] # the number of samples after psm
        balances = check_balance_after_matching(new_all_input, list(X.columns), "treatment")
        unbalanced_covariates = list(balances[balances > 0.2].index)

        cph = CoxPHFitter()
        cox_data = pd.DataFrame({'T': new_all_input['time'], 'event': new_all_input['flag'], 'treatment': new_all_input["treatment"]})
        cph.fit(cox_data, 'T', 'event')
        HR = cph.hazard_ratios_['treatment']
        CI = np.exp(cph.confidence_intervals_.values.reshape(-1))
        print("HR after PSM", HR, CI)

        all_data.append((drug_name, HR_ori, CI_ori, HR, CI, len(ori_unbalanced_covariates), len(unbalanced_covariates)))
    all_data = np.array(all_data)
    pd_results = pd.DataFrame(data = all_data, columns = ["drug", "HR_ori", "CI_ori", "HR", "CI", "ori_unbalanced_covariates", "unbalanced_covariates"])
    pd_results.to_csv('./exp_data/{}/{}/{}/results_v0_sae.csv'.format(cancer_type, therapy_version, drug_key))
    

# COMMAND ----------

# for idx in range(2):
#     drug_name = df_sel_drug_list.iloc[idx]['generic_desc']
#     if drug_name in ["PEMETREXED DISODIUM", "0.9 % SODIUM CHLORIDE", "PACLITAXEL PROTEIN-BOUND", "VANCOMYCIN HCL"]:
#         continue
#     drug_key = drug_name.replace('/', '-')
#     results = pd.read_csv('./exp_data/{}/{}/{}/results_v0.csv'.format(cancer_type, therapy_version, drug_key))
#     print(drug_name)
#     print(results)

# COMMAND ----------

print('DONE.')