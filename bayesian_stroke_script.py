#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian stroke modeling details sex biases in the white matter substrates of aphasia (2023)

@author: Julius M. Kernbach

"""

# =============================================================================
# Import libraries
# =============================================================================

import numpy as np
import pandas as pd
import glob
import os
import arviz as az
import matplotlib.pyplot as plt
import theano
import seaborn as sns
import joblib
import arviz as az
import ptitprince as pt
from imblearn.over_sampling import SMOTE
       
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
import numpy as np, scipy.stats as st

# Careful with updates, latest errors:
# https://discourse.pymc.io/t/attributeerror-module-arviz-has-no-attribute-geweke/6818
# conda install -c conda-forge arviz=0.11.0
import pymc3 as pm

%matplotlib qt

# =============================================================================
# Set directory
# =============================================================================

MAIN_dir = '/MILA_WM'
os.chdir(MAIN_dir)

# Check for existing results folder
OUTPUT_FOLDER = "results/revision"

try:
    os.mkdir(OUTPUT_FOLDER)
except:
    print('\nDirectory << %s >> already exists' % OUTPUT_FOLDER)
    pass

# =============================================================================
#                        Load Data + Preprocess
# =============================================================================

# =============================================================================
# # Load korean dataset
# =============================================================================
sample = 'HallymBundang_n1401'

beh_data = pd.read_excel(
    'data/korea_data/file_datatransfer.xlsx')

# Data imputation
imp = KNNImputer(missing_values=999, n_neighbors=5)
beh_data_imp = pd.DataFrame(imp.fit_transform(beh_data))
beh_data_imp.columns = beh_data.columns
beh_data_imp = beh_data_imp.drop(columns=["ID", "Cohort"])

# =============================================================================
#                      Load lesion data
# =============================================================================
img_data = pd.read_excel(
    'dumps/imaging/catani_load_det_with_bihem.xlsx').iloc[:, :]  
img_data = img_data.drop(
    columns=['Arcuate_Left', 'Arcuate_Right', 'Corpus_Callosum'])
# Log lesion data
img_data_log = np.log(img_data + 1)

# =============================================================================
#                       NMF Analysis
# =============================================================================

#  NMF with log transformed Lesion Load
k_choice=10
nmf = NMF(n_components=k_choice, init='random', random_state=0)
lesion_nmf = nmf.fit_transform(img_data_log)

# Loading Matrix H, (1401,3)
H = pd.DataFrame(lesion_nmf, columns=[
    'Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5', 'Factor 6',
    'Factor 7', 'Factor 8', 'Factor 9', 'Factor 10'])

# Basic Matrix W, components (3,28)
W_ = nmf.components_
W_df = pd.DataFrame(nmf.components_, columns=img_data_log.columns)

# Reorder columns according to anatomy
columns_titles_correct_oder = [

    # Perisylvian Language Network
    # 'Arcuate_Left', 'Arcuate_Right',
    'arcuate_Anterior_Segment_Left', 'arcuate_Anterior_Segment_Right',
    'arcuate_Long_Segment_Left', 'arcuate_Long_Segment_Right',
    'arcuate_Posterior_Segment_Left', 'arcuate_Posterior_Segment_Right',

    # Inferior Network
    'Inferior_Longitudinal_Fasciculus_Left', 'Inferior_Longitudinal_Fasciculus_Right',
    'Inferior_Occipito_Frontal_Fasciculus_Left',                             'Inferior_Occipito_Frontal_Fasciculus_Right',
    'Uncinate_Left', 'Uncinate_Right',

    # Projection Network
    'Cortico_Spinal_Left', 'Cortico_Spinal_Right',
    'Internal_Capsule_L', 'Internal_Capsule_R',

    # Cerebellar Network
    'Cortico_Ponto_Cerebellum_Left', 'Cortico_Ponto_Cerebellum_Right',
    'Superior_Cerebelar_Pedunculus_Left', 'Superior_Cerebelar_Pedunculus_Right',
    'Inferior_Cerebellar_Pedunculus_Left', 'Inferior_Cerebellar_Pedunculus_Right',


    # Optic Radiations
    'Optic_Radiations_Left', 'Optic_Radiations_Right',

    # Miscellanous
    'Cingulum_Left', 'Cingulum_Right',
    'Fornix_L', 'Fornix_R',
    # 'Corpus_Callosum'
]

W = W_df.reindex(columns=columns_titles_correct_oder)

# =============================================================================
# Plot W
# =============================================================================
clean_cols = [

    # Perisylvian Language Network
    # 'Arcuate_Left', 'Arcuate_Right',
    'AF Anterior left', 'AF Anterior right',
    'AF Long left', 'AF Long right',
    'AF Posterior left', 'AF Posterior right',

    # Inferior Network
    'IFL left', 'IFL right',
    'IOFF left',                             'IOFF right',
    'UF left', 'UF right',

    # Projection Network
    'CST left', 'CST right',
    'IC left', 'IC right',

    # Cerebellar Network
    'CPC left', 'CPC right',
    'SCP left', 'SCP right',
    'ICP left', 'ICP right',
    
    # Optic Radiations
    'OR left', 'OR right',

    # Miscellanous
    'Cingulum left', 'Cingulum right',
    'Fornix left', 'Fornix right',
    # 'Corpus_Callosum'
]

fig, ax = plt.subplots(figsize=(15,10))
plt.matshow(W, cmap='viridis', fignum=0)


ax.tick_params(axis="x", bottom=True, top=False,
               labelbottom=True, labeltop=False,
               pad=15)

plt.xticks(np.arange(0, len(columns_titles_correct_oder)),
           clean_cols, fontsize=17, 
           rotation='vertical',
           )

plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],'')


plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4',
            'Factor 5', 'Factor 6', 'Factor 7', 'Factor 8', 'Factor 9', 'Factor 10'],
           fontsize=17)

plt.colorbar(orientation='vertical',shrink=0.4)
plt.clim(0,4)
plt.title('Basic matrix W, k=%.0f factors' % k_choice, fontsize=20, pad=40)

plt.tight_layout()

fig.savefig(OUTPUT_FOLDER + "/new_plotW_NMF_k=%.0f_%s.tiff" %
            (k_choice, sample), bbox_inces='tight', dpi=600)


# =============================================================================
# Plot Lateralization index
# =============================================================================
# Plot Lateralization index, higher=left, lower=right
prior_L = [1, 0]*14
prior_R = [0, 1]*14

LI = []
for ind in range(10):
    sum_L = np.sum(np.abs(W.iloc[ind])*prior_L)
    sum_R = np.sum(np.abs(W.iloc[ind])*prior_R)
    LI.append((sum_L - sum_R) / (sum_L + sum_R))

LI_df = pd.DataFrame(LI)

# Plot figure
fig_lat, ax = plt.subplots(figsize=(6, 13))
ax = sns.heatmap(LI_df, 
                 cmap='RdBu_r', 
                 # cmap='RdBu_r', 
                 square=True,
                 vmax=1, vmin=-1, center=0,
                 annot=False,
                 # linewidths=.1,
                 cbar_kws={"shrink": .8},

                 xticklabels=[],
                 yticklabels=['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4',
                              'Factor 5', 'Factor 6', 'Factor 7', 'Factor 8', 
                              'Factor 9', 'Factor 10'])

ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15)
ax.set_title('Lateralization Index', fontsize=25, pad=15)

fig_lat.tight_layout()

fig_lat.savefig(OUTPUT_FOLDER + '/new_lat_index_NMF%.0f_%s.svg' %
                (k_choice, sample),       bbox_inces='tight', dpi=600)

# =============================================================================
# =============================================================================
# #                     Baysian Model: NMF + Gender
# =============================================================================
# =============================================================================
# Female=1, Male=2 (Datasheet)
# Index with 1=male, 0=female
gender_index = np.array(beh_data_imp["Sex"]>=2, dtype = np.int32) 

# Covariates
vol_ss = StandardScaler().fit_transform(
    beh_data_imp.Total_infarct_volume_ml[:, None])[:, 0]

Age_mean_scaled = beh_data_imp["Age"] - beh_data_imp["Age"].mean()
Age_mean_scaled_2 = np.array(Age_mean_scaled * Age_mean_scaled)

Education_mean_scaled = beh_data_imp["Levelofeducation"] - \
    beh_data_imp["Levelofeducation"].mean()
Education_mean_scaled = np.array(Education_mean_scaled)

IQ_mean_scaled = beh_data_imp["IQCODE"] - beh_data_imp["IQCODE"].mean()
IQ_mean_scaled = np.array(IQ_mean_scaled)

Sex_m = np.array(beh_data_imp["Sex"] == 2, dtype=np.int32)
Sex_w = np.array(beh_data_imp["Sex"] == 1, dtype=np.int32)

# =============================================================================
# Feature input
# =============================================================================
measures = [
    'K_MMSE_total', 
    'Semanticfluencyanimal',
    'BostonNamingTest', 
    'Phonemicfluency_total',
    'DigitSymbolCoding_correct', 'TMT_A_Time', 'TMT_B_Time',
    'ReyComplexFigureTestCopy', 'SVLT_immediate_recall_total',
    'SVLT_delayed_recall', 'SVLT_recognition',
    'ReyComplexFigureTestdelayedrecall'
    ]

n_last_chains = 1000

for m in measures:

    input_name = m
    ix_ = beh_data_imp.columns.get_loc(input_name)

    cur_X = H  # 1401x10
    cur_y = StandardScaler().fit_transform(beh_data_imp[input_name][:, None])[:, 0]

    # Check for existing results folder
    HIER_folder = OUTPUT_FOLDER + '/' + input_name
    try:
        os.mkdir(HIER_folder)
    except:
        print('\nDirectory << %s >> already exists' % HIER_folder)
        pass
    
    with pm.Model() as hierarchical_model:
    
        mu = pm.Normal("a", mu=0, sd=1)
    
        for i_fact, fact in enumerate(range(10)):
            cur_beta = pm.Normal('NMF_factor_%.0f' %
                                 (fact+1), mu=0, sd=1, shape=2)
            mu = mu + cur_beta[gender_index] * cur_X.iloc[:, i_fact]   
    
        # Define covariates
        cov1_beta = pm.Normal("cov1_vol", mu=0, sd=1, shape=1)
        mu = mu + cov1_beta * vol_ss
    
        cov2_beta = pm.Normal("cov2_age", mu=0, sd=10, shape=1)
        mu = mu + cov2_beta * Age_mean_scaled
    
        cov3_beta = pm.Normal("cov3_age_2", mu=0, sd=10, shape=1)
        mu = mu + cov3_beta * Age_mean_scaled_2
    
        cov4_beta = pm.Normal("cov4_ed_years", mu=0, sd=5, shape=1)
        mu = mu + cov4_beta * Education_mean_scaled
    
        cov5_beta = pm.Normal("cov5_IQCode", mu=0, sd=1, shape=1)
        mu = mu + cov5_beta * IQ_mean_scaled
    
        cov6_beta = pm.Normal("cov4_sex_m", mu=0, sd=1, shape=1)
        mu = mu + cov6_beta * Sex_m

        cov7_beta = pm.Normal("cov5_sex_w", mu=0, sd=1, shape=1)
        mu = mu + cov7_beta * Sex_w

        # Model error
        eps = pm.HalfCauchy('eps', 20)
    
        # Data likelihood
        MMSE_like = pm.Normal('%s' % input_name, mu=mu,
                              sd=eps, observed=np.array(cur_y))
        
        # Run model
        with hierarchical_model:
            hierarchical_trace = pm.sample(
                draws=5000, 
                n_init=1000, 
                cores=1, 
                random_seed=[123], 
                chains=1,  
                progressbar=True)  
        
        
            DUMP_folder = HIER_folder + '/model_dump'
            try:
                os.mkdir(DUMP_folder)
            except:
                print('\nDirectory << %s >> already exists' % DUMP_folder)
                pass
            
            # # Save pm.summary as csv
            df_save = pd.DataFrame(pm.summary(hierarchical_trace))
            df_save.to_csv(DUMP_folder + '/%s_gender_pmsummary.csv' % input_name)
    
            # Save model
            joblib.dump(hierarchical_trace,
                        DUMP_folder + '/%s_gender_hierachical_trace_joblib' % input_name)
            pm.save_trace(hierarchical_trace,
                          DUMP_folder + '/%s_gender_hierachical_trace' % input_name)
    
    
    # Prep model output
    gender_components = []
    
    for i_tract, tract in enumerate(range(10)):
        gender_components.append('NMF_factor_%.0f' % (tract+1))
    
    model_lower_vars = gender_components
    model_lower_vars.append("a")
    model_lower_vars.append("cov1_vol")
    model_lower_vars.append("cov2_age")
    model_lower_vars.append("cov3_age_2")
    model_lower_vars.append("cov4_ed_years")
    model_lower_vars.append("cov5_IQCode")
    model_lower_vars.append("cov6_male")
    model_lower_vars.append("cov7_female")
    

    
    # # =======================================================================
    # # PM POSTERIOR PLOTS
    # # =======================================================================
    # Check for existing results folder
    PM_FOLDER = HIER_folder + '/pm_posteriors'
    try:
        os.mkdir(PM_FOLDER)
    except:
        print('\nDirectory << %s >> already exists' % PM_FOLDER)
        pass
    
    # Posterior distributions
    subgroup_labels = ['female', 'male']
    THRESH = 0.1
    
    for index, var_ in enumerate(model_lower_vars[:10]):
        fig, ax = plt.subplots(1,2, figsize=(15, 8), sharey=True)
        pm.plot_posterior(hierarchical_trace[-n_last_chains:], 
                                varnames=var_,
                                round_to=3, 
                                kind='hist',
                                ax=[ax],
                                credible_interval=0.8)
        
        # ax[0].set_xlim(-THRESH, THRESH)
        # ax[1].set_xlim(-THRESH, THRESH)
    
        ax[0].set_title('%s\nfemale' % var_, color='magenta', loc='center')
        ax[1].set_title('%s\nmale' % var_, color='blue', loc='center')
    
        plt.tight_layout()

        fig.savefig(PM_FOLDER + '/%s_pm_posteriors_%s.tiff' % (input_name,var_), dpi=300)
    
    plt.close('all')
    
    # One Plot for PM
    fig, ax = plt.subplots(2, 10, figsize=(40, 22))
    for index, var_ in enumerate(model_lower_vars[:10]):
        pm.plot_posterior(hierarchical_trace[-n_last_chains:], 
                                varnames=var_,
                                round_to=3, 
                                kind='hist',                                
                                ax=ax[:,index],
                                credible_interval=0.8)                                
        
    fig.suptitle('%s [female=0, male=1]' % input_name, )      
    fig.tight_layout()
    fig.savefig(PM_FOLDER + '/%s_ALL_pm_posteriors.tiff' % input_name, dpi=300)
    
    # # =======================================================================
    # # ARVIZ POSTERIOR PLOTS
    # # =======================================================================
    
    # Check for existing results folder
    AZ_FOLDER = HIER_folder + '/arviz_posteriors'
    try:
        os.mkdir(AZ_FOLDER)
    except:
        print('\nDirectory << %s >> already exists' % AZ_FOLDER)
        pass
    
    for index, var_ in enumerate(model_lower_vars[:10]):
    
    
        fig = az.plot_trace(hierarchical_trace[-n_last_chains:], compact=True, var_names=var_)
        
        fig[0][0].get_lines()[0].set_color('magenta')
        fig[0][1].get_lines()[0].set_color('magenta')
        fig[0][0].get_lines()[1].set_color('blue')
        fig[0][1].get_lines()[1].set_color('blue')   
    
        
        post_lines = fig[0][0].get_lines()
        custom_lines = [Line2D([0], [0], color=l.get_c(), lw=4) for l in post_lines]
        if subgroup_labels is None:
            subgroup_labels = ['subgroup %i' % i for i in range(len(custom_lines))]
        fig[0][0].legend(custom_lines, subgroup_labels, loc='upper left', prop={'size': 7.5})
        
        plt.savefig(AZ_FOLDER + '/%s_az_posteriors_%s.svg' % (input_name,var_), dpi=300)
        plt.savefig(AZ_FOLDER + '/%s_az_posteriors_%s.tiff' % (input_name,var_), dpi=300)
    
    plt.close('all')

    
    # # =============================================================================
    # # RAINCLOUD PLOTS
    # # =============================================================================
    
    # Check for existing results folder
    RP_FOLDER = HIER_folder + '/rainclouds'
    try:
        os.mkdir(RP_FOLDER)
    except:
        print('\nDirectory << %s >> already exists' % RP_FOLDER)
        pass
    
    # Prepare Data
    dict_ = {"Factor": [], "Gender": [], "Value": []}
    for index, var_ in enumerate(model_lower_vars[:10]):
        print("Create result dict: %s" % var_)
        for a in range(n_last_chains):
    
            dict_["Factor"].append(var_)
            dict_["Gender"].append('female')
            dict_["Value"].append(hierarchical_trace[-n_last_chains:][var_][a,0])
            
        for a in range(n_last_chains):
     
            dict_["Factor"].append(var_)
            dict_["Gender"].append('male')
            dict_["Value"].append(hierarchical_trace[-n_last_chains:][var_][a,1])
    
    
    df = pd.DataFrame(
        data= dict_
        )
    
    # Plot all Factors
    sns.set_style('white')
    fig, ax = plt.subplots(figsize=(15, 22))
    
    ax = pt.RainCloud(x = "Factor", y = "Value", 
                    hue = "Gender", 
                    data = df, 
                    palette = [(0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
                                    (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)], 
                    # bw = sigma,
                    width_viol = .7, 
                    ax = ax, 
                    orient = "h" , 
                    alpha = .8,
                    dodge = True)
    
    sns.despine(left=True)
    
    ax.set_title('%s' % input_name, pad=25, size=35)
    
    # Turn off axes labels
    ax.set_ylabel('')
    ax.set_xlabel('')
    
    plt.tight_layout()
    fig.savefig(RP_FOLDER + '/%s_rp_all-posteriors.tiff' % input_name, dpi=300)

    # Plot each factor individually
    for index, var_ in enumerate(model_lower_vars[:10]):
        print(var_)
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax = pt.RainCloud(x = "Factor", y = "Value", 
                        hue = "Gender", 
                        data = df[df['Factor'] == var_], 
                        palette = [(0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
                                    (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)], 
                        # bw = sigma,
                        width_viol = .7, 
                        ax = ax, 
                        orient = "h" , 
                        alpha = .8,
                        dodge = True)
        
        sns.despine(left=True)
        
        # Turn off axes labels
        ax.set_ylabel('')
        ax.set_xlabel('')
        
        ax.set_title('%s: %s' % (input_name, var_), pad=25, size=25)
        plt.tight_layout()
        
        fig.savefig(RP_FOLDER + '/%s_rp_posteriors_%s.tiff' % (input_name,var_), dpi=300)
        
    
    
    # =============================================================================
    # Beta plots
    # =============================================================================
    
    beta_modes_female = np.zeros((len(model_lower_vars[:10])))
    for i, var_ in enumerate(model_lower_vars[:10]):
    
        beta_modes_female[i] = hierarchical_trace[var_][-n_last_chains:,0].mean()
        
    beta_modes_male = np.zeros((len(model_lower_vars[:10])))
    for i, var_ in enumerate(model_lower_vars[:10]):
    
        beta_modes_male[i] = hierarchical_trace[var_][-n_last_chains:,1].mean()
        
    allbetas = np.vstack((beta_modes_female, beta_modes_male))
    
    # Plot
    fig, ax = plt.subplots(figsize=(20, 8))
        
    ax = plt.matshow(
        np.atleast_2d(allbetas),
        cmap='RdBu_r',
        vmin=-.25,
        vmax=+.25,
        fignum=0)
    
    plt.xticks(np.arange(0, len(model_lower_vars[:10])),
               model_lower_vars[:10], 
               fontsize=20, 
               rotation=90)
    
    plt.yticks([0, 1], subgroup_labels, fontsize=20)
    
    plt.colorbar(orientation='vertical')
    plt.clim(-.1, .1)
    
    plt.title('Betas for %s' % input_name, fontsize=25, pad=25)
    plt.tight_layout()

    plt.savefig(HIER_folder + '/%s_gender_betas.tiff' % input_name, dpi=600)
    
    

    # # =============================================================================
    # # Predicted vs True Lin Regression
    # # =============================================================================
    Y_ppc_insample = pm.sample_ppc(
        hierarchical_trace, 5000, 
        hierarchical_model, 
        random_seed=123)[input_name]
    
    y_pred_insample = Y_ppc_insample.mean(axis=0)
    ppc_insample = r2_score(cur_y, y_pred_insample)
    
    out_str = 'PPC in sample R^2: %0.2f' % (ppc_insample)
    print(out_str)
    
    plt.figure(figsize=(8, 8))
    sns.regplot(x=cur_y, y=y_pred_insample, fit_reg=True, ci=95, 
                # n_boot=1000, x_bins=10,
                line_kws={'color': 'black', 'linewidth': 2})
    
    plt.xlabel('real output variable')
    plt.ylabel('predicted output variable')
    
    plt.title('Posterior predictive check: pred vs true for %s (%s)' %
              (input_name, out_str), fontsize=10)
    plt.tight_layout()
    
    plt.savefig(HIER_folder + '/%s_gender_PPC.tiff' % input_name, dpi=600)
    
    plt.close('all')




# =============================================================================
# Plot: correlations 
# =============================================================================

# 1. Correlation with lesion atoms (Loading matrix H) and clinical scores
scores = beh_data_imp.drop(columns=[
    'Age', 'Sex', 'Levelofeducation', 'Prior_stroke', 'IQCODE',
    'interval_onset_NPO_days', 'Total_infarct_volume_ml'])

cols = ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4',
        'Factor 5', 'Factor 6', 'Factor 7', 'Factor 8',
        'Factor 9', 'Factor 10', 
        'K_MMSE_total', 'Semanticfluencyanimal', 'BostonNamingTest',
        'Phonemicfluency_total', 'DigitSymbolCoding_correct', 'TMT_A_Time',
        'TMT_B_Time', 'ReyComplexFigureTestCopy', 'SVLT_immediate_recall_total',
        'SVLT_delayed_recall', 'SVLT_recognition',
        'ReyComplexFigureTestdelayedrecall']

# Combine into a single df, n1401x10+12
cor_Data = pd.DataFrame(data=np.hstack((H, scores)), columns=cols)

# Pearson Correlation
cor_scores = cor_Data.corr(method='pearson')

cor_scores = cor_scores.drop(
    ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4',
     'Factor 5', 'Factor 6', 'Factor 7', 'Factor 8',
     'Factor 9', 'Factor 10'], 
    axis=0)

cor_scores = cor_scores.drop(
    [ 'K_MMSE_total', 'Semanticfluencyanimal', 'BostonNamingTest',
     'Phonemicfluency_total', 'DigitSymbolCoding_correct', 'TMT_A_Time',
     'TMT_B_Time', 'ReyComplexFigureTestCopy', 'SVLT_immediate_recall_total',
     'SVLT_delayed_recall', 'SVLT_recognition',
     'ReyComplexFigureTestdelayedrecall'], 
    axis=1)

# Only language scores
cor_scores = cor_scores.drop(
    ['K_MMSE_total','DigitSymbolCoding_correct', 'TMT_A_Time',
     'TMT_B_Time', 'ReyComplexFigureTestCopy', 'SVLT_immediate_recall_total',
     'SVLT_delayed_recall', 'SVLT_recognition',
     'ReyComplexFigureTestdelayedrecall'], axis=0)


# Rank factors
abs_, indices = [], []
for i in range(cor_scores.shape[1]):
    abs_.append(np.abs(cor_scores.iloc[:, i]).sum())
    indices.append(i)
df_order = pd.DataFrame([abs_, indices]).T
df_order_Re = df_order.reindex(df_order[0].sort_values(ascending=False).index)
re_list = []
for i in np.array(df_order_Re[1]):
    re_list.append(cor_scores.columns[i])

re_ = cor_scores.reindex(re_list, axis='columns')

# Quick Bar plot
fig, ax = plt.subplots(figsize=(15, 5))
plt.bar(
    x=np.arange(len(df_order_Re[0])),
    height=np.array(df_order_Re[0]),
    color='Grey')
plt.xticks(np.arange(len(df_order_Re[0])), re_list, fontsize=15, rotation=90)
plt.title('Ranked Factor-to-score assignments by mean', pad=20, fontsize=25)
plt.ylabel('Mean factor-to-scores assignment', labelpad=20, fontsize=15)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + '/pearson_correlations_H_RANKS.svg', dpi=300)
plt.savefig(OUTPUT_FOLDER + '/pearson_correlations_H_RANKS.tiff', dpi=300)

# Reindex scores for better visualization
score_order = ['K_MMSE_total', 'BostonNamingTest', 'Semanticfluencyanimal',
               'Phonemicfluency_total', 'SVLT_immediate_recall_total',
               'SVLT_delayed_recall', 'SVLT_recognition',
               'ReyComplexFigureTestdelayedrecall', 'ReyComplexFigureTestCopy',
               'DigitSymbolCoding_correct', 
               'TMT_A_Time',
               'TMT_B_Time'
               ]
re_ = re_.reindex(score_order, axis='rows')

# PLOT
sns.set(style="white")

fig, ax = plt.subplots(figsize=(15, 15))

ax = sns.heatmap(re_,
                 cmap='RdBu_r',
                 vmax=.1, vmin=-.1,
                 center=0, 
                 square=True,
                 linewidths=.0001, 
                 cbar_kws={"shrink": .9})

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=18)


clean_y_labels =['K-MMSE', 'BNT', 'Semantic fluency',
               'Phonemic fluency', 'SVLT immediate recall',
               'SVLT delayed recall', 'SVLT recognition',
               'RCFT delayed recall', 'RCFT',
               'Digit Symbol Coding', 
               'TMT-A ',
               'TMT-B']

ax.set_yticklabels(clean_y_labels, fontsize=18)   

plt.tight_layout()


fig.savefig(OUTPUT_FOLDER + '/pearson_correlations_H.svg', dpi=300)
fig.savefig(OUTPUT_FOLDER + '/pearson_correlations_H.tiff', dpi=300)
    
# =============================================================================
# NMF-Ranks for each individual score
# =============================================================================
cor_Data['BostonNamingTest']

cor_scores.iloc[2,:].sort_values()
# Rank factors
abs_, indices = [], []
for i in range(cor_scores.shape[1]):
    abs_.append(np.abs(cor_scores.iloc[:, i]).sum())
    indices.append(i)
df_order = pd.DataFrame([abs_, indices]).T
df_order_Re = df_order.reindex(df_order[0].sort_values(ascending=False).index)
re_list = []
for i in np.array(df_order_Re[1]):
    re_list.append(cor_scores.columns[i])

re_ = cor_scores.reindex(re_list, axis='columns')