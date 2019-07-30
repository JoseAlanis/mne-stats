"""
===============================================
Plot group-level effect of continuous covariate
===============================================
"""

# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from mne.datasets import limo
from mne.decoding import Vectorizer, get_coef
from mne.evoked import EvokedArray
from mne import combine_evoked

from mne.viz import plot_compare_evokeds

###############################################################################
# list with subjects ids that should be imported
subjects = list(range(2, 19))
# create a dictionary containing participants data for easy slicing
limo_epochs = {str(subj): limo.load_data(subject=subj) for subj in subjects}

# interpolate missing channels
for subject in limo_epochs.values():
    subject.interpolate_bads(reset_bads=True)
    # only keep eeg channels
    subject.pick_types(eeg=True)

###############################################################################
# regression parameters

# variables to be used in the analysis (i.e., predictors)
predictors = ['intercept', 'face a - face b', 'phase-coherence']

# number of predictors
n_predictors = len(predictors)

# save epochs information (needed for creating a homologous
# epochs object containing linear regression result)
epochs_info = limo_epochs[str(subjects[0])].info

# number of channels and number of time points in each epoch
# we'll use this information later to bring the results of the
# the linear regression algorithm into an eeg-like format
# (i.e., channels x times points)
n_channels = len(epochs_info['ch_names'])
n_times = len(limo_epochs[str(subjects[0])].times)

# also save times first time-point in data
times = limo_epochs[str(subjects[0])].times
tmin = limo_epochs[str(subjects[0])].tmin

###############################################################################
# create empty objects  for the storage of results

# place holders for bootstrap samples
betas = np.zeros((len(limo_epochs.values()),
                  n_channels * n_times))
r_squared = np.zeros((len(limo_epochs.values()),
                      n_channels * n_times))

###############################################################################
# run regression analysis for each subject

# loop through subjects, set up and fit linear model
for iteration, subject in enumerate(limo_epochs.values()):

    # --- 1) create design matrix ---
    # use epochs metadata
    design = subject.metadata.copy()

    # add intercept (constant) to design matrix
    design = design.assign(intercept=1)

    # effect code contrast for categorical variable (i.e., condition a vs. b)
    design['face a - face b'] = np.where(design['face'] == 'A', 1, -1)

    # order columns of design matrix
    design = design[predictors]

    # column of betas array (i.e., predictor) to run bootstrap on
    pred_col = predictors.index('phase-coherence')

    # --- 2) vectorize (eeg-channel) data for linear regression analysis ---
    # data to be analysed
    data = subject.get_data()

    # vectorize data across channels
    Y = Vectorizer().fit_transform(data)

    # --- 3) fit linear model with sklearn's LinearRegression ---
    # we already have an intercept column in the design matrix,
    # thus we'll call LinearRegression with fit_intercept=False
    linear_model = LinearRegression(fit_intercept=False)
    linear_model.fit(design, Y)

    # --- 4) extract the resulting coefficients (i.e., betas) ---
    # extract betas
    coefs = get_coef(linear_model, 'coef_')
    # only keep relevant predictor
    betas[iteration, :] = coefs[:, pred_col]

    # calculate coefficient of determination (r-squared)
    r_squared[iteration, :] = r2_score(Y, linear_model.predict(design),
                                       multioutput='raw_values')

    # clean up
    del linear_model

###############################################################################
# create design matrix fro group-level regression

# get participants age
subj_age = read_csv('./limo_dataset_age.tsv', sep='\t', header=0)

# only keep subjects 2 - 18
group_design = subj_age.iloc[1:]
# add intercept
group_design = group_design.assign(intercept=1)
# order columns of design matrix
group_predictors = ['intercept', 'age']
group_design = group_design[group_predictors]

###############################################################################
# fit linear model with sklearn's LinearRegression
# we already have an intercept column in the design matrix,
# thus we'll call LinearRegression with fit_intercept=False

# set up and fit model
linear_model = LinearRegression(fit_intercept=False)
linear_model.fit(group_design, betas)

# extract group-level beta coefficients
group_coefs = get_coef(linear_model, 'coef_')

# only keep relevant predictor
group_pred_col = group_predictors.index('age')
group_betas = group_coefs[:, group_pred_col]

# back projection to channels x time points
group_betas = group_betas.reshape((n_channels, n_times))
# create evoked object containing the back projected coefficients
group_betas_evoked = EvokedArray(group_betas, epochs_info, tmin)

###############################################################################
# plot the modulating effect of age on the phase coherence predictor (i.e.,
# how the effect of phase coherence varies as a function of subject age)

# index of B8 in array
electrode = 'B8'
pick = group_betas_evoked.ch_names.index(electrode)

plot_compare_evokeds(group_betas_evoked,
                     picks=pick)

###############################################################################
# run analysis on optimized electrode (i.e., electrode showing best fit for
# phase-coherence predictor).

# reshape subjects' R-squared to channels * time-points space
r_squared = r_squared.reshape((r_squared.shape[0], n_channels, n_times))

# look for electrode with the highest R-squared variance in each subject
optimized_electrode = []
for i in range(r_squared.shape[0]):
    subj_r2 = r_squared[i, :, :]

    electrode_variance = []
    for e in range(subj_r2.shape[0]):
        electrode_variance.append(np.var(subj_r2[e, :]))

    optimized_electrode.append(np.argmax(electrode_variance))
    # or use max R-squared ?
    # optimized_electrode.append(np.unravel_index(r_squared[i, :, :].argmax(),
    #                                             r_squared[i, :, :].shape)[0])

###############################################################################
# extract beta coefficients for electrode showing best fit

# reshape subjects' beats to channels * time-points space
betas = betas.reshape((betas.shape[0], n_channels, n_times))

optimized_electrode_betas = np.zeros((betas.shape[0], n_times))
for i in range(betas.shape[0]):
    optimized_electrode_betas[i, :] = betas[i, optimized_electrode[i], :]

###############################################################################
# fit linear model with sklearn's LinearRegression
# we already have an intercept column in the design matrix,
# thus we'll call LinearRegression with fit_intercept=False
linear_model = LinearRegression(fit_intercept=False)
linear_model.fit(group_design, optimized_electrode_betas)

# extract group-level beta coefficients
group_coefs = get_coef(linear_model, 'coef_')

# only keep relevant predictor
group_pred_col = group_predictors.index('age')
group_betas = group_coefs[:, group_pred_col]

###############################################################################
# plot the modulating effect of age on the phase coherence predictor for
# optimized electrode

plt.plot(times, group_betas * 1e6)  # transform betas to microvolt
plt.ylim(ymax=1, ymin=-1)
