"""
========================
Plot group-level effects
========================

"""

# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from mne.stats.parametric import ttest_1samp_no_p
from mne.datasets import limo
from mne.decoding import Vectorizer, get_coef
from mne.evoked import EvokedArray
from mne import grand_average

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

# dicts for results evoked-objects
betas_evoked = dict()
t_evokeds = dict()

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

    # --- 3) Parameters for significance testing ---
    # number of trials
    n_trials = design.shape[0]
    # degrees of freedom
    dfs = float(n_trials - n_predictors)

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

    # the matrix of coefficients has a shape of number of observations in
    # the vertorized channel data by number of predictors;
    # thus, we can loop through the columns i.e., the predictors)
    # of the coefficient matrix and extract coefficients for each predictor
    # in order to project them back to a channels x time points space.
    lm_betas = dict()

    # extract coefficients
    beta = betas[iteration, :]
    # back projection to channels x time points
    beta = beta.reshape((n_channels, n_times))
    # create evoked object containing the back projected coefficients
    lm_betas['phase-coherence'] = EvokedArray(beta, epochs_info, tmin)

    # save results
    betas_evoked[str(subjects[iteration])] = lm_betas

    # # --- 5) compute t-values ---
    lm_t = dict()

    # compute model predictions
    predictions = linear_model.predict(design)

    # compute sum of squared residuals and mean squared error
    residuals = (Y - predictions)
    # sum of squared residuals
    ssr = np.sum(residuals ** 2, axis=0)
    # mean squared error
    sqrt_mse = np.sqrt(ssr / dfs)
    # raw error terms:
    # here, we take the inverse of the design matrix's projections
    # (i.e., A^T*A)^-1 and extract the square root of the diagonal values.
    error = np.sqrt(np.diag(np.linalg.pinv(np.dot(design.T, design))))
    # only keep relevant predictor
    error = error[pred_col]
    # compute standard error
    stderr = sqrt_mse * error

    # compute t values
    t_val = coefs[:, pred_col] / stderr
    # back projection to channels x time points
    t_val = t_val.reshape((n_channels, n_times))

    # create evoked object containing the back projected coefficients
    lm_t['phase-coherence'] = EvokedArray(t_val, epochs_info, tmin)

    # save results
    t_evokeds[str(subjects[iteration])] = lm_t

    # clean up
    del linear_model

###############################################################################
# compute mean beta-coefficient for predictor phase-coherence

# subject ids
subjects = [str(subj) for subj in subjects]

# extract phase-coherence betas for each subject
phase_coherence = [betas_evoked[subj]['phase-coherence'] for subj in subjects]
# average phase-coherence betas
ga_phase_coherence = grand_average(phase_coherence)

###############################################################################
# compute bootstrap confidence interval for phase-coherence betas and t-values

# column of betas array (i.e., predictor) to run bootstrap on
pred_col = predictors.index('phase-coherence')

# set random state for replication
random_state = 42
random = np.random.RandomState(random_state)

# number of random samples
boot = 2000

boot_betas = np.zeros((boot, n_channels * n_times))
# run bootstrap for regression coefficients
for i in range(boot):
    # extract random subjects from overall sample
    resampled_subjects = random.choice(range(betas.shape[0]),
                                       betas.shape[0],
                                       replace=True)
    # compute centrality estimator on re-sampled betas
    boot_betas[i, :] = betas[resampled_subjects, :].mean(axis=0)

# compute low and high percentile
lower, upper = np.quantile(boot_betas, [.025, .975], axis=0)

# reshape to channels * time-points space
lower = lower.reshape((n_channels, n_times))
upper = upper.reshape((n_channels, n_times))

###############################################################################
# plot mean beta parameter for phase coherence and 95%
# confidence interval for the electrode showing the strongest effect (i.e., C1)

# index of C1 in array
pick = ga_phase_coherence.ch_names.index('C1')

# create figure
fig, ax = plt.subplots(figsize=(7, 4))
ax = plot_compare_evokeds(ga_phase_coherence,
                          ylim=dict(eeg=[-1.5, 3.5]),
                          picks=pick,
                          show_sensors='upper right',
                          axes=ax)
ax.axes[0].fill_between(times,
                        # tranform values to microvolt
                        upper[pick] * 1e6,
                        lower[pick] * 1e6,
                        alpha=0.2)
plt.plot()

###############################################################################
# bootstrap one-sample t test

# set random state for replication
random_state = 42
random = np.random.RandomState(random_state)

# number of random samples
boot = 2000

boot_t = np.zeros((boot, n_channels * n_times))

# run bootstrap t-test
for ind in range(boot):
    # extract random subjects from overall sample
    resampled_subjects = random.choice(subjects,
                                       len(subjects),
                                       replace=True)

    # extract t-values for the ramdom sample of subjects
    t_sample = np.zeros((len(limo_epochs.values()), n_channels, n_times))
    for iteration, subj in enumerate(resampled_subjects):
        # extract evoked data
        t_sample[iteration, :, :] = t_evokeds[subj]['phase-coherence'].data

    # vectorize channel data
    t_sample = Vectorizer().fit_transform(t_sample)

    # compute t-test for bootstrap sample
    boot_t[ind, :] = ttest_1samp_no_p(t_sample)

# compute low and high percentile for bootstrap t-test
lower_t, upper_t = np.quantile(boot_t, [.025, .975], axis=0)

###############################################################################
# correct p-values for multiple testing and create a mask for non-significant

# extract phase-coherence t-values for each subject
t_phase_coherence = [t_evokeds[subj]['phase-coherence'] for subj in subjects]

t_vals = np.zeros((len(limo_epochs.values()), n_channels, n_times))

for iteration, subject in enumerate(t_phase_coherence):
    # extract evoked data
    t_vals[iteration, :, :] = subject.data

# vectorize channel data
t_vals = Vectorizer().fit_transform(t_vals)
# compute one-sample t-test on phase-coherence t-values extracted from
# each subjects linear regression results
t_vals = ttest_1samp_no_p(t_vals)

# compute p-values from bootstrap t-test
#
# [WIP]: average number of times the T-values obtained from original data
#        are above or below lower_t / upper_t
#
# lower_t = lower_t.reshape((n_channels, n_times))
# upper_t = upper_t.reshape((n_channels, n_times))

# back projection to channels x time points
t_vals = t_vals.reshape((n_channels, n_times))

# create evoked object containing the back projected coefficients
group_t = dict()
group_t['phase-coherence'] = EvokedArray(t_vals, epochs_info, tmin)

# electrode to plot (reverse order to be compatible whit LIMO paper)
picks = group_t['phase-coherence'].ch_names[::-1]
# plot t-values, masking non-significant time points.
fig = group_t['phase-coherence'].plot_image(time_unit='s',
                                            picks=picks,
                                            xlim=(-.1, None),
                                            # mask=mask,
                                            unit=False,
                                            # keep values scale
                                            scalings=dict(eeg=1))
fig.axes[1].set_title('T-value')

# plot topo-map for n170 effect
fig = group_t['phase-coherence'].plot_topomap(times=[.12, .16, .20],
                                              scalings=dict(eeg=1),
                                              sensors=False,
                                              outlines='skirt')
