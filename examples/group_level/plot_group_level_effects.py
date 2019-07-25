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
# run regression analysis for each subject

# variables to be used in the analysis (i.e., predictors)
predictors = ['intercept', 'face a - face b', 'phase-coherence']

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

# create empty array for the storage of results
betas = np.zeros((len(limo_epochs.values()),
                  n_channels * n_times,
                  len(predictors)))

betas_evoked = dict()

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
    # extract betas for each predictor, i.e, columns = len(predictors)
    betas[iteration, :, :] = get_coef(linear_model, 'coef_')

    # thus, we can loop through the columns (i.e., the predictors) of the
    # coefficient matrix and extract coefficients for each predictor and project
    # them back to a channels x time points space.
    lm_betas = dict()
    for ind, predictor in enumerate(predictors):
        # extract coefficients
        beta = betas[iteration, :, ind]
        # back projection to channels x time points
        beta = beta.reshape((n_channels, n_times))
        # create evoked object containing the back projected coefficients
        # for each predictor
        lm_betas[predictor] = EvokedArray(beta, epochs_info, tmin)

    # save results
    betas_evoked[str(subjects[iteration])] = lm_betas

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
# compute bootstrap confidence interval for phase-coherence betas

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
    # extract random epochs from data
    resamples = random.choice(range(betas.shape[0]),
                              betas.shape[0],
                              replace=True)
    # compute centrality estimator on re-sampled data
    boot_betas[i, :] = betas[resamples, :, pred_col].mean(axis=0)

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
